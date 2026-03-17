import asyncio
import httpx
import json
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


async def _cancel_tasks(tasks):
    """Cancel a list of asyncio Tasks and await them (swallow CancelledError)."""
    if not tasks:
        return
    for t in tasks:
        if not t.done():
            t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _append_trace(trace_output_file, text):
    if not trace_output_file:
        return
    with open(trace_output_file, "a", encoding="utf-8") as f:
        f.write(text)


def _format_intervention_markup(original_text, replacement_text):
    """Return a compact word-level diff with HTML strike markers for removed text."""
    original_words = original_text.split()
    replacement_words = replacement_text.split()
    matcher = SequenceMatcher(a=original_words, b=replacement_words)
    parts = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            parts.extend(original_words[i1:i2])
        elif op in ("replace", "delete"):
            removed = " ".join(original_words[i1:i2])
            if removed:
                parts.append(f"<strike>{removed}</strike>")
        if op in ("replace", "insert"):
            added = " ".join(replacement_words[j1:j2])
            if added:
                parts.append(f"[{added}]")
    return " ".join(parts)


async def stream_completion(
    prompt,
    prev_text="",
    llm_server=None,
    monitors=[],
    add_delay=False,
    num_calls_index=0,
    termination_requires_validation=False,
    async_execution=True,
    trace_output_file=None,
):
    stop_event = asyncio.Event()
    stop_info = {"generated_text": None, "feedback": None, "token_index": None}
    monitor_tasks = []

    logger.warning("=" * 50 + f"Calling LM with prompt (call #{num_calls_index})" + "=" * 50)
    generated_text = prev_text
    llm_server["payload"]["prompt"] = prompt + prev_text

    if num_calls_index == 0 and trace_output_file:
        _append_trace(trace_output_file, "\n=== interwhen run start ===\n")

    logger.info(f"#{num_calls_index}: EOS")
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            llm_server["url"],
            headers=llm_server["headers"],
            json=llm_server["payload"],
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[len("data: ") :].strip()
                    if data == "[DONE]":
                        break
                    # Obtain the current token (text chunk)
                    chunk = json.loads(data)["choices"][0]["text"]
                    # If any event is already set, break immediately (we don't want more chunks)
                    if stop_event.is_set():
                        logger.info(f"\n[Early stop already triggered, ignoring chunk: {chunk}]")
                        break
                    print(chunk, end="", flush=True)
                    _append_trace(trace_output_file, chunk)
                    generated_text += chunk

                    # Start monitor task in background with chunk index
                    if len(monitors) > 0 and not stop_event.is_set():
                        stepFlag, step = monitors[0].step_extractor(chunk, generated_text)
                        if stepFlag and not stop_event.is_set():
                            task = asyncio.create_task(
                                monitors[0].verify(
                                    step,
                                    len(generated_text) - len(chunk),
                                    stop_event,
                                    stop_info,
                                )
                            )
                            monitor_tasks.append(task)
                            if not async_execution:
                                await task
                    if add_delay:
                        await asyncio.sleep(0.1)

    # If any monitor event fired, cancel remaining monitor tasks right away.
    if len(monitors) > 0 and async_execution:
        if stop_event.is_set():
            logger.debug("Monitor event detected - cancelling pending monitor tasks immediately.")
            await _cancel_tasks(monitor_tasks)
        else:
            await asyncio.gather(*monitor_tasks, return_exceptions=True)

    if stop_event.is_set():
        if num_calls_index >= 50:
            logger.info("\n\\n[Maximum correction attempts reached. Stopping generation.]")
            return generated_text

        original_chunk = stop_info.get("generated_text") or ""
        replacement_chunk = stop_info.get("feedback") or ""
        correction_index = stop_info.get("correction_index")
        if trace_output_file:
            markup = _format_intervention_markup(original_chunk, replacement_chunk)
            _append_trace(
                trace_output_file,
                (
                    "\n\n[INTERVENTION]\n"
                    f"call={num_calls_index} index={correction_index}\n"
                    f"before={original_chunk}\n"
                    f"after={replacement_chunk}\n"
                    f"markup={markup}\n"
                    "[/INTERVENTION]\n"
                ),
            )

        corrected_text = await monitors[0].fix(generated_text, stop_info)
        if stop_info["feedback"] == "\nthe answer is \\boxed{no solution}":
            return corrected_text
        return await stream_completion(
            prompt,
            prev_text=corrected_text,
            llm_server=llm_server,
            monitors=monitors,
            add_delay=add_delay,
            num_calls_index=num_calls_index + 1,
            termination_requires_validation=termination_requires_validation,
            async_execution=async_execution,
            trace_output_file=trace_output_file,
        )

    if trace_output_file and num_calls_index == 0:
        _append_trace(trace_output_file, "\n=== interwhen run end ===\n")

    return generated_text
