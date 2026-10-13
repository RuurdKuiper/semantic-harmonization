"""Local MLX reviewer for Apple silicon; no API calls or model substitution."""
from functools import lru_cache
import re
from time import perf_counter


@lru_cache(maxsize=1)
def load_mlx_reviewer(model_name):
    from mlx_vlm import load
    from src.llm.model_paths import prepared_model_path
    return load(prepared_model_path(model_name))


def call_mlx_json(system_prompt, user_prompt, model_name, max_tokens=8192, reasoning_effort='medium',
                  thinking_budget=1024):
    from mlx_vlm import stream_generate
    from src.llm.client import extract_json
    model, processor = load_mlx_reviewer(model_name)
    messages = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=reasoning_effort not in (None,'none'))
    chunks = []; started = perf_counter(); reported = 0
    for output in stream_generate(model, processor, prompt=prompt, max_tokens=max_tokens,
                                   temperature=.6, top_p=.95, top_k=20, seed=42,
                                   enable_thinking=reasoning_effort not in (None,'none'),
                                   thinking_budget=thinking_budget):
        chunks.append(output.text)
        if output.generation_tokens >= reported+256:
            reported = output.generation_tokens
            print(f'[Qwen reviewer] {reported} tokens generated, {perf_counter()-started:.1f}s',flush=True)
    text = ''.join(chunks)
    # Thinking text can contain braces; parse only the visible final response.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    if '</think>' in text:
        text = text.split('</think>',1)[1]
    return extract_json(text)
