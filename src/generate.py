import torch
from rich.console import Console
from rich.live import Live
from rich.text import Text
from .config import DEFAULT_DRAFT_K, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from .cache import KVCache, run_forward
from .draft import draft_step, verify_step, rejection_sample
from .tracker import GenerationMetrics

console = Console()

def speculative_generate(tokenizer, draft_model, target_model, prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, k: int = DEFAULT_DRAFT_K, temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P, stream: bool = True,) -> tuple[str, GenerationMetrics]:
    device = next(target_model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    generated_ids: list[int] = []
    draft_cache = KVCache()
    target_cache = KVCache()
    metrics = GenerationMetrics(prompt_tokens=prompt_len)
    _prefill(draft_model, target_model, input_ids, draft_cache, target_cache)
    all_ids = input_ids
    output_text = ""

    with Live(console=console, refresh_per_second=20) as live:
        while len(generated_ids) < max_new_tokens:
            remaining = max_new_tokens - len(generated_ids)
            current_k = min(k, remaining)

            draft_tokens, draft_probs, draft_cache = draft_step(draft_model, all_ids, draft_cache, k=current_k, temperature=temperature, top_p=top_p,)
            target_logits, updated_target_cache = verify_step(target_model, all_ids, draft_tokens, target_cache,)
            result = rejection_sample(draft_tokens, draft_probs, target_logits, temperature=temperature, top_p=top_p,)
            new_tokens = result.accepted_tokens.copy()
            if result.num_accepted == current_k:
                new_tokens.append(result.bonus_token)
            eos_id = tokenizer.eos_token_id
            if eos_id in new_tokens:
                eos_pos = new_tokens.index(eos_id)
                new_tokens = new_tokens[:eos_pos]
                generated_ids.extend(new_tokens)
                break

            generated_ids.extend(new_tokens)

            accepted_len = result.num_accepted + 1
            updated_target_cache.rollback(updated_target_cache.seq_len - (current_k + 1 - accepted_len))
            target_cache = updated_target_cache
            if result.num_accepted < current_k:
                draft_cache.rollback(draft_cache.seq_len - (current_k - result.num_accepted))

            new_ids_tensor = torch.tensor([new_tokens], device=device)
            all_ids = torch.cat([all_ids, new_ids_tensor], dim=1)

            metrics.record_cycle(num_accepted=result.num_accepted, k=current_k, tokens_produced=len(new_tokens),)
            if stream:
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                live.update(Text(f"[generating...]\n\n{output_text}", style="dim"))

    metrics.finish()
    final_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return final_text, metrics


def vanilla_generate(tokenizer, target_model, prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P,) -> tuple[str, GenerationMetrics]:
    device = next(target_model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    metrics = GenerationMetrics(prompt_tokens=inputs["input_ids"].shape[1])
    with torch.no_grad():
        output_ids = target_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=(temperature > 0), temperature=temperature, top_p=top_p,)
    metrics.total_tokens_generated = output_ids.shape[1] - inputs["input_ids"].shape[1]
    metrics.finish()

    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, metrics


def _prefill(draft_model, target_model, input_ids, draft_cache, target_cache):
    with torch.no_grad():
        _, new_draft_cache = run_forward(draft_model, input_ids, draft_cache)
        _, new_target_cache = run_forward(target_model, input_ids, target_cache)

    draft_cache.past_key_values = new_draft_cache.past_key_values
    draft_cache.seq_len = new_draft_cache.seq_len
    target_cache.past_key_values = new_target_cache.past_key_values
    target_cache.seq_len = new_target_cache.seq_len
