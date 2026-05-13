import torch
import torch.nn.functional as F
from .cache import KVCache, run_forward
from .config import DEFAULT_DRAFT_K, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from dataclasses import dataclass
from typing import List

@dataclass
class SamplingResult:
    accepted_tokens: List[int]
    bonus_token: int
    num_accepted: int
    acceptance_rate: float


def get_probs(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature == 0:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs

    scaled = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = (cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p)
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    filtered_logits = torch.full_like(scaled, float("-inf"))
    filtered_logits[sorted_indices] = sorted_logits
    return F.softmax(filtered_logits, dim=-1)


def draft_step(draft_model, input_ids: torch.Tensor, draft_cache: KVCache, k: int = DEFAULT_DRAFT_K, temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P, device: str = "cuda",) -> tuple[list[int], list[torch.Tensor], KVCache]:
    draft_tokens: list[int] = []
    draft_probs: list[torch.Tensor] = []
    current_ids = input_ids[:, -1:] if not draft_cache.is_empty() else input_ids
    for _ in range(k):
        logits, draft_cache = run_forward(draft_model, current_ids, draft_cache)
        step_logits = logits[0, -1, :]
        probs = get_probs(step_logits, temperature, top_p)
        next_token = torch.multinomial(probs, num_samples=1).item()
        draft_tokens.append(next_token)
        draft_probs.append(probs)
        current_ids = torch.tensor([[next_token]], device=logits.device)
    return draft_tokens, draft_probs, draft_cache


def verify_step(target_model, input_ids: torch.Tensor, draft_tokens: list[int], target_cache: KVCache,) -> tuple[torch.Tensor, KVCache]:
    if target_cache.is_empty():
        verify_ids = torch.cat([input_ids, torch.tensor([draft_tokens], device=input_ids.device)], dim=1)
    else:
        last_token = input_ids[:, -1:]
        draft_tensor = torch.tensor([draft_tokens], device=input_ids.device)
        verify_ids = torch.cat([last_token, draft_tensor], dim=1)

    logits, updated_cache = run_forward(target_model, verify_ids, target_cache)
    k = len(draft_tokens)
    relevant_logits = logits[:, -(k + 1) :, :]

    return relevant_logits, updated_cache


def _residual_sample(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> int:
    residual = torch.clamp(target_probs - _align_probs(draft_probs, target_probs.shape[0]), min=0.0)
    total = residual.sum()

    if total < 1e-8:
        return torch.multinomial(target_probs, num_samples=1).item()

    residual /= total
    return torch.multinomial(residual, num_samples=1).item()


def _align_probs(probs: torch.Tensor, target_size: int) -> torch.Tensor:
    current_size = probs.shape[0]
    if current_size == target_size:
        return probs
    if current_size > target_size:
        return probs[:target_size]
    padded = torch.zeros(target_size, device=probs.device, dtype=probs.dtype)
    padded[:current_size] = probs
    return padded


def rejection_sample(draft_tokens: list[int], draft_probs: list[torch.Tensor], target_logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9,) -> SamplingResult:
    k = len(draft_tokens)
    accepted_tokens: list[int] = []
    accepted_count = 0

    for i in range(k):
        token = draft_tokens[i]
        q_probs = draft_probs[i]
        t_logits = target_logits[0, i, :]
        p_probs = get_probs(t_logits, temperature, top_p)
        q_probs_aligned = _align_probs(q_probs, p_probs.shape[0])
        p_token = p_probs[token].item()
        q_token = q_probs_aligned[token].item()
        accept_prob = min(1.0, p_token / (q_token + 1e-8))
        rand_val = torch.rand(1).item()

        if rand_val < accept_prob:
            accepted_tokens.append(token)
            accepted_count += 1
        else:
            corrected = _residual_sample(p_probs, q_probs)
            accepted_tokens.append(corrected)
            return SamplingResult(accepted_tokens=accepted_tokens, bonus_token=corrected, num_accepted=accepted_count, acceptance_rate=accepted_count / k,)

    bonus_logits = target_logits[0, k, :]
    bonus_probs = get_probs(bonus_logits, temperature, top_p)
    bonus_token = torch.multinomial(bonus_probs, num_samples=1).item()

    return SamplingResult(accepted_tokens=accepted_tokens, bonus_token=bonus_token, num_accepted=accepted_count, acceptance_rate=1.0,)
