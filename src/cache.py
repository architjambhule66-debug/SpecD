from dataclasses import dataclass
from typing import Any, Optional
import torch

@dataclass
class KVCache:
    past_key_values: Optional[Any] = None
    seq_len: int = 0

    def is_empty(self) -> bool:
        return self.past_key_values is None

    def rollback(self, keep_len: int):
        if self.past_key_values is None:
            return

        if hasattr(self.past_key_values, "crop"):
            self.past_key_values.crop(keep_len)
        else:
            trimmed = []
            for layer_kv in self.past_key_values:
                k = layer_kv[0][:, :, :keep_len, :]
                v = layer_kv[1][:, :, :keep_len, :]
                trimmed.append((k, v))
            self.past_key_values = tuple(trimmed)

        self.seq_len = keep_len

    def update(self, new_past_key_values, new_seq_len: int):
        self.past_key_values = new_past_key_values
        self.seq_len = new_seq_len


def run_forward(model, input_ids: torch.Tensor, cache: KVCache, use_cache: bool = True,) -> tuple[torch.Tensor, KVCache]:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, past_key_values=cache.past_key_values, use_cache=use_cache,)

    new_past_key_values = outputs.past_key_values
    if hasattr(new_past_key_values, "get_seq_length"):
        new_seq_len = new_past_key_values.get_seq_length()
    else:
        new_seq_len = cache.seq_len + input_ids.shape[1]

    updated_cache = KVCache(past_key_values=new_past_key_values, seq_len=new_seq_len,)
    return outputs.logits, updated_cache
