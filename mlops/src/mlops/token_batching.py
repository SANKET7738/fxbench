from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from torch.utils.data import Sampler


@dataclass(frozen=True)
class TokenBudgetConfig:
    max_tokens_per_device: int
    max_examples_per_device: int
    length_column_name: str = "length"
    megabatch_mult: int = 50
    drop_last: bool = False


def _length_grouped_order(
    lengths: Sequence[int],
    group_size: int,
    seed: int,
) -> list[int]:
    indices = list(range(len(lengths)))
    random.Random(seed).shuffle(indices)

    if group_size <= 0:
        group_size = len(indices) or 1

    ordered: list[int] = []
    for start in range(0, len(indices), group_size):
        chunk = indices[start : start + group_size]
        chunk.sort(key=lengths.__getitem__, reverse=True)
        ordered.extend(chunk)
    return ordered


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Batch sampler that budgets batches by padded-token cost.

    For a local batch with example lengths L, the budget metric is:

        max(L) * len(L)

    which better matches the memory behavior of dynamically padded training
    batches than a raw token sum.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        rank: int,
        world_size: int,
        config: TokenBudgetConfig,
        seed: int,
    ) -> None:
        if config.max_tokens_per_device <= 0:
            raise ValueError("max_tokens_per_device must be positive")
        if config.max_examples_per_device <= 0:
            raise ValueError("max_examples_per_device must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"rank must be in [0, {world_size}), got rank={rank}"
            )

        normalized_lengths = [int(length) for length in lengths]
        if any(length <= 0 for length in normalized_lengths):
            raise ValueError("all batch lengths must be positive integers")
        if normalized_lengths and max(normalized_lengths) > config.max_tokens_per_device:
            raise ValueError(
                "max_tokens_per_device is smaller than the longest example length; "
                "token-budget batching cannot fit the worst-case example"
            )

        self.lengths = tuple(normalized_lengths)
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.seed = int(seed)
        self.epoch = 0
        self._cached_epoch: int | None = None
        self._cached_local_batches: list[list[int]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cached_epoch = None
        self._cached_local_batches = None

    def _build_local_batches(self) -> list[list[int]]:
        if self._cached_epoch == self.epoch and self._cached_local_batches is not None:
            return self._cached_local_batches

        megabatch_size = max(
            1,
            self.config.max_examples_per_device
            * self.world_size
            * self.config.megabatch_mult,
        )
        ordered_indices = _length_grouped_order(
            self.lengths,
            group_size=megabatch_size,
            seed=self.seed + self.epoch,
        )

        all_local_batches: list[list[int]] = []
        current_batch: list[int] = []
        current_max_length = 0

        for idx in ordered_indices:
            length = self.lengths[idx]
            next_batch_size = len(current_batch) + 1
            next_max_length = length if not current_batch else max(current_max_length, length)
            next_padded_tokens = next_max_length * next_batch_size

            if current_batch and (
                next_batch_size > self.config.max_examples_per_device
                or next_padded_tokens > self.config.max_tokens_per_device
            ):
                all_local_batches.append(current_batch)
                current_batch = [idx]
                current_max_length = length
            else:
                current_batch.append(idx)
                current_max_length = next_max_length

        if current_batch:
            all_local_batches.append(current_batch)

        remainder = len(all_local_batches) % self.world_size
        if remainder:
            if self.config.drop_last:
                all_local_batches = all_local_batches[:-remainder]
            elif all_local_batches:
                needed = self.world_size - remainder
                while needed > 0:
                    take = min(needed, len(all_local_batches))
                    all_local_batches.extend(all_local_batches[:take])
                    needed -= take

        local_batches = all_local_batches[self.rank :: self.world_size]
        self._cached_epoch = self.epoch
        self._cached_local_batches = local_batches
        return local_batches

    def __iter__(self):
        for batch in self._build_local_batches():
            yield batch

    def __len__(self) -> int:
        return len(self._build_local_batches())

    @property
    def sampler(self) -> "TokenBudgetBatchSampler":
        # Expose the sampler through the conventional nested attribute name so
        # Accelerate's resume wrappers can still forward set_epoch() to us.
        return self
