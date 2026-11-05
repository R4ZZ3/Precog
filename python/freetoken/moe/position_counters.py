"""Cumulative counters per decode step (FREETOKEN_POSITION_COUNTERS, EXP605, untimed serves only).

Both decode hook classes use it: PrefetchHooks (predictor arms) and StripedDemandHooks (no predictor). With the flag set to a
folder, the hooks write one file per request, positions_NNNN.pt, in request order. The file holds a (steps + 1, 3) int64 tensor:

  row 0       the counters before decode step 0 (after the last prefill chunk)
  row s + 1   the counters after decode step s

Columns: COLUMNS. A column is cumulative over the request, so the counts of decode step s are row s + 1 minus row s, and the
prefill's misses stay in row 0. The file also holds the request counters that the hook read at the end of the request ('end'),
so that a reader can check the last row against them.

Each snapshot is a device-to-device copy on the current (main) stream, after the decode step's graph joined its side streams,
so it needs no host synchronization. Only the write at the end of a request synchronizes. The copies add a few small kernel
launches per decode step, which is why timed serves never set the flag.
"""
from __future__ import annotations

from pathlib import Path

import torch

COLUMNS = ('published', 'from_stage', 'from_host')
# One row per decode step plus row 0. 8448 is the --max-seq-len-override of every EXP598-EXP605 serve, so no request of those
# serves can take more decode steps; a longer request stops the serve instead of writing a short file.
MAX_ROWS = 8448 + 1


class PositionCounters:
    def __init__(self, folder: Path, device: torch.device, max_rows: int = MAX_ROWS):
        if max_rows < 2:
            raise ValueError(f'max_rows {max_rows}: needs row 0 and at least one decode step')
        folder.mkdir(parents=False, exist_ok=False)   # a second serve into the same folder fails at once
        self.folder, self.max_rows = folder, max_rows
        self.rows = torch.zeros((max_rows, len(COLUMNS)), dtype=torch.int64, device=device)
        self.request = -1   # ordinal of the requests that reached decode step 0: the file index
        self.used = 0

    def snap(self, row: int, published: torch.Tensor | None, misses: torch.Tensor) -> None:
        """Copy the cumulative counters into ``row``. ``published``: a one-element tensor, or None for an arm without copies.
        ``misses``: (from_stage, from_host), summed over layers. Row 0 starts a request."""
        if not 0 <= row < self.max_rows:
            raise RuntimeError(f'position counters: row {row} is outside 0..{self.max_rows - 1}')
        if row == 0:
            self.request += 1
        elif row != self.used:
            raise RuntimeError(f'position counters: row {row} after {self.used} rows (a decode step was skipped)')
        target = self.rows[row]
        if published is None:
            target[0].zero_()
        else:
            target[0].copy_(published.reshape(()))
        target[1:].copy_(misses.reshape(2))
        self.used = row + 1

    def write(self, uid: int, steps: int, end: list[int]) -> Path:
        """Write the finished request. ``steps``: its decode steps; ``end``: the request counters in COLUMNS order."""
        if self.used != steps + 1:
            raise RuntimeError(f'position counters: {self.used} rows for {steps} decode steps of request {uid}')
        if len(end) != len(COLUMNS):
            raise ValueError(f'end has {len(end)} values, expected {len(COLUMNS)}')
        path = self.folder / f'positions_{self.request:04d}.pt'
        tmp = path.with_suffix('.tmp')
        torch.save(dict(uid=int(uid), steps=int(steps), columns=COLUMNS, rows=self.rows[:self.used].cpu(),
                        end=torch.tensor(end, dtype=torch.int64)), tmp)
        tmp.rename(path)
        self.used = 0
        return path


def folder_from_env(env, name: str = 'FREETOKEN_POSITION_COUNTERS') -> Path | None:
    value = env.get(name)
    return Path(value) if value else None
