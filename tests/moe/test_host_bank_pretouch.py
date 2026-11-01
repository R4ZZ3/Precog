"""--pretouch-host-banks: an mmap bank is resident right after allocation, and only when asked.

CPU-only (Linux): reads the bank mapping's Rss from /proc/self/smaps.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe import host_banks
from freetoken.moe.host_banks import HostBank, set_pretouch

NBYTES = 8 << 20


def _rss_kib(addr: int) -> int:
    inside = False
    with open("/proc/self/smaps") as smaps:
        for line in smaps:
            head = line.split()[0]
            if "-" in head and ":" not in head:
                start, end = (int(x, 16) for x in head.split("-"))
                inside = start <= addr < end
            elif inside and head == "Rss:":
                return int(line.split()[1])
    raise AssertionError("bank mapping not found")


@pytest.fixture(autouse=True)
def _restore():
    yield
    set_pretouch(False)


def test_default_bank_is_lazy():
    bank = HostBank((NBYTES,), torch.uint8, backing="mmap")
    assert _rss_kib(bank.addr) < NBYTES // 1024 // 4


def test_pretouched_bank_is_resident_and_zero():
    set_pretouch(True)
    bank = HostBank((NBYTES,), torch.uint8, backing="mmap")
    assert _rss_kib(bank.addr) >= NBYTES // 1024
    assert int(bank.tensor.count_nonzero()) == 0


def test_setting_is_module_state():
    set_pretouch(True)
    assert host_banks._pretouch is True
    set_pretouch(False)
    assert host_banks._pretouch is False
