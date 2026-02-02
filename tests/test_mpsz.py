from __future__ import annotations

import numpy as np

from mahjong_py.mpsz import from_mpsz, to_hand34, to_mpsz


def _count_tiles(hand37: np.ndarray) -> int:
    return int(hand37[:34].sum())


def test_roundtrip_without_red():
    s = "123m789p123s77z"
    h37 = from_mpsz(s)
    assert _count_tiles(h37) == 11
    assert to_mpsz(h37) == s


def test_roundtrip_with_reds():
    s = "0m123m0p789p0s123s77z"
    h37 = from_mpsz(s)
    assert _count_tiles(h37) == 14
    # to_mpsz groups each suit once, so reds are emitted as '0' within suit digits
    assert to_mpsz(h37) == "0123m0789p0123s77z"


def test_to_hand34_merges_reds():
    s = "0m123m0p789p0s123s77z"
    h37 = from_mpsz(s)
    h34 = to_hand34(h37)
    assert h34.shape[0] == 34
    assert int(h34[4]) >= 1  # 5m includes red
    assert int(h34[13]) >= 1  # 5p includes red
    assert int(h34[22]) >= 1  # 5s includes red
