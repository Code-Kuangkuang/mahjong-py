from __future__ import annotations

from pathlib import Path

import numpy as np

from mahjong_py.utils import to_no_reddora
from mahjong_py.shanten import ShantenFlag, shanten, shanten_regular, shanten_seven_pairs, shanten_thirteen_orphans
from mahjong_py.tables import load_tables


def _repo_root() -> Path:
    # .../mahjong-py/tests/test_shanten.py -> parents[1] is mahjong-py
    return Path(__file__).resolve().parents[1]


def _cpp_testcase(path_name: str) -> Path:
    return Path(__file__).resolve().parent / "testcase" / path_name


def _hand_from_tiles(tiles: list[int]) -> np.ndarray:
    hand = np.zeros(34, dtype=np.int16)
    for t in tiles:
        hand[to_no_reddora(t)] += 1
    return hand


def test_shanten_matches_gold_file_regular_sevenpairs_orphans():
    suits_raw, honors_raw = load_tables()  # ensure tables exist

    fp = _cpp_testcase("test_shanten_calculator.txt")
    with fp.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            tiles = [int(x) for x in parts[:14]]
            reg = int(parts[14])
            orph = int(parts[15])
            seven = int(parts[16])

            hand = _hand_from_tiles(tiles)

            assert shanten_regular(hand, 0, suits_raw, honors_raw) == reg, f"line {line_no}"
            assert shanten_thirteen_orphans(hand) == orph, f"line {line_no}"
            assert shanten_seven_pairs(hand) == seven, f"line {line_no}"

            best = min(reg, orph, seven)
            best_type = 0
            if reg == best:
                best_type |= ShantenFlag.Regular
            if orph == best:
                best_type |= ShantenFlag.ThirteenOrphans
            if seven == best:
                best_type |= ShantenFlag.SevenPairs

            t, s = shanten(hand, 0, ShantenFlag.All)
            assert s == best, f"line {line_no}"
            assert t == best_type, f"line {line_no}"
