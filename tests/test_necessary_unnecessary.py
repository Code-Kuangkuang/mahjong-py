from __future__ import annotations

from pathlib import Path

import numpy as np

from mahjong_py.utils import to_no_reddora
from mahjong_py.necessary import necessary_tiles
from mahjong_py.shanten import ShantenFlag, shanten
from mahjong_py.unnecessary import unnecessary_tiles


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cpp_testcase(path_name: str) -> Path:
    return Path(__file__).resolve().parent / "testcase" / path_name


def _hand_from_tiles(tiles: list[int]) -> np.ndarray:
    hand = np.zeros(34, dtype=np.int16)
    for t in tiles:
        hand[to_no_reddora(t)] += 1
    return hand


def test_unnecessary_regular_matches_bruteforce_subset():
    fp = _cpp_testcase("test_unnecessary_tile_calculator.txt")
    # keep runtime reasonable in pytest
    limit = 2000

    with fp.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            tiles = [int(x) for x in line.split()[:14]]
            hand = _hand_from_tiles(tiles)

            _, base = shanten(hand, 0, ShantenFlag.All)

            brute = []
            for t in range(34):
                if hand[t] <= 0:
                    continue
                hand[t] -= 1
                _, after = shanten(hand, 0, ShantenFlag.All)
                if after == base:
                    brute.append(t)
                hand[t] += 1

            _, s2, tiles2 = unnecessary_tiles(hand, 0, ShantenFlag.All)
            assert s2 == base
            assert tiles2 == brute


def test_necessary_regular_matches_bruteforce_subset():
    fp = _cpp_testcase("test_unnecessary_tile_calculator.txt")
    limit = 2000

    with fp.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            # mimic C++ test: read only 13 tiles
            tiles = [int(x) for x in line.split()[:13]]
            hand = _hand_from_tiles(tiles)

            _, base = shanten(hand, 0, ShantenFlag.All)

            brute = []
            for t in range(34):
                if hand[t] >= 4:
                    continue
                hand[t] += 1
                _, after = shanten(hand, 0, ShantenFlag.All)
                if after < base:
                    brute.append(t)
                hand[t] -= 1

            _, s2, tiles2 = necessary_tiles(hand, 0, ShantenFlag.All)
            assert s2 == base
            assert tiles2 == brute
