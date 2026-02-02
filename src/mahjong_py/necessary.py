"""进张（必要牌）计算。

给定 34 张计数的手牌与副露数量，返回：
- 最优向听类型（一般形/七对/国士）
- 最小向听数
- 进张列表（哪些牌能使向听数下降或进入和了形）

底层依赖预生成的稠密查表数据，并用 numba 加速热路径。
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .constants import YAOCHUU_TILES
from .hashing import honors_hash, suits_hash
from .shanten import ShantenFlag
from .tables import load_tables


@njit(cache=True)
def _decode_dist_wait(row_u32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = np.empty(10, dtype=np.int32)
    wait = np.empty(10, dtype=np.int64)
    for i in range(10):
        v = row_u32[i]
        dist[i] = int(v & 0xF)
        wait[i] = int((v >> 4) & 0x1FF)
    return dist, wait


@njit(cache=True)
def _shift(best_dist: int, cand_dist: int, best_mask: int, cand_mask: int) -> tuple[int, int]:
    if cand_dist < best_dist:
        return cand_dist, cand_mask
    if cand_dist == best_dist:
        return best_dist, best_mask | cand_mask
    return best_dist, best_mask


@njit(cache=True)
def _add1_dist_mask(lhs_dist: np.ndarray, lhs_mask: np.ndarray, rhs_dist: np.ndarray, rhs_mask: np.ndarray, m: int) -> None:
    for i in range(m + 5, 4, -1):
        best_dist = int(lhs_dist[i] + rhs_dist[0])
        best_mask = int((lhs_mask[i] << 9) | rhs_mask[0])

        cand_dist = int(lhs_dist[0] + rhs_dist[i])
        cand_mask = int((lhs_mask[0] << 9) | rhs_mask[i])
        best_dist, best_mask = _shift(best_dist, cand_dist, best_mask, cand_mask)

        for j in range(5, i):
            cand_dist = int(lhs_dist[j] + rhs_dist[i - j])
            cand_mask = int((lhs_mask[j] << 9) | rhs_mask[i - j])
            best_dist, best_mask = _shift(best_dist, cand_dist, best_mask, cand_mask)

            cand_dist = int(lhs_dist[i - j] + rhs_dist[j])
            cand_mask = int((lhs_mask[i - j] << 9) | rhs_mask[j])
            best_dist, best_mask = _shift(best_dist, cand_dist, best_mask, cand_mask)

        lhs_dist[i] = best_dist
        lhs_mask[i] = best_mask

    for i in range(m, -1, -1):
        best_dist = int(lhs_dist[i] + rhs_dist[0])
        best_mask = int((lhs_mask[i] << 9) | rhs_mask[0])

        for j in range(0, i):
            cand_dist = int(lhs_dist[j] + rhs_dist[i - j])
            cand_mask = int((lhs_mask[j] << 9) | rhs_mask[i - j])
            best_dist, best_mask = _shift(best_dist, cand_dist, best_mask, cand_mask)

        lhs_dist[i] = best_dist
        lhs_mask[i] = best_mask


@njit(cache=True)
def _add2_dist_mask(lhs_dist: np.ndarray, lhs_mask: np.ndarray, rhs_dist: np.ndarray, rhs_mask: np.ndarray, m: int) -> None:
    i = m + 5
    best_dist = int(lhs_dist[i] + rhs_dist[0])
    best_mask = int((lhs_mask[i] << 9) | rhs_mask[0])

    cand_dist = int(lhs_dist[0] + rhs_dist[i])
    cand_mask = int((lhs_mask[0] << 9) | rhs_mask[i])
    best_dist, best_mask = _shift(best_dist, cand_dist, best_mask, cand_mask)

    for j in range(5, i):
        cand_dist = int(lhs_dist[j] + rhs_dist[i - j])
        cand_mask = int((lhs_mask[j] << 9) | rhs_mask[i - j])
        best_dist, best_mask = _shift(best_dist, cand_dist, best_mask, cand_mask)

        cand_dist = int(lhs_dist[i - j] + rhs_dist[j])
        cand_mask = int((lhs_mask[i - j] << 9) | rhs_mask[j])
        best_dist, best_mask = _shift(best_dist, cand_dist, best_mask, cand_mask)

    lhs_dist[i] = best_dist
    lhs_mask[i] = best_mask


@njit(cache=True)
def necessary_regular(hand34: np.ndarray, num_melds: int, suits_raw: np.ndarray, honors_raw: np.ndarray) -> tuple[int, int]:
    manzu_h = suits_hash(hand34, 0)
    pinzu_h = suits_hash(hand34, 9)
    souzu_h = suits_hash(hand34, 18)
    honors_h = honors_hash(hand34)

    m = 4 - num_melds

    # Start from honors, then souzu, pinzu, then manzu (matches C++ packing)
    dist, wait = _decode_dist_wait(honors_raw[honors_h])
    _add1_dist_mask(dist, wait, *_decode_dist_wait(suits_raw[souzu_h]), m)
    _add1_dist_mask(dist, wait, *_decode_dist_wait(suits_raw[pinzu_h]), m)
    _add2_dist_mask(dist, wait, *_decode_dist_wait(suits_raw[manzu_h]), m)

    shanten = int(dist[5 + m] - 1)
    mask = int(wait[5 + m])
    return shanten, mask


def necessary_seven_pairs(hand34: np.ndarray) -> tuple[int, int]:
    """七对子形的向听与进张位图。

    返回：
    - shanten: 七对向听数
    - wait: 进张位图（bit i 表示牌 i 是进张）
    """
    num_pairs = 0
    num_types = 0
    count0 = 0
    count1 = 0

    for i in range(34):
        c = int(hand34[i])
        if c == 0:
            count0 |= 1 << i
        elif c == 1:
            num_types += 1
            count1 |= 1 << i
        else:
            num_pairs += 1
            num_types += 1

    shanten = 6 - num_pairs + max(0, 7 - num_types)
    if num_types < 7:
        wait = count0 | count1
    elif num_pairs == 7:
        wait = 0
    else:
        wait = count1

    return shanten, int(wait)


def necessary_thirteen_orphans(hand34: np.ndarray) -> tuple[int, int]:
    """国士无双形的向听与进张位图。"""
    num_pairs = 0
    num_types = 0
    count0 = 0
    count1 = 0

    for t in YAOCHUU_TILES:
        c = int(hand34[t])
        if c == 0:
            count0 |= 1 << t
        elif c == 1:
            count1 |= 1 << t
            num_types += 1
        else:
            num_types += 1
            num_pairs += 1

    shanten = 13 - num_types - (1 if num_pairs else 0)
    wait = count0 if num_pairs else (count0 | count1)
    return shanten, int(wait)


def necessary_tiles(hand34: np.ndarray, num_melds: int = 0, type_mask: int = ShantenFlag.All) -> tuple[int, int, list[int]]:
    """计算必要牌（进张）。

    参数：
        hand34: 长度 34 的计数数组（不含赤宝牌 flag）
        num_melds: 副露数量（用来调整一般形的面子数量）
        type_mask: 计算哪些向听类型（一般形/七对/国士）的组合

    返回：
        (best_type, best_shanten, tiles)
        - best_type: 可能包含多个 ShantenFlag（若并列最优）
        - best_shanten: 最小向听数
        - tiles: 进张牌列表（0..33）
    """
    suits_raw, honors_raw = load_tables()

    best_type = 0
    best_shanten = 10**9
    best_mask = 0

    if type_mask & ShantenFlag.Regular:
        s, m = necessary_regular(hand34, num_melds, suits_raw, honors_raw)
        if s < best_shanten:
            best_shanten, best_type, best_mask = s, ShantenFlag.Regular, m
        elif s == best_shanten:
            best_type |= ShantenFlag.Regular
            best_mask |= m

    if (type_mask & ShantenFlag.SevenPairs) and num_melds == 0:
        s, m = necessary_seven_pairs(hand34)
        if s < best_shanten:
            best_shanten, best_type, best_mask = s, ShantenFlag.SevenPairs, m
        elif s == best_shanten:
            best_type |= ShantenFlag.SevenPairs
            best_mask |= m

    if (type_mask & ShantenFlag.ThirteenOrphans) and num_melds == 0:
        s, m = necessary_thirteen_orphans(hand34)
        if s < best_shanten:
            best_shanten, best_type, best_mask = s, ShantenFlag.ThirteenOrphans, m
        elif s == best_shanten:
            best_type |= ShantenFlag.ThirteenOrphans
            best_mask |= m

    tiles = [i for i in range(34) if (best_mask >> i) & 1]
    return best_type, int(best_shanten), tiles
