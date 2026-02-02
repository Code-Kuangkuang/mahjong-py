"""向听数计算。

提供：
- 一般形（面子手）向听
- 七对子向听
- 国士无双向听

一般形使用预生成查表 + numba 加速。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .constants import YAOCHUU_TILES
from .hashing import honors_hash, suits_hash
from .tables import load_tables


class ShantenFlag:
    """向听类型掩码。"""
    Regular = 1
    SevenPairs = 2
    ThirteenOrphans = 4
    All = 7


@njit(cache=True)
def _decode_dist(row_u32: np.ndarray) -> np.ndarray:
    dist = np.empty(10, dtype=np.int32)
    for i in range(10):
        dist[i] = int(row_u32[i] & 0xF)
    return dist


@njit(cache=True)
def _add1_dist(lhs: np.ndarray, rhs: np.ndarray, m: int) -> None:
    for i in range(m + 5, 4, -1):
        # 有雀头的最快进张
        best = lhs[i] + rhs[0]
        alt = lhs[0] + rhs[i]
        if alt < best:
            best = alt

        for j in range(5, i):
            v1 = lhs[j] + rhs[i - j]
            if v1 < best:
                best = v1
            v2 = lhs[i - j] + rhs[j]
            if v2 < best:
                best = v2

        lhs[i] = best

    for i in range(m, -1, -1):
        # 无雀头的最快进张
        best = lhs[i] + rhs[0]
        for j in range(0, i):
            v = lhs[j] + rhs[i - j]
            if v < best:
                best = v
        lhs[i] = best


@njit(cache=True)
def _add2_dist(lhs: np.ndarray, rhs: np.ndarray, m: int) -> None:
    i = m + 5
    best = lhs[i] + rhs[0]
    alt = lhs[0] + rhs[i]
    if alt < best:
        best = alt

    for j in range(5, i):
        v1 = lhs[j] + rhs[i - j]
        if v1 < best:
            best = v1
        v2 = lhs[i - j] + rhs[j]
        if v2 < best:
            best = v2

    lhs[i] = best


@njit(cache=True)
def shanten_regular(hand34: np.ndarray, num_melds: int, suits_raw: np.ndarray, honors_raw: np.ndarray) -> int:
    manzu_h = suits_hash(hand34, 0)
    pinzu_h = suits_hash(hand34, 9)
    souzu_h = suits_hash(hand34, 18)
    honors_h = honors_hash(hand34)

    m = 4 - num_melds

    lhs = _decode_dist(suits_raw[manzu_h])
    _add1_dist(lhs, _decode_dist(suits_raw[pinzu_h]), m)
    _add1_dist(lhs, _decode_dist(suits_raw[souzu_h]), m)
    _add2_dist(lhs, _decode_dist(honors_raw[honors_h]), m)

    return int(lhs[5 + m] - 1)


def shanten_seven_pairs(hand34: np.ndarray) -> int:
    """计算七对子向听数（仅在无副露时有意义）。"""
    num_types = 0
    num_pairs = 0
    for i in range(34):
        c = int(hand34[i])
        if c > 0:
            num_types += 1
        if c >= 2:
            num_pairs += 1
    return 6 - num_pairs + max(0, 7 - num_types)


def shanten_thirteen_orphans(hand34: np.ndarray) -> int:
    """计算国士无双向听数（仅在无副露时有意义）。"""
    num_types = 0
    has_pair = False
    for t in YAOCHUU_TILES:
        c = int(hand34[t])
        if c > 0:
            num_types += 1
        if c >= 2:
            has_pair = True
    return 13 - num_types - (1 if has_pair else 0)


def shanten(hand34: np.ndarray, num_melds: int = 0, type_mask: int = ShantenFlag.All) -> tuple[int, int]:
    """计算向听数。

    参数：
        hand34: 长度 34 的计数数组（牌 0..33）
        num_melds: 副露数量
        type_mask: 计算类型掩码（可选一般形/七对/国士）

    返回：
        (type_flag, shanten_number)
        - type_flag: 最优类型（可能为多种并列的按位或）
        - shanten_number: 最小向听数（和了形为 -1）
    """
    suits_raw, honors_raw = load_tables()

    best_type = 0
    best = 10**9

    if type_mask & ShantenFlag.Regular:
        s = shanten_regular(hand34, num_melds, suits_raw, honors_raw)
        if s < best:
            best = s
            best_type = ShantenFlag.Regular
        elif s == best:
            best_type |= ShantenFlag.Regular

    if (type_mask & ShantenFlag.SevenPairs) and num_melds == 0:
        s = shanten_seven_pairs(hand34)
        if s < best:
            best = s
            best_type = ShantenFlag.SevenPairs
        elif s == best:
            best_type |= ShantenFlag.SevenPairs

    if (type_mask & ShantenFlag.ThirteenOrphans) and num_melds == 0:
        s = shanten_thirteen_orphans(hand34)
        if s < best:
            best = s
            best_type = ShantenFlag.ThirteenOrphans
        elif s == best:
            best_type |= ShantenFlag.ThirteenOrphans

    return best_type, int(best)
