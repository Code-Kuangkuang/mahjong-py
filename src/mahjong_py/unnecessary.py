"""无效牌（不必要牌）计算。

给定 34 张计数的手牌与副露数量，返回：
- 最优向听类型（一般形/七对/国士）
- 最小向听数
- 无效牌列表（打掉这些牌不会使向听变差；用于“手替/改良”分析）

实现与 `necessary.py` 对应，底层同样依赖预生成查表与 numba。
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .constants import YAOCHUU_TILES, Tile
from .hashing import honors_hash, suits_hash
from .shanten import ShantenFlag
from .tables import load_tables


@njit(cache=True)
def _decode_dist_disc(row_u32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = np.empty(10, dtype=np.int32)
    disc = np.empty(10, dtype=np.int64)
    for i in range(10):
        v = row_u32[i]
        dist[i] = int(v & 0xF)
        disc[i] = int((v >> 13) & 0x1FF)
    return dist, disc


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
def unnecessary_regular(hand34: np.ndarray, num_melds: int, suits_raw: np.ndarray, honors_raw: np.ndarray) -> tuple[int, int]:
    manzu_h = suits_hash(hand34, 0)
    pinzu_h = suits_hash(hand34, 9)
    souzu_h = suits_hash(hand34, 18)
    honors_h = honors_hash(hand34)

    m = 4 - num_melds

    dist, disc = _decode_dist_disc(honors_raw[honors_h])
    _add1_dist_mask(dist, disc, *_decode_dist_disc(suits_raw[souzu_h]), m)
    _add1_dist_mask(dist, disc, *_decode_dist_disc(suits_raw[pinzu_h]), m)
    _add2_dist_mask(dist, disc, *_decode_dist_disc(suits_raw[manzu_h]), m)

    shanten = int(dist[5 + m] - 1)
    mask = int(disc[5 + m])
    return shanten, mask


def unnecessary_seven_pairs(hand34: np.ndarray) -> tuple[int, int]:
    """七对子形的向听与无效牌位图。"""
    num_pairs = 0
    num_types = 0
    count1 = 0
    count_ge3 = 0

    for i in range(34):
        c = int(hand34[i])
        if c == 1:
            num_types += 1
            count1 |= 1 << i
        elif c == 2:
            num_pairs += 1
            num_types += 1
        elif c >= 3:
            num_pairs += 1
            num_types += 1
            count_ge3 |= 1 << i

    shanten = 6 - num_pairs + max(0, 7 - num_types)
    disc = (count1 | count_ge3) if num_types > 7 else count_ge3
    return shanten, int(disc)


def unnecessary_thirteen_orphans(hand34: np.ndarray) -> tuple[int, int]:
    """国士无双形的向听与无效牌位图。"""
    tanyao_tiles = (
        Tile.Manzu2, Tile.Manzu3, Tile.Manzu4, Tile.Manzu5, Tile.Manzu6, Tile.Manzu7, Tile.Manzu8,
        Tile.Pinzu2, Tile.Pinzu3, Tile.Pinzu4, Tile.Pinzu5, Tile.Pinzu6, Tile.Pinzu7, Tile.Pinzu8,
        Tile.Souzu2, Tile.Souzu3, Tile.Souzu4, Tile.Souzu5, Tile.Souzu6, Tile.Souzu7, Tile.Souzu8,
    )

    num_pairs = 0
    num_types = 0
    tanyao_flag = 0
    count2 = 0
    count_gt2 = 0

    for t in tanyao_tiles:
        if int(hand34[t]) > 0:
            tanyao_flag |= 1 << t

    for t in YAOCHUU_TILES:
        c = int(hand34[t])
        if c == 1:
            num_types += 1
        elif c == 2:
            num_types += 1
            num_pairs += 1
            count2 |= 1 << t
        elif c > 2:
            num_types += 1
            num_pairs += 1
            count_gt2 |= 1 << t

    shanten = 13 - num_types - (1 if num_pairs else 0)
    disc = (tanyao_flag | count_gt2 | count2) if num_pairs >= 2 else (tanyao_flag | count_gt2)
    return shanten, int(disc)


def unnecessary_tiles(hand34: np.ndarray, num_melds: int = 0, type_mask: int = ShantenFlag.All) -> tuple[int, int, list[int]]:
    """计算无效牌（打掉不影响最优向听的牌）。

    参数/返回的含义与 `necessary_tiles` 类似，只是输出为“可舍牌列表”。
    """
    suits_raw, honors_raw = load_tables()

    best_type = 0
    best_shanten = 10**9
    best_mask = 0

    if type_mask & ShantenFlag.Regular:
        s, m = unnecessary_regular(hand34, num_melds, suits_raw, honors_raw)
        if s < best_shanten:
            best_shanten, best_type, best_mask = s, ShantenFlag.Regular, m
        elif s == best_shanten:
            best_type |= ShantenFlag.Regular
            best_mask |= m

    if (type_mask & ShantenFlag.SevenPairs) and num_melds == 0:
        s, m = unnecessary_seven_pairs(hand34)
        if s < best_shanten:
            best_shanten, best_type, best_mask = s, ShantenFlag.SevenPairs, m
        elif s == best_shanten:
            best_type |= ShantenFlag.SevenPairs
            best_mask |= m

    if (type_mask & ShantenFlag.ThirteenOrphans) and num_melds == 0:
        s, m = unnecessary_thirteen_orphans(hand34)
        if s < best_shanten:
            best_shanten, best_type, best_mask = s, ShantenFlag.ThirteenOrphans, m
        elif s == best_shanten:
            best_type |= ShantenFlag.ThirteenOrphans
            best_mask |= m

    tiles = [i for i in range(34) if (best_mask >> i) & 1]
    return best_type, int(best_shanten), tiles
