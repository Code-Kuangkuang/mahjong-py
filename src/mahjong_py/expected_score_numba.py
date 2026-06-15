"""Experimental array-backed expected-score calculator.

This module is intentionally separate from ``expected_score_calculator``.  It
keeps the public result shape compatible while moving the graph storage and the
backward probability pass to dense arrays.  Graph construction still reuses the
existing Python scoring and tile-selection code, so this is a first migration
step rather than a full replacement.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import sys

import numpy as np
from numba import njit, prange

from .constants import Tile
from .expected_score_calculator import (
    ExpectedScoreCalculator,
    ExpectedScoreConfig,
    ExpectedScoreStat,
    _BASE_TO_RED,
    _RED_TO_BASE,
)
from .necessary import (
    necessary_regular,
)
from .score_calculator import ScoreCalculator
from .hand_separator import _create_block_patterns, _tables, separate
from .score_constants import (
    NormalYaku,
    ScoreTitle,
    ToDora,
    ToIndicator,
    WinFlag,
    Yakuman,
    Yaku,
    YakuHan,
)
from .score_types import Block, BlockType, Player, Round, RuleFlag, WaitType
from .shanten import ShantenFlag, shanten
from .tables import load_tables
from .unnecessary import (
    unnecessary_regular,
)

# Module-level cache for tables to avoid repeated loading
_TABLES_CACHE: tuple[np.ndarray, np.ndarray] | None = None

# Fast red tile lookup: _RED_BASE_LOOKUP[tile] = base_tile (or -1 if not red)
_RED_BASE_LOOKUP = [-1] * 37
for _base, _red in {
    Tile.Manzu5: Tile.RedManzu5,
    Tile.Pinzu5: Tile.RedPinzu5,
    Tile.Souzu5: Tile.RedSouzu5,
}.items():
    _RED_BASE_LOOKUP[_red] = _base


def _get_tables() -> tuple[np.ndarray, np.ndarray]:
    global _TABLES_CACHE
    if _TABLES_CACHE is None:
        _TABLES_CACHE = load_tables()
    return _TABLES_CACHE


_YAOCHUU_TILES = (
    Tile.Manzu1,
    Tile.Manzu9,
    Tile.Pinzu1,
    Tile.Pinzu9,
    Tile.Souzu1,
    Tile.Souzu9,
    Tile.East,
    Tile.South,
    Tile.West,
    Tile.North,
    Tile.White,
    Tile.Green,
    Tile.Red,
)
_TANYAO_TILES = (
    Tile.Manzu2,
    Tile.Manzu3,
    Tile.Manzu4,
    Tile.Manzu5,
    Tile.Manzu6,
    Tile.Manzu7,
    Tile.Manzu8,
    Tile.Pinzu2,
    Tile.Pinzu3,
    Tile.Pinzu4,
    Tile.Pinzu5,
    Tile.Pinzu6,
    Tile.Pinzu7,
    Tile.Pinzu8,
    Tile.Souzu2,
    Tile.Souzu3,
    Tile.Souzu4,
    Tile.Souzu5,
    Tile.Souzu6,
    Tile.Souzu7,
    Tile.Souzu8,
)
_SHANTEN_REGULAR = ShantenFlag.Regular
_SHANTEN_SEVEN_PAIRS = ShantenFlag.SevenPairs
_SHANTEN_THIRTEEN_ORPHANS = ShantenFlag.ThirteenOrphans
_T_MANZU1 = Tile.Manzu1
_T_MANZU5 = Tile.Manzu5
_T_MANZU9 = Tile.Manzu9
_T_PINZU1 = Tile.Pinzu1
_T_PINZU5 = Tile.Pinzu5
_T_PINZU9 = Tile.Pinzu9
_T_SOUZU1 = Tile.Souzu1
_T_SOUZU2 = Tile.Souzu2
_T_SOUZU3 = Tile.Souzu3
_T_SOUZU4 = Tile.Souzu4
_T_SOUZU5 = Tile.Souzu5
_T_SOUZU6 = Tile.Souzu6
_T_SOUZU8 = Tile.Souzu8
_T_SOUZU9 = Tile.Souzu9
_T_EAST = Tile.East
_T_SOUTH = Tile.South
_T_WEST = Tile.West
_T_NORTH = Tile.North
_T_WHITE = Tile.White
_T_GREEN = Tile.Green
_T_RED = Tile.Red
_T_RED_MANZU5 = Tile.RedManzu5
_T_RED_PINZU5 = Tile.RedPinzu5
_T_RED_SOUZU5 = Tile.RedSouzu5
_WIN_TSUMO = WinFlag.Tsumo
_WIN_RIICHI = WinFlag.Riichi
_WIN_IPPATSU = WinFlag.Ippatsu
_WIN_ROBBING_A_KONG = WinFlag.RobbingAKong
_WIN_AFTER_A_KONG = WinFlag.AfterAKong
_WIN_UNDER_THE_SEA = WinFlag.UnderTheSea
_WIN_UNDER_THE_RIVER = WinFlag.UnderTheRiver
_WIN_DOUBLE_RIICHI = WinFlag.DoubleRiichi
_WIN_NAGASHI_MANGAN = WinFlag.NagashiMangan
_WIN_BLESSING_OF_HEAVEN = WinFlag.BlessingOfHeaven
_WIN_BLESSING_OF_EARTH = WinFlag.BlessingOfEarth
_WIN_HAND_OF_MAN = WinFlag.HandOfMan
_YAKU_TSUMO = Yaku.Tsumo
_YAKU_RIICHI = Yaku.Riichi
_YAKU_IPPATSU = Yaku.Ippatsu
_YAKU_ROBBING_A_KONG = Yaku.RobbingAKong
_YAKU_AFTER_A_KONG = Yaku.AfterAKong
_YAKU_UNDER_THE_SEA = Yaku.UnderTheSea
_YAKU_UNDER_THE_RIVER = Yaku.UnderTheRiver
_YAKU_DOUBLE_RIICHI = Yaku.DoubleRiichi
_YAKU_NAGASHI_MANGAN = Yaku.NagashiMangan
_YAKU_BLESSING_OF_HEAVEN = Yaku.BlessingOfHeaven
_YAKU_BLESSING_OF_EARTH = Yaku.BlessingOfEarth
_YAKU_HAND_OF_MAN = Yaku.HandOfMan
_YAKU_TANYAO = Yaku.Tanyao
_YAKU_PINFU = Yaku.Pinfu
_YAKU_PURE_DOUBLE_SEQUENCE = Yaku.PureDoubleSequence
_YAKU_WHITE_DRAGON = Yaku.WhiteDragon
_YAKU_GREEN_DRAGON = Yaku.GreenDragon
_YAKU_RED_DRAGON = Yaku.RedDragon
_YAKU_SELF_WIND_EAST = Yaku.SelfWindEast
_YAKU_SELF_WIND_SOUTH = Yaku.SelfWindSouth
_YAKU_SELF_WIND_WEST = Yaku.SelfWindWest
_YAKU_SELF_WIND_NORTH = Yaku.SelfWindNorth
_YAKU_ROUND_WIND_EAST = Yaku.RoundWindEast
_YAKU_ROUND_WIND_SOUTH = Yaku.RoundWindSouth
_YAKU_ROUND_WIND_WEST = Yaku.RoundWindWest
_YAKU_ROUND_WIND_NORTH = Yaku.RoundWindNorth
_YAKU_SEVEN_PAIRS = Yaku.SevenPairs
_YAKU_ALL_TERMINALS_AND_HONORS = Yaku.AllTerminalsAndHonors
_YAKU_THREE_CONCEALED_TRIPLETS = Yaku.ThreeConcealedTriplets
_YAKU_TRIPLE_TRIPLETS = Yaku.TripleTriplets
_YAKU_MIXED_TRIPLE_SEQUENCE = Yaku.MixedTripleSequence
_YAKU_PURE_STRAIGHT = Yaku.PureStraight
_YAKU_HALF_OUTSIDE_HAND = Yaku.HalfOutsideHand
_YAKU_THREE_KONGS = Yaku.ThreeKongs
_YAKU_HALF_FLUSH = Yaku.HalfFlush
_YAKU_FULLY_OUTSIDE_HAND = Yaku.FullyOutsideHand
_YAKU_TWICE_PURE_DOUBLE_SEQUENCE = Yaku.TwicePureDoubleSequence
_YAKU_FULL_FLUSH = Yaku.FullFlush
_YAKU_LITTLE_THREE_DRAGONS = Yaku.LittleThreeDragons
_YAKU_ALL_TRIPLETS = Yaku.AllTriplets
_YAKU_ALL_GREEN = Yaku.AllGreen
_YAKU_BIG_THREE_DRAGONS = Yaku.BigThreeDragons
_YAKU_LITTLE_FOUR_WINDS = Yaku.LittleFourWinds
_YAKU_ALL_HONORS = Yaku.AllHonors
_YAKU_THIRTEEN_ORPHANS = Yaku.ThirteenOrphans
_YAKU_NINE_GATES = Yaku.NineGates
_YAKU_FOUR_CONCEALED_TRIPLETS = Yaku.FourConcealedTriplets
_YAKU_ALL_TERMINALS = Yaku.AllTerminals
_YAKU_SINGLE_WAIT_FOUR_CONCEALED_TRIPLETS = Yaku.SingleWaitFourConcealedTriplets
_YAKU_BIG_FOUR_WINDS = Yaku.BigFourWinds
_YAKU_TRUE_NINE_GATES = Yaku.TrueNineGates
_YAKU_THIRTEEN_WAIT_THIRTEEN_ORPHANS = Yaku.ThirteenWaitThirteenOrphans


@njit(cache=True)
def _calc_stats_array(
    tenpai_prob: np.ndarray,
    win_prob: np.ndarray,
    exp_score: np.ndarray,
    out_head: np.ndarray,
    in_head: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    edge_weight: np.ndarray,
    edge_score: np.ndarray,
    edge_next_out: np.ndarray,
    edge_next_in: np.ndarray,
    draw_vertices: np.ndarray,
    discard_vertices: np.ndarray,
    t_min: int,
    t_max: int,
    wall_sum: int,
) -> None:
    for turn in range(t_max, t_min - 1, -1):
        if turn < t_max:
            denom = wall_sum - turn
            for i in range(draw_vertices.size):
                vertex = int(draw_vertices[i])
                edge = int(out_head[vertex])
                while edge != -1:
                    target = int(edge_target[edge])
                    weight = float(edge_weight[edge])
                    score = float(edge_score[edge])

                    tenpai_prob[vertex, turn] += weight * (
                        tenpai_prob[target, turn + 1]
                        - tenpai_prob[vertex, turn + 1]
                    )
                    win_prob[vertex, turn] += weight * (
                        win_prob[target, turn + 1] - win_prob[vertex, turn + 1]
                    )

                    downstream_score = exp_score[target, turn + 1]
                    if score > downstream_score:
                        downstream_score = score
                    exp_score[vertex, turn] += weight * (
                        downstream_score - exp_score[vertex, turn + 1]
                    )

                    edge = int(edge_next_out[edge])

                if denom > 0:
                    tenpai_prob[vertex, turn] = (
                        tenpai_prob[vertex, turn + 1]
                        + tenpai_prob[vertex, turn] / denom
                    )
                    win_prob[vertex, turn] = (
                        win_prob[vertex, turn + 1] + win_prob[vertex, turn] / denom
                    )
                    exp_score[vertex, turn] = (
                        exp_score[vertex, turn + 1] + exp_score[vertex, turn] / denom
                    )
                else:
                    tenpai_prob[vertex, turn] = tenpai_prob[vertex, turn + 1]
                    win_prob[vertex, turn] = win_prob[vertex, turn + 1]
                    exp_score[vertex, turn] = exp_score[vertex, turn + 1]

        for i in range(discard_vertices.size):
            vertex = int(discard_vertices[i])
            edge = int(in_head[vertex])
            while edge != -1:
                source = int(edge_source[edge])
                if tenpai_prob[source, turn] > tenpai_prob[vertex, turn]:
                    tenpai_prob[vertex, turn] = tenpai_prob[source, turn]
                if win_prob[source, turn] > win_prob[vertex, turn]:
                    win_prob[vertex, turn] = win_prob[source, turn]
                if exp_score[source, turn] > exp_score[vertex, turn]:
                    exp_score[vertex, turn] = exp_score[source, turn]
                edge = int(edge_next_in[edge])


class _ArrayGraph:
    # Use 16 bits per vertex id - sufficient for typical graph sizes (< 65535)
    _EDGE_KEY_SHIFT = 16

    def __init__(self, t_size: int, vertex_capacity: int = 1024) -> None:
        self.t_size = t_size
        self.vertex_count = 0
        self.edge_count = 0
        self.edge_keys: set[int] = set()

        self.tenpai_prob = np.zeros((vertex_capacity, t_size), dtype=np.float64)
        self.win_prob = np.zeros((vertex_capacity, t_size), dtype=np.float64)
        self.exp_score = np.zeros((vertex_capacity, t_size), dtype=np.float64)
        self.out_head = np.full(vertex_capacity, -1, dtype=np.int32)
        self.in_head = np.full(vertex_capacity, -1, dtype=np.int32)

        edge_capacity = vertex_capacity * 4
        self.edge_source = np.empty(edge_capacity, dtype=np.int32)
        self.edge_target = np.empty(edge_capacity, dtype=np.int32)
        self.edge_weight = np.empty(edge_capacity, dtype=np.int16)
        self.edge_score = np.empty(edge_capacity, dtype=np.float64)
        self.edge_next_out = np.empty(edge_capacity, dtype=np.int32)
        self.edge_next_in = np.empty(edge_capacity, dtype=np.int32)

    def add_vertex(
        self,
        tenpai_init: float,
        win_init: float,
        exp_init: float,
        tenpai_last: float | None = None,
    ) -> int:
        self._ensure_vertex_capacity(self.vertex_count + 1)
        vertex = self.vertex_count
        self.vertex_count += 1

        self.tenpai_prob[vertex, :] = tenpai_init
        self.win_prob[vertex, :] = win_init
        self.exp_score[vertex, :] = exp_init
        self.out_head[vertex] = -1
        self.in_head[vertex] = -1
        if tenpai_last is not None:
            self.tenpai_prob[vertex, self.t_size - 1] = tenpai_last
        return vertex

    def reserve_edge(self, source: int, target: int) -> bool:
        key = (source << self._EDGE_KEY_SHIFT) | target
        if key in self.edge_keys:
            return False
        self.edge_keys.add(key)
        return True

    def add_edge(self, source: int, target: int, weight: int, score: float) -> None:
        self._ensure_edge_capacity(self.edge_count + 1)
        edge = self.edge_count
        self.edge_count += 1

        self.edge_source[edge] = source
        self.edge_target[edge] = target
        self.edge_weight[edge] = weight
        self.edge_score[edge] = score
        self.edge_next_out[edge] = self.out_head[source]
        self.out_head[source] = edge
        self.edge_next_in[edge] = self.in_head[target]
        self.in_head[target] = edge

    def trim(self) -> None:
        n = self.vertex_count
        m = self.edge_count
        self.tenpai_prob = self.tenpai_prob[:n].copy()
        self.win_prob = self.win_prob[:n].copy()
        self.exp_score = self.exp_score[:n].copy()
        self.out_head = self.out_head[:n].copy()
        self.in_head = self.in_head[:n].copy()
        self.edge_source = self.edge_source[:m].copy()
        self.edge_target = self.edge_target[:m].copy()
        self.edge_weight = self.edge_weight[:m].copy()
        self.edge_score = self.edge_score[:m].copy()
        self.edge_next_out = self.edge_next_out[:m].copy()
        self.edge_next_in = self.edge_next_in[:m].copy()

    def _ensure_vertex_capacity(self, needed: int) -> None:
        current = self.tenpai_prob.shape[0]
        if needed <= current:
            return
        new_size = max(needed, current * 2)
        self.tenpai_prob = _grow_2d(self.tenpai_prob, new_size)
        self.win_prob = _grow_2d(self.win_prob, new_size)
        self.exp_score = _grow_2d(self.exp_score, new_size)
        self.out_head = _grow_1d(self.out_head, new_size, -1)
        self.in_head = _grow_1d(self.in_head, new_size, -1)

    def _ensure_edge_capacity(self, needed: int) -> None:
        current = self.edge_source.size
        if needed <= current:
            return
        new_size = max(needed, current * 2)
        self.edge_source = _grow_1d(self.edge_source, new_size, 0)
        self.edge_target = _grow_1d(self.edge_target, new_size, 0)
        self.edge_weight = _grow_1d(self.edge_weight, new_size, 0)
        self.edge_score = _grow_1d(self.edge_score, new_size, 0.0)
        self.edge_next_out = _grow_1d(self.edge_next_out, new_size, -1)
        self.edge_next_in = _grow_1d(self.edge_next_in, new_size, -1)


def _grow_2d(arr: np.ndarray, new_rows: int) -> np.ndarray:
    grown = np.zeros((new_rows, arr.shape[1]), dtype=arr.dtype)
    grown[: arr.shape[0], :] = arr
    return grown


def _grow_1d(arr: np.ndarray, new_size: int, fill_value) -> np.ndarray:
    grown = np.empty(new_size, dtype=arr.dtype)
    grown[: arr.size] = arr
    grown[arr.size :] = fill_value
    return grown


@njit(cache=True)
def _necessary_seven_pairs_mask_numba(hand34: np.ndarray) -> tuple[int, int]:
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

    shanten_num = 6 - num_pairs
    if num_types < 7:
        shanten_num += 7 - num_types

    if num_types < 7:
        wait = count0 | count1
    elif num_pairs == 7:
        wait = 0
    else:
        wait = count1
    return shanten_num, wait


@njit(cache=True)
def _necessary_thirteen_orphans_mask_numba(hand34: np.ndarray) -> tuple[int, int]:
    num_pairs = 0
    num_types = 0
    count0 = 0
    count1 = 0

    for t in _YAOCHUU_TILES:
        c = int(hand34[t])
        if c == 0:
            count0 |= 1 << t
        elif c == 1:
            count1 |= 1 << t
            num_types += 1
        else:
            num_types += 1
            num_pairs += 1

    shanten_num = 13 - num_types
    if num_pairs:
        shanten_num -= 1
        wait = count0
    else:
        wait = count0 | count1
    return shanten_num, wait


@njit(cache=True)
def _unnecessary_seven_pairs_mask_numba(hand34: np.ndarray) -> tuple[int, int]:
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

    shanten_num = 6 - num_pairs
    if num_types < 7:
        shanten_num += 7 - num_types
    disc = count1 | count_ge3 if num_types > 7 else count_ge3
    return shanten_num, disc


@njit(cache=True)
def _unnecessary_thirteen_orphans_mask_numba(hand34: np.ndarray) -> tuple[int, int]:
    num_pairs = 0
    num_types = 0
    tanyao_flag = 0
    count2 = 0
    count_gt2 = 0

    for t in _TANYAO_TILES:
        if int(hand34[t]) > 0:
            tanyao_flag |= 1 << t

    for t in _YAOCHUU_TILES:
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

    shanten_num = 13 - num_types
    if num_pairs:
        shanten_num -= 1
    disc = tanyao_flag | count_gt2 | count2 if num_pairs >= 2 else tanyao_flag | count_gt2
    return shanten_num, disc


@njit(cache=True)
def _necessary_mask_numba(
    hand34: np.ndarray,
    num_melds: int,
    type_mask: int,
    suits_raw: np.ndarray,
    honors_raw: np.ndarray,
) -> tuple[int, int, int]:
    best_type = 0
    best_shanten = 10**9
    best_mask = 0

    if type_mask & _SHANTEN_REGULAR:
        s, m = necessary_regular(hand34, num_melds, suits_raw, honors_raw)
        if s < best_shanten:
            best_shanten = s
            best_type = _SHANTEN_REGULAR
            best_mask = m
        elif s == best_shanten:
            best_type |= _SHANTEN_REGULAR
            best_mask |= m

    if (type_mask & _SHANTEN_SEVEN_PAIRS) and num_melds == 0:
        s, m = _necessary_seven_pairs_mask_numba(hand34)
        if s < best_shanten:
            best_shanten = s
            best_type = _SHANTEN_SEVEN_PAIRS
            best_mask = m
        elif s == best_shanten:
            best_type |= _SHANTEN_SEVEN_PAIRS
            best_mask |= m

    if (type_mask & _SHANTEN_THIRTEEN_ORPHANS) and num_melds == 0:
        s, m = _necessary_thirteen_orphans_mask_numba(hand34)
        if s < best_shanten:
            best_shanten = s
            best_type = _SHANTEN_THIRTEEN_ORPHANS
            best_mask = m
        elif s == best_shanten:
            best_type |= _SHANTEN_THIRTEEN_ORPHANS
            best_mask |= m

    return best_type, best_shanten, best_mask


@njit(cache=True)
def _unnecessary_mask_numba(
    hand34: np.ndarray,
    num_melds: int,
    type_mask: int,
    suits_raw: np.ndarray,
    honors_raw: np.ndarray,
) -> tuple[int, int, int]:
    best_type = 0
    best_shanten = 10**9
    best_mask = 0

    if type_mask & _SHANTEN_REGULAR:
        s, m = unnecessary_regular(hand34, num_melds, suits_raw, honors_raw)
        if s < best_shanten:
            best_shanten = s
            best_type = _SHANTEN_REGULAR
            best_mask = m
        elif s == best_shanten:
            best_type |= _SHANTEN_REGULAR
            best_mask |= m

    if (type_mask & _SHANTEN_SEVEN_PAIRS) and num_melds == 0:
        s, m = _unnecessary_seven_pairs_mask_numba(hand34)
        if s < best_shanten:
            best_shanten = s
            best_type = _SHANTEN_SEVEN_PAIRS
            best_mask = m
        elif s == best_shanten:
            best_type |= _SHANTEN_SEVEN_PAIRS
            best_mask |= m

    if (type_mask & _SHANTEN_THIRTEEN_ORPHANS) and num_melds == 0:
        s, m = _unnecessary_thirteen_orphans_mask_numba(hand34)
        if s < best_shanten:
            best_shanten = s
            best_type = _SHANTEN_THIRTEEN_ORPHANS
            best_mask = m
        elif s == best_shanten:
            best_type |= _SHANTEN_THIRTEEN_ORPHANS
            best_mask |= m

    return best_type, best_shanten, best_mask


def _iter_bits(mask: int):
    """Iterate over set bits in a mask. Yields tile indices (0-36)."""
    m = mask
    while m:
        lsb = m & -m
        tile = (lsb.bit_length() - 1)
        yield tile
        m ^= lsb


@njit(cache=True)
def _hand_hashes_numba(hand: np.ndarray) -> tuple[int, int, int, int]:
    manzu = 0
    pinzu = 0
    souzu = 0
    honors = 0

    for tile in range(0, 9):
        manzu = manzu * 8 + int(hand[tile])
    for tile in range(9, 18):
        pinzu = pinzu * 8 + int(hand[tile])
    for tile in range(18, 27):
        souzu = souzu * 8 + int(hand[tile])
    for tile in range(27, 34):
        honors = honors * 8 + int(hand[tile])

    return manzu, pinzu, souzu, honors


@njit(cache=True)
def _check_not_pattern_yaku_numba(
    hand: np.ndarray,
    round_wind: int,
    seat_wind: int,
    win_tile: int,
    win_flag: int,
    shanten_type: int,
) -> int:
    yaku_list = 0
    if win_flag & _WIN_TSUMO:
        yaku_list |= _YAKU_TSUMO
    if win_flag & _WIN_RIICHI:
        yaku_list |= _YAKU_RIICHI
    if win_flag & _WIN_IPPATSU:
        yaku_list |= _YAKU_IPPATSU
    if win_flag & _WIN_ROBBING_A_KONG:
        yaku_list |= _YAKU_ROBBING_A_KONG
    if win_flag & _WIN_AFTER_A_KONG:
        yaku_list |= _YAKU_AFTER_A_KONG
    if win_flag & _WIN_UNDER_THE_SEA:
        yaku_list |= _YAKU_UNDER_THE_SEA
    if win_flag & _WIN_UNDER_THE_RIVER:
        yaku_list |= _YAKU_UNDER_THE_RIVER
    if win_flag & _WIN_DOUBLE_RIICHI:
        yaku_list |= _YAKU_DOUBLE_RIICHI
    if win_flag & _WIN_NAGASHI_MANGAN:
        yaku_list |= _YAKU_NAGASHI_MANGAN
    if win_flag & _WIN_BLESSING_OF_HEAVEN:
        yaku_list |= _YAKU_BLESSING_OF_HEAVEN
    if win_flag & _WIN_BLESSING_OF_EARTH:
        yaku_list |= _YAKU_BLESSING_OF_EARTH
    if win_flag & _WIN_HAND_OF_MAN:
        yaku_list |= _YAKU_HAND_OF_MAN

    nored_win_tile = win_tile
    if win_tile == _T_RED_MANZU5:
        nored_win_tile = _T_MANZU5
    elif win_tile == _T_RED_PINZU5:
        nored_win_tile = _T_PINZU5
    elif win_tile == _T_RED_SOUZU5:
        nored_win_tile = _T_SOUZU5

    has_manzu = False
    has_pinzu = False
    has_souzu = False
    has_honor = False
    tanyao = True
    all_green = True
    all_terminals_or_honors = True

    for tile in range(34):
        count = int(hand[tile])
        if count == 0:
            continue

        if tile <= _T_MANZU9:
            has_manzu = True
        elif tile <= _T_PINZU9:
            has_pinzu = True
        elif tile <= _T_SOUZU9:
            has_souzu = True
        else:
            has_honor = True

        n = tile % 9
        terminal_or_honor = tile >= _T_EAST or n == 0 or n == 8
        if terminal_or_honor:
            tanyao = False
        else:
            all_terminals_or_honors = False

        if not (
            tile == _T_SOUZU2
            or tile == _T_SOUZU3
            or tile == _T_SOUZU4
            or tile == _T_SOUZU6
            or tile == _T_SOUZU8
            or tile == _T_GREEN
        ):
            all_green = False

    if shanten_type & _SHANTEN_REGULAR:
        if all_green:
            yaku_list |= _YAKU_ALL_GREEN

        dragon_total = int(hand[_T_WHITE] + hand[_T_GREEN] + hand[_T_RED])
        if dragon_total == 8:
            yaku_list |= _YAKU_LITTLE_THREE_DRAGONS
        elif dragon_total == 9:
            yaku_list |= _YAKU_BIG_THREE_DRAGONS

        wind_total = int(
            hand[_T_EAST] + hand[_T_SOUTH] + hand[_T_WEST] + hand[_T_NORTH]
        )
        if wind_total == 11:
            yaku_list |= _YAKU_LITTLE_FOUR_WINDS
        elif wind_total == 12:
            yaku_list |= _YAKU_BIG_FOUR_WINDS

        if not (has_manzu or has_pinzu or has_souzu):
            yaku_list |= _YAKU_ALL_HONORS

        num_triplets = 0
        num_pairs = 0
        single_wait = False
        for tile in range(34):
            count = int(hand[tile])
            if count == 3:
                num_triplets += 1
            elif count == 2:
                num_pairs += 1
                if tile == nored_win_tile:
                    single_wait = True
        if num_triplets == 4 and num_pairs == 1:
            if single_wait:
                yaku_list |= _YAKU_SINGLE_WAIT_FOUR_CONCEALED_TRIPLETS
            else:
                yaku_list |= _YAKU_FOUR_CONCEALED_TRIPLETS

        if all_terminals_or_honors:
            if has_honor:
                yaku_list |= _YAKU_ALL_TERMINALS_AND_HONORS
            else:
                yaku_list |= _YAKU_ALL_TERMINALS

        yaku_list |= _check_nine_gates_numba(hand, nored_win_tile)

        if tanyao:
            yaku_list |= _YAKU_TANYAO
        yaku_list |= _check_flush_numba(has_manzu, has_pinzu, has_souzu, has_honor)
        yaku_list |= _check_value_tile_numba(hand, round_wind, seat_wind)
        return yaku_list

    if shanten_type & _SHANTEN_SEVEN_PAIRS:
        yaku_list |= _YAKU_SEVEN_PAIRS
        if not (has_manzu or has_pinzu or has_souzu):
            yaku_list |= _YAKU_ALL_HONORS
        if all_terminals_or_honors:
            if has_honor:
                yaku_list |= _YAKU_ALL_TERMINALS_AND_HONORS
            else:
                yaku_list |= _YAKU_ALL_TERMINALS
        if tanyao:
            yaku_list |= _YAKU_TANYAO
        yaku_list |= _check_flush_numba(has_manzu, has_pinzu, has_souzu, has_honor)
        return yaku_list

    if shanten_type & _SHANTEN_THIRTEEN_ORPHANS:
        if _is_thirteen_wait_thirteen_orphans_numba(hand, nored_win_tile):
            yaku_list |= _YAKU_THIRTEEN_WAIT_THIRTEEN_ORPHANS
        else:
            yaku_list |= _YAKU_THIRTEEN_ORPHANS

    return yaku_list


@njit(cache=True)
def _check_flush_numba(
    has_manzu: bool,
    has_pinzu: bool,
    has_souzu: bool,
    has_honor: bool,
) -> int:
    if int(has_manzu) + int(has_pinzu) + int(has_souzu) == 1:
        if has_honor:
            return _YAKU_HALF_FLUSH
        return _YAKU_FULL_FLUSH
    return 0


@njit(cache=True)
def _check_value_tile_numba(hand: np.ndarray, round_wind: int, seat_wind: int) -> int:
    yaku_list = 0
    if int(hand[_T_WHITE]) == 3:
        yaku_list |= _YAKU_WHITE_DRAGON
    if int(hand[_T_GREEN]) == 3:
        yaku_list |= _YAKU_GREEN_DRAGON
    if int(hand[_T_RED]) == 3:
        yaku_list |= _YAKU_RED_DRAGON

    if int(hand[round_wind]) == 3:
        if round_wind == _T_EAST:
            yaku_list |= _YAKU_ROUND_WIND_EAST
        elif round_wind == _T_SOUTH:
            yaku_list |= _YAKU_ROUND_WIND_SOUTH
        elif round_wind == _T_WEST:
            yaku_list |= _YAKU_ROUND_WIND_WEST
        elif round_wind == _T_NORTH:
            yaku_list |= _YAKU_ROUND_WIND_NORTH

    if int(hand[seat_wind]) == 3:
        if seat_wind == _T_EAST:
            yaku_list |= _YAKU_SELF_WIND_EAST
        elif seat_wind == _T_SOUTH:
            yaku_list |= _YAKU_SELF_WIND_SOUTH
        elif seat_wind == _T_WEST:
            yaku_list |= _YAKU_SELF_WIND_WEST
        elif seat_wind == _T_NORTH:
            yaku_list |= _YAKU_SELF_WIND_NORTH
    return yaku_list


@njit(cache=True)
def _check_nine_gates_numba(hand: np.ndarray, win_tile: int) -> int:
    start = -1
    if win_tile <= _T_MANZU9:
        start = _T_MANZU1
        other_a = _T_PINZU1
        other_b = _T_PINZU9
        other_c = _T_SOUZU1
        other_d = _T_SOUZU9
    elif win_tile <= _T_PINZU9:
        start = _T_PINZU1
        other_a = _T_MANZU1
        other_b = _T_MANZU9
        other_c = _T_SOUZU1
        other_d = _T_SOUZU9
    elif win_tile <= _T_SOUZU9:
        start = _T_SOUZU1
        other_a = _T_MANZU1
        other_b = _T_MANZU9
        other_c = _T_PINZU1
        other_d = _T_PINZU9
    else:
        return 0

    for tile in range(other_a, other_b + 1):
        if int(hand[tile]) != 0:
            return 0
    for tile in range(other_c, other_d + 1):
        if int(hand[tile]) != 0:
            return 0
    for tile in range(_T_EAST, _T_RED + 1):
        if int(hand[tile]) != 0:
            return 0

    if int(hand[start]) < 3 or int(hand[start + 8]) < 3:
        return 0
    for tile in range(start + 1, start + 8):
        if int(hand[tile]) == 0:
            return 0

    is_true = True
    for offset in range(9):
        expected = 1
        if offset == 0 or offset == 8:
            expected = 3
        count = int(hand[start + offset])
        if start + offset == win_tile:
            count -= 1
        if count != expected:
            is_true = False
            break

    if is_true:
        return _YAKU_TRUE_NINE_GATES
    return _YAKU_NINE_GATES


@njit(cache=True)
def _is_thirteen_wait_thirteen_orphans_numba(hand: np.ndarray, win_tile: int) -> bool:
    if not (
        win_tile == _T_MANZU1
        or win_tile == _T_MANZU9
        or win_tile == _T_PINZU1
        or win_tile == _T_PINZU9
        or win_tile == _T_SOUZU1
        or win_tile == _T_SOUZU9
        or win_tile == _T_EAST
        or win_tile == _T_SOUTH
        or win_tile == _T_WEST
        or win_tile == _T_NORTH
        or win_tile == _T_WHITE
        or win_tile == _T_GREEN
        or win_tile == _T_RED
    ):
        return False

    for tile in range(34):
        count = int(hand[tile])
        is_yaochuu = (
            tile == _T_MANZU1
            or tile == _T_MANZU9
            or tile == _T_PINZU1
            or tile == _T_PINZU9
            or tile == _T_SOUZU1
            or tile == _T_SOUZU9
            or tile == _T_EAST
            or tile == _T_SOUTH
            or tile == _T_WEST
            or tile == _T_NORTH
            or tile == _T_WHITE
            or tile == _T_GREEN
            or tile == _T_RED
        )
        if is_yaochuu:
            expected = 1
            if tile == win_tile:
                expected = 2
            if count != expected:
                return False
        elif count != 0:
            return False
    return True


@njit(cache=True)
def _normal_han_closed_numba(yaku_list: int) -> int:
    han = 0
    if yaku_list & _YAKU_TSUMO:
        han += 1
    if yaku_list & _YAKU_RIICHI:
        han += 1
    if yaku_list & _YAKU_IPPATSU:
        han += 1
    if yaku_list & _YAKU_TANYAO:
        han += 1
    if yaku_list & _YAKU_PINFU:
        han += 1
    if yaku_list & _YAKU_PURE_DOUBLE_SEQUENCE:
        han += 1
    if yaku_list & _YAKU_ROBBING_A_KONG:
        han += 1
    if yaku_list & _YAKU_AFTER_A_KONG:
        han += 1
    if yaku_list & _YAKU_UNDER_THE_SEA:
        han += 1
    if yaku_list & _YAKU_UNDER_THE_RIVER:
        han += 1
    if yaku_list & _YAKU_WHITE_DRAGON:
        han += 1
    if yaku_list & _YAKU_GREEN_DRAGON:
        han += 1
    if yaku_list & _YAKU_RED_DRAGON:
        han += 1
    if yaku_list & _YAKU_SELF_WIND_EAST:
        han += 1
    if yaku_list & _YAKU_SELF_WIND_SOUTH:
        han += 1
    if yaku_list & _YAKU_SELF_WIND_WEST:
        han += 1
    if yaku_list & _YAKU_SELF_WIND_NORTH:
        han += 1
    if yaku_list & _YAKU_ROUND_WIND_EAST:
        han += 1
    if yaku_list & _YAKU_ROUND_WIND_SOUTH:
        han += 1
    if yaku_list & _YAKU_ROUND_WIND_WEST:
        han += 1
    if yaku_list & _YAKU_ROUND_WIND_NORTH:
        han += 1
    if yaku_list & _YAKU_DOUBLE_RIICHI:
        han += 2
    if yaku_list & _YAKU_SEVEN_PAIRS:
        han += 2
    if yaku_list & _YAKU_ALL_TRIPLETS:
        han += 2
    if yaku_list & _YAKU_THREE_CONCEALED_TRIPLETS:
        han += 2
    if yaku_list & _YAKU_TRIPLE_TRIPLETS:
        han += 2
    if yaku_list & _YAKU_MIXED_TRIPLE_SEQUENCE:
        han += 2
    if yaku_list & _YAKU_ALL_TERMINALS_AND_HONORS:
        han += 2
    if yaku_list & _YAKU_PURE_STRAIGHT:
        han += 2
    if yaku_list & _YAKU_HALF_OUTSIDE_HAND:
        han += 2
    if yaku_list & _YAKU_LITTLE_THREE_DRAGONS:
        han += 2
    if yaku_list & _YAKU_THREE_KONGS:
        han += 2
    if yaku_list & _YAKU_HALF_FLUSH:
        han += 3
    if yaku_list & _YAKU_FULLY_OUTSIDE_HAND:
        han += 3
    if yaku_list & _YAKU_TWICE_PURE_DOUBLE_SEQUENCE:
        han += 3
    if yaku_list & _YAKU_FULL_FLUSH:
        han += 6
    return han


def _count_dora_closed_hand(hand: np.ndarray, indicators: list[int]) -> int:
    """Count dora for closed hand (no melds) - optimized inline version."""
    count = 0
    for t in indicators:
        dora = ToDora[t]
        count += int(hand[dora])
    return count


class ExpectedScoreNumbaCalculator:
    """Experimental calculator using an array graph and a Numba stats pass."""

    @staticmethod
    def calc(
        config: ExpectedScoreConfig,
        round_: Round,
        player: Player,
        wall: list[int] | None = None,
    ) -> tuple[list[ExpectedScoreStat], int]:
        if sys.getrecursionlimit() < 100_000:
            sys.setrecursionlimit(100_000)

        cfg = replace(config)
        wall_counts_normal = (
            list(wall)
            if wall is not None
            else ExpectedScoreCalculator.create_wall(round_, player, cfg.enable_reddora)
        )
        if cfg.sum == 0:
            cfg.sum = sum(wall_counts_normal[:34])

        cfg.enable_riichi = True

        graph = _ArrayGraph(cfg.t_max + 1)
        draw_cache: dict[tuple[int, ...], int] = {}
        discard_cache: dict[tuple[int, ...], int] = {}

        work_player = ExpectedScoreCalculator._copy_player(player)
        hand_counts = ExpectedScoreCalculator._encode(
            work_player.hand, cfg.enable_reddora
        )
        wall_counts = ExpectedScoreCalculator._encode(wall_counts_normal, cfg.enable_reddora)
        hand_origin = tuple(hand_counts)
        shanten_org = shanten(
            work_player.hand[:34],
            work_player.num_melds(),
            cfg.shanten_type,
        )[1]
        num_tiles = work_player.num_tiles() + work_player.num_melds() * 3
        riichi = cfg.enable_riichi and work_player.is_closed() and shanten_org <= 0

        stats: list[ExpectedScoreStat] = []

        if num_tiles == 13:
            if cfg.calc_stats:
                ExpectedScoreNumbaCalculator._draw_node(
                    cfg, round_, work_player, graph,
                    draw_cache, discard_cache, hand_counts,
                    wall_counts, hand_origin, shanten_org, riichi,
                )
                ExpectedScoreNumbaCalculator._calc_stats(
                    cfg, graph, draw_cache, discard_cache
                )
                vertex = draw_cache.get(tuple(hand_counts))
                if vertex is not None:
                    shanten2, needed = ExpectedScoreCalculator._get_necessary_tiles(
                        cfg, work_player, wall_counts_normal
                    )
                    stats.append(
                        ExpectedScoreStat(
                            Tile.Null,
                            graph.tenpai_prob[vertex].tolist(),
                            graph.win_prob[vertex].tolist(),
                            graph.exp_score[vertex].tolist(),
                            needed,
                            shanten2,
                        )
                    )
            else:
                shanten2, needed = ExpectedScoreCalculator._get_necessary_tiles(
                    cfg, work_player, wall_counts_normal
                )
                stats.append(ExpectedScoreStat(Tile.Null, [], [], [], needed, shanten2))
        elif num_tiles == 14:
            if cfg.calc_stats:
                ExpectedScoreNumbaCalculator._discard_node(
                    cfg, round_, work_player, graph,
                    draw_cache, discard_cache, hand_counts,
                    wall_counts, hand_origin, shanten_org, riichi,
                )
                ExpectedScoreNumbaCalculator._calc_stats(
                    cfg, graph, draw_cache, discard_cache
                )

                for tile in range(Tile.Length):
                    if hand_counts[tile] <= 0:
                        continue
                    ExpectedScoreCalculator._discard(
                        work_player, hand_counts, wall_counts, tile
                    )
                    vertex = draw_cache.get(tuple(hand_counts))
                    if vertex is not None:
                        shanten2, needed = ExpectedScoreCalculator._get_necessary_tiles(
                            cfg, work_player, wall_counts_normal
                        )
                        stats.append(
                            ExpectedScoreStat(
                                tile,
                                graph.tenpai_prob[vertex].tolist(),
                                graph.win_prob[vertex].tolist(),
                                graph.exp_score[vertex].tolist(),
                                needed,
                                shanten2,
                            )
                        )
                    ExpectedScoreCalculator._draw(
                        work_player, hand_counts, wall_counts, tile
                    )
            else:
                for tile in range(Tile.Length):
                    if hand_counts[tile] <= 0:
                        continue
                    ExpectedScoreCalculator._discard(
                        work_player, hand_counts, wall_counts, tile
                    )
                    shanten2, needed = ExpectedScoreCalculator._get_necessary_tiles(
                        cfg, work_player, wall_counts_normal
                    )
                    stats.append(ExpectedScoreStat(tile, [], [], [], needed, shanten2))
                    ExpectedScoreCalculator._draw(
                        work_player, hand_counts, wall_counts, tile
                    )
        else:
            raise ValueError("Invalid number of tiles.")

        return stats, graph.vertex_count

    @staticmethod
    def _draw_node(
        config: ExpectedScoreConfig,
        round_: Round,
        player: Player,
        graph: _ArrayGraph,
        draw_cache: dict[tuple[int, ...], int],
        discard_cache: dict[tuple[int, ...], int],
        hand_counts: list[int],
        wall_counts: list[int],
        hand_origin: list[int],
        shanten_org: int,
        riichi: bool,
    ) -> int:
        key = tuple(hand_counts)
        if key in draw_cache:
            return draw_cache[key]

        shanten_type, shanten_num, wait_mask = _necessary_mask(
            player.hand[:34],
            player.num_melds(),
            config.shanten_type,
        )
        wait_mask = _expand_red_mask(wait_mask)
        allow_tegawari = (
            config.enable_tegawari
            and ExpectedScoreCalculator._distance(hand_counts, hand_origin)
            + shanten_num
            < shanten_org + config.extra
        )

        vertex = graph.add_vertex(
            0.0,
            0.0,
            0.0,
            tenpai_last=1.0 if shanten_num == 0 else 0.0,
        )
        draw_cache[key] = vertex

        for tile in range(Tile.Length):
            is_wait = ((wait_mask >> tile) & 1) != 0
            if wall_counts[tile] <= 0 or not (allow_tegawari or is_wait):
                continue

            weight = wall_counts[tile]
            # Inline draw: hand_counts[tile] += 1; wall_counts[tile] -= 1; player.hand[tile] += 1
            hand_counts[tile] += 1
            wall_counts[tile] -= 1
            player.hand[tile] += 1
            _base = _RED_BASE_LOOKUP[tile]
            if _base != -1:
                player.hand[_base] += 1

            call_riichi = (
                True
                if (
                    config.enable_riichi
                    and player.is_closed()
                    and shanten_num == 1
                    and is_wait
                )
                else riichi
            )
            target = ExpectedScoreNumbaCalculator._discard_node(
                config,
                round_,
                player,
                graph,
                draw_cache,
                discard_cache,
                hand_counts,
                wall_counts,
                hand_origin,
                shanten_org,
                call_riichi,
            )
            if graph.reserve_edge(vertex, target):
                score = (
                    ExpectedScoreNumbaCalculator._calc_score(
                        config,
                        round_,
                        player,
                        hand_counts,
                        wall_counts,
                        shanten_type,
                        tile,
                        riichi,
                    )
                    if shanten_num == 0 and is_wait
                    else 0.0
                )
                graph.add_edge(vertex, target, weight, score)

            ExpectedScoreCalculator._discard(player, hand_counts, wall_counts, tile)

        return vertex

    @staticmethod
    def _discard_node(
        config: ExpectedScoreConfig,
        round_: Round,
        player: Player,
        graph: _ArrayGraph,
        draw_cache: dict[tuple[int, ...], int],
        discard_cache: dict[tuple[int, ...], int],
        hand_counts: list[int],
        wall_counts: list[int],
        hand_origin: list[int],
        shanten_org: int,
        riichi: bool,
    ) -> int:
        key = tuple(hand_counts)
        if key in discard_cache:
            return discard_cache[key]

        shanten_type, shanten_num, discard_mask = _unnecessary_mask(
            player.hand[:34],
            player.num_melds(),
            config.shanten_type,
        )
        discard_mask = _expand_red_mask(discard_mask)
        allow_shanten_down = (
            config.enable_shanten_down
            and ExpectedScoreCalculator._distance(hand_counts, hand_origin)
            + shanten_num
            < shanten_org + config.extra
        )

        vertex = graph.add_vertex(
            1.0 if shanten_num == 0 else 0.0,
            1.0 if shanten_num == -1 else 0.0,
            0.0,
        )
        discard_cache[key] = vertex

        for tile in range(Tile.Length):
            is_discard = ((discard_mask >> tile) & 1) != 0
            if hand_counts[tile] <= 0 or not (allow_shanten_down or is_discard):
                continue

            # Inline discard: hand_counts[tile] -= 1; wall_counts[tile] += 1; player.hand[tile] -= 1
            hand_counts[tile] -= 1
            wall_counts[tile] += 1
            player.hand[tile] -= 1
            _base = _RED_BASE_LOOKUP[tile]
            if _base != -1:
                player.hand[_base] -= 1
            weight = wall_counts[tile]
            source = ExpectedScoreNumbaCalculator._draw_node(
                config,
                round_,
                player,
                graph,
                draw_cache,
                discard_cache,
                hand_counts,
                wall_counts,
                hand_origin,
                shanten_org,
                riichi,
            )
            ExpectedScoreCalculator._draw(player, hand_counts, wall_counts, tile)

            if graph.reserve_edge(source, vertex):
                score = (
                    ExpectedScoreNumbaCalculator._calc_score(
                        config,
                        round_,
                        player,
                        hand_counts,
                        wall_counts,
                        shanten_type,
                        tile,
                        riichi,
                    )
                    if shanten_num == -1
                    else 0.0
                )
                graph.add_edge(source, vertex, weight, score)

        return vertex

    @staticmethod
    def _calc_stats(
        config: ExpectedScoreConfig,
        graph: _ArrayGraph,
        draw_cache: dict[tuple[int, ...], int],
        discard_cache: dict[tuple[int, ...], int],
    ) -> None:
        graph.trim()
        draw_vertices = np.asarray(list(draw_cache.values()), dtype=np.int32)
        discard_vertices = np.asarray(list(discard_cache.values()), dtype=np.int32)
        _calc_stats_array(
            graph.tenpai_prob,
            graph.win_prob,
            graph.exp_score,
            graph.out_head,
            graph.in_head,
            graph.edge_source,
            graph.edge_target,
            graph.edge_weight,
            graph.edge_score,
            graph.edge_next_out,
            graph.edge_next_in,
            draw_vertices,
            discard_vertices,
            config.t_min,
            config.t_max,
            config.sum,
    )

    @staticmethod
    def _calc_score(
        config: ExpectedScoreConfig,
        round_: Round,
        player: Player,
        hand_counts: list[int],
        wall_counts: list[int],
        shanten_type: int,
        win_tile: int,
        riichi: bool,
    ) -> float:
        score = _calc_score_fast_or_none(
            config,
            round_,
            player,
            hand_counts,
            wall_counts,
            shanten_type,
            win_tile,
            riichi,
        )
        if score is None:
            score = ExpectedScoreCalculator._calc_score(
                config,
                round_,
                player,
                hand_counts,
                wall_counts,
                shanten_type,
                win_tile,
                riichi,
            )
        return score


def _check_pattern_yaku_fast(
    round_: Round,
    player: Player,
    win_tile: int,
    win_flag: int,
    shanten_type: int,
) -> tuple[int, int]:
    if shanten_type == _SHANTEN_SEVEN_PAIRS:
        return 0, 25

    # Try cache for closed hands (no melds)
    if not player.melds:
        manzu_hash, pinzu_hash, souzu_hash, honors_hash = _hand_hashes_numba(player.hand)
        cache_key = (
            manzu_hash, pinzu_hash, souzu_hash, honors_hash,
            win_tile, win_flag, shanten_type,
            round_.wind, player.wind,
        )
        cached = _PATTERN_CACHE.get(cache_key)
        if cached is not None:
            return cached

    patterns = _separate_closed_fast(player, win_tile, win_flag)
    if not patterns:
        return 0, 0

    max_han = 0
    max_fu = 0
    max_yaku_list = 0
    is_closed = player.is_closed()
    is_tsumo = bool(win_flag & WinFlag.Tsumo)

    for blocks, wait_type in patterns:
        yaku_list, fu = _evaluate_blocks_fast(
            blocks,
            wait_type,
            is_closed,
            is_tsumo,
            round_.wind,
            player.wind,
        )

        if is_closed:
            han = _normal_han_closed_numba(yaku_list)
        else:
            han = 0
            if yaku_list & Yaku.Pinfu:
                han += YakuHan[Yaku.Pinfu][1]
            if yaku_list & Yaku.PureDoubleSequence:
                han += YakuHan[Yaku.PureDoubleSequence][1]
            if yaku_list & Yaku.AllTriplets:
                han += YakuHan[Yaku.AllTriplets][1]
            if yaku_list & Yaku.ThreeConcealedTriplets:
                han += YakuHan[Yaku.ThreeConcealedTriplets][1]
            if yaku_list & Yaku.TripleTriplets:
                han += YakuHan[Yaku.TripleTriplets][1]
            if yaku_list & Yaku.MixedTripleSequence:
                han += YakuHan[Yaku.MixedTripleSequence][1]
            if yaku_list & Yaku.PureStraight:
                han += YakuHan[Yaku.PureStraight][1]
            if yaku_list & Yaku.HalfOutsideHand:
                han += YakuHan[Yaku.HalfOutsideHand][1]
            if yaku_list & Yaku.ThreeKongs:
                han += YakuHan[Yaku.ThreeKongs][1]
            if yaku_list & Yaku.FullyOutsideHand:
                han += YakuHan[Yaku.FullyOutsideHand][1]
            if yaku_list & Yaku.TwicePureDoubleSequence:
                han += YakuHan[Yaku.TwicePureDoubleSequence][1]

        if han > max_han or (han == max_han and fu > max_fu):
            max_han = han
            max_fu = fu
            max_yaku_list = yaku_list

    max_fu = ((max_fu + 9) // 10) * 10

    # Cache result for closed hands
    if not player.melds:
        cache_key = (
            manzu_hash, pinzu_hash, souzu_hash, honors_hash,
            win_tile, win_flag, shanten_type,
            round_.wind, player.wind,
        )
        _PATTERN_CACHE[cache_key] = (max_yaku_list, max_fu)

    return max_yaku_list, max_fu


# Cache for _separate_closed_fast results
# Key: (manzu_hash, pinzu_hash, souzu_hash, honors_hash, win_tile, tsumo)
# Value: list of (blocks_tuple, wait_type) where blocks_tuple = tuple of (block_type, min_tile)
_SEPARATE_CACHE: dict[tuple, list[tuple[tuple, int]]] = {}

# Cache for _check_pattern_yaku_fast results (second-level)
# Key: (manzu_hash, pinzu_hash, souzu_hash, honors_hash, win_tile, win_flag, shanten_type, round_wind, seat_wind)
# Value: (yaku_list, fu)
_PATTERN_CACHE: dict[tuple, tuple[int, int]] = {}


def _separate_closed_fast(player: Player, win_tile: int, win_flag: int):
    if player.melds:
        return separate(player, win_tile, win_flag)

    s_tbl, z_tbl = _tables()
    hand = player.hand

    manzu_hash, pinzu_hash, souzu_hash, honors_hash = _hand_hashes_numba(hand)
    nored_win = _RED_TO_BASE.get(win_tile, win_tile)
    tsumo = bool(win_flag & WinFlag.Tsumo)

    cache_key = (manzu_hash, pinzu_hash, souzu_hash, honors_hash, nored_win, tsumo)
    cached = _SEPARATE_CACHE.get(cache_key)
    if cached is not None:
        # Reconstruct Block objects from cached tuples
        result = []
        for blocks_tuple, wait_type in cached:
            blocks = [Block(btype, mt) for btype, mt in blocks_tuple]
            result.append((blocks, wait_type))
        return result

    patterns = []
    blocks = [Block() for _ in range(5)]
    _create_block_patterns(
        nored_win,
        tsumo,
        patterns,
        blocks,
        0,
        0,
        s_tbl.get(manzu_hash, []),
        s_tbl.get(pinzu_hash, []),
        s_tbl.get(souzu_hash, []),
        z_tbl.get(honors_hash, []),
    )

    # Cache as tuples (hashable)
    cached = []
    for pattern_blocks, wait_type in patterns:
        blocks_tuple = tuple((b.type, b.min_tile) for b in pattern_blocks)
        cached.append((blocks_tuple, wait_type))
    _SEPARATE_CACHE[cache_key] = cached

    return patterns


def _evaluate_blocks_fast(
    blocks,
    wait_type: int,
    is_closed: bool,
    is_tsumo: bool,
    round_wind: int,
    seat_wind: int,
) -> tuple[int, int]:
    seq_counts = [0] * 34
    triplet_counts = [0] * 34
    no_triplet_or_kong = True
    all_triplets = True
    outside_valid = True
    outside_has_honor = False
    outside_has_sequence = False
    num_concealed_triplets = 0
    num_kongs = 0

    fu = 20
    if is_closed and not is_tsumo:
        fu += 10
    elif is_tsumo:
        fu += 2
    if wait_type in (WaitType.ClosedWait, WaitType.EdgeWait, WaitType.PairWait):
        fu += 2

    for block in blocks:
        block_type = block.type
        tile = block.min_tile
        is_sequence = bool(block_type & BlockType.Sequence)
        is_triplet_or_kong = bool(block_type & (BlockType.Triplet | BlockType.Kong))
        is_pair = bool(block_type & BlockType.Pair)

        if is_sequence:
            seq_counts[tile] += 1
            all_triplets = False
            outside_has_sequence = True
            if tile not in (
                Tile.Manzu1,
                Tile.Manzu7,
                Tile.Pinzu1,
                Tile.Pinzu7,
                Tile.Souzu1,
                Tile.Souzu7,
            ):
                outside_valid = False
        else:
            if tile not in (
                Tile.Manzu1,
                Tile.Manzu9,
                Tile.Pinzu1,
                Tile.Pinzu9,
                Tile.Souzu1,
                Tile.Souzu9,
                Tile.East,
                Tile.South,
                Tile.West,
                Tile.North,
                Tile.White,
                Tile.Green,
                Tile.Red,
            ):
                outside_valid = False
            if tile >= Tile.East:
                outside_has_honor = True

        if is_triplet_or_kong:
            no_triplet_or_kong = False
            triplet_counts[tile] += 1
            if block_type == BlockType.Triplet or block_type == BlockType.Kong:
                num_concealed_triplets += 1
            if block_type & BlockType.Kong:
                num_kongs += 1

            if block_type == (BlockType.Triplet | BlockType.Open):
                block_fu = 2
            elif block_type == BlockType.Triplet:
                block_fu = 4
            elif block_type == (BlockType.Kong | BlockType.Open):
                block_fu = 8
            else:
                block_fu = 16
            fu += block_fu * 2 if _is_terminal_or_honor(tile) else block_fu
        elif is_pair:
            if tile == seat_wind and tile == round_wind:
                fu += 4
            elif tile == seat_wind or tile == round_wind or tile >= Tile.White:
                fu += 2
        else:
            all_triplets = False

    is_pinfu = (
        wait_type == WaitType.DoubleEdgeWait
        and no_triplet_or_kong
        and not _has_value_pair(blocks, round_wind, seat_wind)
    )
    if is_pinfu and is_tsumo and is_closed:
        fu = 20
    elif is_pinfu and (not is_tsumo) and (not is_closed):
        fu = 30

    yaku_list = 0
    if is_closed and is_pinfu:
        yaku_list |= Yaku.Pinfu

    if is_closed:
        num_double = 0
        for count in seq_counts:
            if count == 4:
                num_double += 2
            elif count >= 2:
                num_double += 1
        if num_double == 1:
            yaku_list |= Yaku.PureDoubleSequence
        elif num_double == 2:
            yaku_list |= Yaku.TwicePureDoubleSequence

    if _has_pure_straight(seq_counts):
        yaku_list |= Yaku.PureStraight
    elif _has_triple_triplets(triplet_counts):
        yaku_list |= Yaku.TripleTriplets
    elif _has_mixed_triple_sequence(seq_counts):
        yaku_list |= Yaku.MixedTripleSequence

    if outside_valid and outside_has_sequence:
        yaku_list |= Yaku.HalfOutsideHand if outside_has_honor else Yaku.FullyOutsideHand
    if all_triplets:
        yaku_list |= Yaku.AllTriplets
    if num_concealed_triplets == 3:
        yaku_list |= Yaku.ThreeConcealedTriplets
    if num_kongs == 3:
        yaku_list |= Yaku.ThreeKongs

    return yaku_list, fu


def _has_value_pair(blocks, round_wind: int, seat_wind: int) -> bool:
    for block in blocks:
        if (block.type & BlockType.Pair) and (
            block.min_tile == round_wind
            or block.min_tile == seat_wind
            or block.min_tile >= Tile.White
        ):
            return True
    return False


def _has_pure_straight(seq_counts: list[int]) -> bool:
    return (
        bool(seq_counts[Tile.Manzu1] and seq_counts[Tile.Manzu4] and seq_counts[Tile.Manzu7])
        or bool(seq_counts[Tile.Pinzu1] and seq_counts[Tile.Pinzu4] and seq_counts[Tile.Pinzu7])
        or bool(seq_counts[Tile.Souzu1] and seq_counts[Tile.Souzu4] and seq_counts[Tile.Souzu7])
    )


def _has_triple_triplets(triplet_counts: list[int]) -> bool:
    for i in range(9):
        if triplet_counts[i] and triplet_counts[i + 9] and triplet_counts[i + 18]:
            return True
    return False


def _has_mixed_triple_sequence(seq_counts: list[int]) -> bool:
    for i in range(9):
        if seq_counts[i] and seq_counts[i + 9] and seq_counts[i + 18]:
            return True
    return False


def _is_terminal_or_honor(tile: int) -> bool:
    return tile >= Tile.East or tile in (
        Tile.Manzu1,
        Tile.Manzu9,
        Tile.Pinzu1,
        Tile.Pinzu9,
        Tile.Souzu1,
        Tile.Souzu9,
    )


def _calc_score_fast_or_none(
    config: ExpectedScoreConfig,
    round_: Round,
    player: Player,
    hand_counts: list[int],
    wall_counts: list[int],
    shanten_type: int,
    win_tile: int,
    riichi: bool,
) -> float | None:
    if player.melds:
        return None

    win_flag = WinFlag.Tsumo | WinFlag.Riichi if riichi else WinFlag.Tsumo
    yaku_list = _check_not_pattern_yaku_numba(
        player.hand,
        round_.wind,
        player.wind,
        win_tile,
        win_flag,
        shanten_type,
    )

    if yaku_list & (Yaku.NagashiMangan | Yakuman):
        return None

    pattern_yaku, fu = _check_pattern_yaku_fast(round_, player, win_tile, win_flag, shanten_type)
    yaku_list |= pattern_yaku
    if yaku_list == 0:
        return 0.0

    han = _normal_han_closed_numba(yaku_list)

    han += _count_dora_closed_hand(player.hand, round_.dora_indicators)
    han += _count_dora_closed_hand(player.hand, round_.uradora_indicators)
    if round_.rules & RuleFlag.RedDora:
        han += int(player.hand[Tile.RedManzu5] + player.hand[Tile.RedPinzu5] + player.hand[Tile.RedSouzu5])

    is_dealer = player.wind == Tile.East
    score_title = _score_title_cached(fu, han)
    base_score = _score_total_cached(
        is_dealer,
        True,
        round_.honba,
        round_.kyotaku,
        score_title,
        han,
        fu,
    )

    if (
        not config.enable_uradora
        or not (win_flag & WinFlag.Riichi)
        or not round_.dora_indicators
    ):
        return float(base_score)

    if score_title >= ScoreTitle.CountedYakuman:
        return float(base_score)

    num_indicators = len(round_.dora_indicators)
    up_scores = _get_up_scores_fast(round_, player, win_flag, han, fu, num_indicators)

    if num_indicators == 1:
        wall = list(wall_counts)
        hand_and_melds = list(hand_counts)
        for red_tile, base_tile in _RED_TO_BASE.items():
            wall[base_tile] += wall[red_tile]
            hand_and_melds[base_tile] += hand_and_melds[red_tile]

        indicator_counts = [0.0] * 5
        for tile in range(34):
            num = hand_and_melds[tile]
            if 0 <= num <= 4:
                indicator_counts[num] += wall[ToIndicator[tile]]

        if config.sum <= 0:
            return float(base_score)
        score = 0.0
        for i in range(5):
            score += up_scores[i] * indicator_counts[i] / config.sum
        return score

    table = ExpectedScoreCalculator._load_uradora_table()
    table_idx = min(num_indicators, len(table) - 1)
    score = 0.0
    for i in range(13):
        score += up_scores[i] * table[table_idx][i]
    return score


@lru_cache(maxsize=512)
def _score_title_cached(fu: int, han: int) -> int:
    return int(ScoreCalculator.get_score_title(fu, han))


@lru_cache(maxsize=4096)
def _score_total_cached(
    is_dealer: bool,
    is_tsumo: bool,
    honba: int,
    kyotaku: int,
    score_title: int,
    han: int,
    fu: int,
) -> int:
    return ScoreCalculator.calc_score(
        is_dealer,
        is_tsumo,
        honba,
        kyotaku,
        score_title,
        han,
        fu,
    )[0]


def _get_up_scores_fast(
    round_: Round,
    player: Player,
    win_flag: int,
    han: int,
    fu: int,
    num_indicators: int,
) -> tuple[int, ...]:
    return _get_up_scores_cached(
        player.wind == Tile.East,
        bool(win_flag & WinFlag.Tsumo),
        round_.honba,
        round_.kyotaku,
        han,
        fu,
        num_indicators,
    )


@lru_cache(maxsize=2048)
def _get_up_scores_cached(
    is_dealer: bool,
    is_tsumo: bool,
    honba: int,
    kyotaku: int,
    han: int,
    fu: int,
    num_indicators: int,
) -> tuple[int, ...]:
    n = 4 if num_indicators == 1 else 12
    scores = [0] * (n + 1)
    for i in range(n + 1):
        next_han = han + i
        score_title = _score_title_cached(fu, next_han)
        scores[i] = _score_total_cached(
            is_dealer,
            is_tsumo,
            honba,
            kyotaku,
            score_title,
            next_han,
            fu,
        )
    return tuple(scores)


def _necessary_mask(
    hand34: np.ndarray,
    num_melds: int,
    type_mask: int,
) -> tuple[int, int, int]:
    suits_raw, honors_raw = _get_tables()
    best_type, best_shanten, best_mask = _necessary_mask_numba(
        hand34,
        num_melds,
        type_mask,
        suits_raw,
        honors_raw,
    )
    return int(best_type), int(best_shanten), int(best_mask)


def _necessary_mask_cached(
    player: Player,
    num_melds: int,
    type_mask: int,
    cache: MaskCache,
) -> tuple[int, int, int]:
    key = (_hand34_key(player), num_melds, type_mask)
    cached = cache.get(key)
    if cached is None:
        cached = _necessary_mask(player.hand[:34], num_melds, type_mask)
        cache[key] = cached
    return cached


def _unnecessary_mask(
    hand34: np.ndarray,
    num_melds: int,
    type_mask: int,
) -> tuple[int, int, int]:
    suits_raw, honors_raw = _get_tables()
    best_type, best_shanten, best_mask = _unnecessary_mask_numba(
        hand34,
        num_melds,
        type_mask,
        suits_raw,
        honors_raw,
    )
    return int(best_type), int(best_shanten), int(best_mask)


def _unnecessary_mask_cached(
    player: Player,
    num_melds: int,
    type_mask: int,
    cache: MaskCache,
) -> tuple[int, int, int]:
    key = (_hand34_key(player), num_melds, type_mask)
    cached = cache.get(key)
    if cached is None:
        cached = _unnecessary_mask(player.hand[:34], num_melds, type_mask)
        cache[key] = cached
    return cached


# Precomputed red tile expansion: for each base tile (0-33), which red tile to add
_RED_EXPANSION = [0] * 34
for _base, _red in _BASE_TO_RED.items():
    _RED_EXPANSION[_base] = 1 << _red


def _expand_red_mask(mask: int) -> int:
    if mask & 0x1F:  # manzu tiles (0-8): check bits 0-8 for any set manzu5 bit
        if mask & (1 << Tile.Manzu5):
            mask |= 1 << Tile.RedManzu5
    if mask & 0x3E00:  # pinzu tiles (9-17): check bits 9-17 for any set pinzu5 bit
        if mask & (1 << Tile.Pinzu5):
            mask |= 1 << Tile.RedPinzu5
    if mask & 0x7C0000:  # souzu tiles (18-26): check bits 18-26 for any set souzu5 bit
        if mask & (1 << Tile.Souzu5):
            mask |= 1 << Tile.RedSouzu5
    return mask
