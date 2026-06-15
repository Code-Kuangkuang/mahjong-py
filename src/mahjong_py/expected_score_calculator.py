"""手牌期望值计算。

迁移自 mahjong-cpp 的 ExpectedScoreCalculator：
- 以摸牌节点 / 打牌节点构建手牌转移图
- 倒推计算听牌概率、和牌概率、点数期望
- 支持赤宝牌、手变、向听回退与立直后的里宝牌期望
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import struct
import sys
from typing import ClassVar

import numpy as np

from .constants import Tile
from .necessary import necessary_tiles
from .score_calculator import ScoreCalculator
from .score_constants import ScoreTitle, ToIndicator, WinFlag
from .score_types import Meld, Player, Round
from .shanten import ShantenFlag, shanten
from .unnecessary import unnecessary_tiles
from .utils import is_reddora, to_no_reddora


Count = list[int]
CountRed = list[int]

_RED_TO_BASE = {
    Tile.RedManzu5: Tile.Manzu5,
    Tile.RedPinzu5: Tile.Pinzu5,
    Tile.RedSouzu5: Tile.Souzu5,
}
_BASE_TO_RED = {
    Tile.Manzu5: Tile.RedManzu5,
    Tile.Pinzu5: Tile.RedPinzu5,
    Tile.Souzu5: Tile.RedSouzu5,
}


@dataclass
class ExpectedScoreConfig:
    """期望值计算配置。"""

    t_min: int = 1
    t_max: int = 18
    sum: int = 0
    extra: int = 0
    shanten_type: int = ShantenFlag.All
    enable_reddora: bool = True
    enable_uradora: bool = True
    enable_shanten_down: bool = True
    enable_tegawari: bool = True
    enable_riichi: bool = False
    calc_stats: bool = True
    use_numba: bool = True  # True = use Numba accelerator


@dataclass
class ExpectedScoreStat:
    """单个候选手牌状态的统计结果。"""

    tile: int
    tenpai_prob: list[float]
    win_prob: list[float]
    exp_score: list[float]
    necessary_tiles: list[tuple[int, int]]
    shanten: int


@dataclass
class _VertexData:
    tenpai_prob: list[float]
    win_prob: list[float]
    exp_score: list[float]

    @classmethod
    def create(
        cls,
        size: int,
        tenpai_init: float,
        win_init: float,
        exp_init: float,
    ) -> "_VertexData":
        return cls(
            [tenpai_init] * size,
            [win_init] * size,
            [exp_init] * size,
        )


class _Graph:
    """最小图结构，保留 C++ 版需要的出边、入边与去重行为。"""

    def __init__(self) -> None:
        self.vertices: list[_VertexData] = []
        self.out_edges: dict[int, list[tuple[int, int, float]]] = {}
        self.in_edges: dict[int, list[tuple[int, int, float]]] = {}
        self.edge_keys: set[tuple[int, int]] = set()

    def add_vertex(self, data: _VertexData) -> int:
        vertex = len(self.vertices)
        self.vertices.append(data)
        return vertex

    def add_edge(self, source: int, target: int, weight: int, score: float) -> bool:
        key = (source, target)
        if key in self.edge_keys:
            return False
        self.edge_keys.add(key)
        self.out_edges.setdefault(source, []).append((target, weight, score))
        self.in_edges.setdefault(target, []).append((source, weight, score))
        return True

    def has_edge(self, source: int, target: int) -> bool:
        return (source, target) in self.edge_keys


class ExpectedScoreCalculator:
    """听牌概率、和牌概率与点数期望计算器。"""

    _uradora_table: ClassVar[list[list[float]] | None] = None

    @staticmethod
    def create_wall(round_: Round, player: Player, enable_reddora: bool) -> Count:
        """根据手牌、副露与宝牌指示牌估计剩余牌山。"""
        wall = [0] * Tile.Length
        melds = [0] * Tile.Length
        indicators = [0] * Tile.Length

        for tile in round_.dora_indicators:
            indicators[to_no_reddora(tile)] += 1
            if is_reddora(tile):
                indicators[tile] += 1

        for meld in player.melds:
            for tile in meld.tiles:
                melds[to_no_reddora(tile)] += 1
                if is_reddora(tile):
                    melds[tile] += 1

        for tile in range(34):
            wall[tile] = 4 - (
                int(player.hand[tile]) + melds[tile] + indicators[tile]
            )

        if enable_reddora:
            for tile in range(34, Tile.Length):
                wall[tile] = 1 - (
                    int(player.hand[tile]) + melds[tile] + indicators[tile]
                )

        return wall

    @staticmethod
    def calc(
        config: ExpectedScoreConfig,
        round_: Round,
        player: Player,
        wall: Count | None = None,
    ) -> tuple[list[ExpectedScoreStat], int]:
        """计算候选打牌/当前 13 张状态的概率与期望。"""
        if sys.getrecursionlimit() < 100_000:
            sys.setrecursionlimit(100_000)

        graph = _Graph()
        draw_cache: dict[tuple[int, ...], int] = {}
        discard_cache: dict[tuple[int, ...], int] = {}

        cfg = replace(config)
        wall_counts_normal = (
            list(wall)
            if wall is not None
            else ExpectedScoreCalculator.create_wall(round_, player, cfg.enable_reddora)
        )
        if cfg.sum == 0:
            cfg.sum = sum(wall_counts_normal[:34])

        # C++ 版目前强制开启听牌立直，以避免默听路径带来的评估偏差。
        cfg.enable_riichi = True

        work_player = ExpectedScoreCalculator._copy_player(player)
        hand_counts = ExpectedScoreCalculator._encode(work_player.hand, cfg.enable_reddora)
        wall_counts = ExpectedScoreCalculator._encode(wall_counts_normal, cfg.enable_reddora)
        hand_origin = list(hand_counts)
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
                ExpectedScoreCalculator._draw_node(
                    cfg,
                    round_,
                    work_player,
                    graph,
                    draw_cache,
                    discard_cache,
                    hand_counts,
                    wall_counts,
                    hand_origin,
                    shanten_org,
                    riichi,
                )
                ExpectedScoreCalculator._calc_stats(cfg, graph, draw_cache, discard_cache)
                vertex = draw_cache.get(tuple(hand_counts))
                if vertex is not None:
                    state = graph.vertices[vertex]
                    shanten2, needed = ExpectedScoreCalculator._get_necessary_tiles(
                        cfg, work_player, wall_counts_normal
                    )
                    stats.append(
                        ExpectedScoreStat(
                            Tile.Null,
                            list(state.tenpai_prob),
                            list(state.win_prob),
                            list(state.exp_score),
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
                ExpectedScoreCalculator._discard_node(
                    cfg,
                    round_,
                    work_player,
                    graph,
                    draw_cache,
                    discard_cache,
                    hand_counts,
                    wall_counts,
                    hand_origin,
                    shanten_org,
                    riichi,
                )
                ExpectedScoreCalculator._calc_stats(cfg, graph, draw_cache, discard_cache)

                for tile in range(Tile.Length):
                    if hand_counts[tile] <= 0:
                        continue
                    ExpectedScoreCalculator._discard(
                        work_player, hand_counts, wall_counts, tile
                    )
                    vertex = draw_cache.get(tuple(hand_counts))
                    if vertex is not None:
                        state = graph.vertices[vertex]
                        shanten2, needed = ExpectedScoreCalculator._get_necessary_tiles(
                            cfg, work_player, wall_counts_normal
                        )
                        stats.append(
                            ExpectedScoreStat(
                                tile,
                                list(state.tenpai_prob),
                                list(state.win_prob),
                                list(state.exp_score),
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

        return stats, len(graph.vertices)

    @staticmethod
    def _copy_player(player: Player) -> Player:
        melds = [
            Meld(
                type=meld.type,
                tiles=list(meld.tiles),
                discarded_tile=meld.discarded_tile,
                from_player=meld.from_player,
            )
            for meld in player.melds
        ]
        return Player(hand=player.hand.astype(np.int16, copy=True), melds=melds, wind=player.wind)

    @staticmethod
    def _encode(counts, enable_reddora: bool) -> CountRed:
        """把普通 37 计数转成赤 5 与非赤 5 分离的表示。"""
        encoded = [0] * Tile.Length
        for tile in range(34):
            encoded[tile] = int(counts[tile])

        if enable_reddora:
            for red_tile, base_tile in _RED_TO_BASE.items():
                num_red = int(counts[red_tile])
                if num_red:
                    encoded[base_tile] -= num_red
                    encoded[red_tile] += num_red
        return encoded

    @staticmethod
    def _distance(hand: CountRed, origin: CountRed) -> int:
        dist = 0
        for tile in range(Tile.Length):
            if hand[tile] > origin[tile]:
                dist += hand[tile] - origin[tile]
        return dist

    @staticmethod
    def _draw(
        player: Player,
        hand_counts: CountRed,
        wall_counts: CountRed,
        tile: int,
    ) -> None:
        hand_counts[tile] += 1
        wall_counts[tile] -= 1

        player.hand[tile] += 1
        if tile in _RED_TO_BASE:
            player.hand[_RED_TO_BASE[tile]] += 1

    @staticmethod
    def _discard(
        player: Player,
        hand_counts: CountRed,
        wall_counts: CountRed,
        tile: int,
    ) -> None:
        hand_counts[tile] -= 1
        wall_counts[tile] += 1

        player.hand[tile] -= 1
        if tile in _RED_TO_BASE:
            player.hand[_RED_TO_BASE[tile]] -= 1

    @staticmethod
    def _draw_node(
        config: ExpectedScoreConfig,
        round_: Round,
        player: Player,
        graph: _Graph,
        draw_cache: dict[tuple[int, ...], int],
        discard_cache: dict[tuple[int, ...], int],
        hand_counts: CountRed,
        wall_counts: CountRed,
        hand_origin: CountRed,
        shanten_org: int,
        riichi: bool,
    ) -> int:
        key = tuple(hand_counts)
        if key in draw_cache:
            return draw_cache[key]

        shanten_type, shanten_num, wait_tiles = necessary_tiles(
            player.hand[:34],
            player.num_melds(),
            config.shanten_type,
        )
        allow_tegawari = (
            config.enable_tegawari
            and ExpectedScoreCalculator._distance(hand_counts, hand_origin)
            + shanten_num
            < shanten_org + config.extra
        )
        wait_set = ExpectedScoreCalculator._expand_red_tiles(wait_tiles)

        vertex_data = _VertexData.create(config.t_max + 1, 0.0, 0.0, 0.0)
        vertex_data.tenpai_prob[config.t_max] = 1.0 if shanten_num == 0 else 0.0
        vertex = graph.add_vertex(vertex_data)
        draw_cache[key] = vertex

        for tile in range(Tile.Length):
            is_wait = tile in wait_set
            if wall_counts[tile] <= 0 or not (allow_tegawari or is_wait):
                continue

            weight = wall_counts[tile]
            ExpectedScoreCalculator._draw(player, hand_counts, wall_counts, tile)

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
            target = ExpectedScoreCalculator._discard_node(
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
            if not graph.has_edge(vertex, target):
                score = (
                    ExpectedScoreCalculator._calc_score(
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
        graph: _Graph,
        draw_cache: dict[tuple[int, ...], int],
        discard_cache: dict[tuple[int, ...], int],
        hand_counts: CountRed,
        wall_counts: CountRed,
        hand_origin: CountRed,
        shanten_org: int,
        riichi: bool,
    ) -> int:
        key = tuple(hand_counts)
        if key in discard_cache:
            return discard_cache[key]

        shanten_type, shanten_num, discard_tiles = unnecessary_tiles(
            player.hand[:34],
            player.num_melds(),
            config.shanten_type,
        )
        allow_shanten_down = (
            config.enable_shanten_down
            and ExpectedScoreCalculator._distance(hand_counts, hand_origin)
            + shanten_num
            < shanten_org + config.extra
        )
        discard_set = ExpectedScoreCalculator._expand_red_tiles(discard_tiles)

        vertex = graph.add_vertex(
            _VertexData.create(
                config.t_max + 1,
                1.0 if shanten_num == 0 else 0.0,
                1.0 if shanten_num == -1 else 0.0,
                0.0,
            )
        )
        discard_cache[key] = vertex

        for tile in range(Tile.Length):
            is_discard = tile in discard_set
            if hand_counts[tile] <= 0 or not (allow_shanten_down or is_discard):
                continue

            ExpectedScoreCalculator._discard(player, hand_counts, wall_counts, tile)
            weight = wall_counts[tile]
            source = ExpectedScoreCalculator._draw_node(
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

            if not graph.has_edge(source, vertex):
                score = (
                    ExpectedScoreCalculator._calc_score(
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
        graph: _Graph,
        draw_cache: dict[tuple[int, ...], int],
        discard_cache: dict[tuple[int, ...], int],
    ) -> None:
        for turn in range(config.t_max, config.t_min - 1, -1):
            if turn < config.t_max:
                denom = config.sum - turn
                for vertex in draw_cache.values():
                    state = graph.vertices[vertex]
                    for target, weight, score in graph.out_edges.get(vertex, []):
                        target_state = graph.vertices[target]
                        state.tenpai_prob[turn] += weight * (
                            target_state.tenpai_prob[turn + 1]
                            - state.tenpai_prob[turn + 1]
                        )
                        state.win_prob[turn] += weight * (
                            target_state.win_prob[turn + 1]
                            - state.win_prob[turn + 1]
                        )
                        state.exp_score[turn] += weight * (
                            max(float(score), target_state.exp_score[turn + 1])
                            - state.exp_score[turn + 1]
                        )

                    if denom > 0:
                        state.tenpai_prob[turn] = (
                            state.tenpai_prob[turn + 1]
                            + state.tenpai_prob[turn] / denom
                        )
                        state.win_prob[turn] = (
                            state.win_prob[turn + 1] + state.win_prob[turn] / denom
                        )
                        state.exp_score[turn] = (
                            state.exp_score[turn + 1] + state.exp_score[turn] / denom
                        )
                    else:
                        state.tenpai_prob[turn] = state.tenpai_prob[turn + 1]
                        state.win_prob[turn] = state.win_prob[turn + 1]
                        state.exp_score[turn] = state.exp_score[turn + 1]

            for vertex in discard_cache.values():
                state = graph.vertices[vertex]
                for source, _weight, _score in graph.in_edges.get(vertex, []):
                    source_state = graph.vertices[source]
                    state.tenpai_prob[turn] = max(
                        state.tenpai_prob[turn],
                        source_state.tenpai_prob[turn],
                    )
                    state.win_prob[turn] = max(
                        state.win_prob[turn],
                        source_state.win_prob[turn],
                    )
                    state.exp_score[turn] = max(
                        state.exp_score[turn],
                        source_state.exp_score[turn],
                    )

    @staticmethod
    def _calc_score(
        config: ExpectedScoreConfig,
        round_: Round,
        player: Player,
        hand_counts: CountRed,
        wall_counts: CountRed,
        shanten_type: int,
        win_tile: int,
        riichi: bool,
    ) -> float:
        win_flag = (
            WinFlag.Tsumo | WinFlag.Riichi
            if riichi
            else WinFlag.Tsumo
        )
        result = ScoreCalculator.calc_fast(round_, player, win_tile, win_flag, shanten_type)
        if not result.success:
            return 0.0

        if (
            not config.enable_uradora
            or not (win_flag & WinFlag.Riichi)
            or not round_.dora_indicators
        ):
            return float(result.score[0])

        if result.score_title >= ScoreTitle.CountedYakuman:
            return float(result.score[0])

        num_indicators = len(round_.dora_indicators)
        if num_indicators == 1:
            wall = list(wall_counts)
            hand_and_melds = list(hand_counts)
            for red_tile, base_tile in _RED_TO_BASE.items():
                wall[base_tile] += wall[red_tile]
                hand_and_melds[base_tile] += hand_and_melds[red_tile]

            for meld in player.melds:
                for tile in meld.tiles:
                    hand_and_melds[to_no_reddora(tile)] += 1

            up_scores = ScoreCalculator.get_up_scores(round_, player, result, win_flag, 4)
            indicator_counts = [0.0] * 5
            for tile in range(34):
                num = hand_and_melds[tile]
                if 0 <= num <= 4:
                    indicator_counts[num] += wall[ToIndicator[tile]]

            if config.sum <= 0:
                return float(result.score[0])
            score = 0.0
            for i in range(5):
                score += up_scores[i] * indicator_counts[i] / config.sum
            return score

        table = ExpectedScoreCalculator._load_uradora_table()
        table_idx = min(num_indicators, len(table) - 1)
        up_scores = ScoreCalculator.get_up_scores(round_, player, result, win_flag, 12)
        score = 0.0
        for i in range(13):
            score += up_scores[i] * table[table_idx][i]
        return score

    @staticmethod
    def _get_necessary_tiles(
        config: ExpectedScoreConfig,
        player: Player,
        wall: Count,
    ) -> tuple[int, list[tuple[int, int]]]:
        _type, shanten_num, tiles = necessary_tiles(
            player.hand[:34],
            player.num_melds(),
            config.shanten_type,
        )
        return shanten_num, [(tile, wall[tile]) for tile in tiles]

    @staticmethod
    def _expand_red_tiles(tiles: list[int]) -> set[int]:
        expanded = set(int(tile) for tile in tiles)
        for base_tile, red_tile in _BASE_TO_RED.items():
            if base_tile in expanded:
                expanded.add(red_tile)
        return expanded

    @staticmethod
    def _load_uradora_table() -> list[list[float]]:
        if ExpectedScoreCalculator._uradora_table is not None:
            return ExpectedScoreCalculator._uradora_table

        path = ExpectedScoreCalculator._find_data_file("uradora.bin")
        data = path.read_bytes()
        expected_bytes = 6 * 13 * 8
        if len(data) != expected_bytes:
            raise IOError(f"Invalid uradora table size: {path}")

        values = struct.unpack("<" + "d" * (6 * 13), data)
        table = [list(values[i * 13 : (i + 1) * 13]) for i in range(6)]
        ExpectedScoreCalculator._uradora_table = table
        return table

    @staticmethod
    def _find_data_file(name: str) -> Path:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "data" / "config" / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"data/config/{name} not found")
