"""计分/牌型相关的数据结构。

本文件定义：
- 面子/雀头块（`Block`）与其类型（`BlockType`）
- 副露（`Meld`）与其类型（`MeldType`）
- 玩家/局信息（`Player`/`Round`）
- 计分结果（`Result`）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .constants import Tile


class BlockType:
    """块类型（面子/雀头）以及是否副露。"""
    Null = 0
    Triplet = 1
    Sequence = 2
    Kong = 4
    Pair = 8
    Open = 16


class MeldType:
    """副露类型。"""
    Null = -1
    Pong = 0
    Chow = 1
    ClosedKong = 2
    OpenKong = 3
    AddedKong = 4


class PlayerType:
    """被鸣牌/放铳的玩家索引。"""
    Null = -1
    Player0 = 0
    Player1 = 1
    Player2 = 2
    Player3 = 3


class WaitType:
    """等待形（两面/边张/嵌张/双碰/单骑）。"""
    Null = -1
    DoubleEdgeWait = 0
    EdgeWait = 1
    ClosedWait = 2
    TripletWait = 3
    PairWait = 4


class RuleFlag:
    """规则开关（赤宝牌/食断等）。"""
    Null = 0
    RedDora = 1
    OpenTanyao = 2


@dataclass
class Block:
    """一个块：顺子/刻子/杠子/雀头。"""
    type: int = BlockType.Null
    min_tile: int = Tile.Null


@dataclass
class Meld:
    """副露信息。"""
    type: int = MeldType.Null
    tiles: List[int] = field(default_factory=list)
    discarded_tile: int = Tile.Null
    from_player: int = PlayerType.Null

    def __post_init__(self) -> None:
        if self.tiles and self.discarded_tile == Tile.Null:
            self.discarded_tile = self.tiles[0]


@dataclass
class Player:
    """玩家手牌与副露状态。"""
    hand: np.ndarray = field(default_factory=lambda: np.zeros(37, dtype=np.int16))
    melds: List[Meld] = field(default_factory=list)
    wind: int = Tile.Null

    def num_tiles(self) -> int:
        """返回手牌（不含赤 flag）张数。"""
        return int(self.hand[:34].sum())

    def num_melds(self) -> int:
        """返回副露数量。"""
        return len(self.melds)

    def is_closed(self) -> bool:
        """是否门前清（仅允许暗杠）。"""
        for meld in self.melds:
            if meld.type != MeldType.ClosedKong:
                return False
        return True


def hand_from_tiles(tiles: List[int]) -> np.ndarray:
    """将牌 id 列表转换为长度 37 的计数数组（包含赤宝牌 flag）。"""
    hand = np.zeros(37, dtype=np.int16)
    for tile in tiles:
        if tile == Tile.RedManzu5:
            hand[Tile.Manzu5] += 1
        elif tile == Tile.RedPinzu5:
            hand[Tile.Pinzu5] += 1
        elif tile == Tile.RedSouzu5:
            hand[Tile.Souzu5] += 1
        hand[tile] += 1
    return hand


@dataclass
class Round:
    """局信息与规则开关。"""
    rules: int = RuleFlag.RedDora | RuleFlag.OpenTanyao
    wind: int = Tile.Null
    kyoku: int = 1
    honba: int = 0
    kyotaku: int = 0
    dora_indicators: List[int] = field(default_factory=list)
    uradora_indicators: List[int] = field(default_factory=list)

    def set_dora(self, tiles: List[int]) -> None:
        """设置宝牌指示牌列表（输入为宝牌 id，内部映射为指示牌）。"""
        from .score_constants import ToIndicator

        for t in tiles:
            self.dora_indicators.append(ToIndicator[t])

    def set_dora_indicators(self, tiles: List[int]) -> None:
        """设置宝牌指示牌列表（输入为指示牌 id，直接存储）。"""
        for t in tiles:
            self.dora_indicators.append(t)

    def set_uradora(self, tiles: List[int]) -> None:
        """设置里宝牌指示牌列表（输入为宝牌 id，内部映射为指示牌）。"""
        from .score_constants import ToIndicator

        for t in tiles:
            self.uradora_indicators.append(ToIndicator[t])

    def set_uradora_indicators(self, tiles: List[int]) -> None:
        """设置里宝牌指示牌列表（输入为指示牌 id，直接存储）。"""
        for t in tiles:
            self.uradora_indicators.append(t)


@dataclass
class Result:
    """计分结果。

    - `yaku_list`: [(役, 番)]
    - `score`: 点数列表（格式由自摸/荣和与是否庄家决定）
    - `blocks`/`wait_type`: 最优分解形与等待形（用于解释/调试）
    """
    success: bool
    player: Player
    win_tile: int
    win_flag: int
    yaku_list: List[Tuple[int, int]] = field(default_factory=list)
    han: int = 0
    fu: int = 0
    score_title: int = -1
    score: List[int] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    wait_type: int = WaitType.Null
    err_msg: str = ""
