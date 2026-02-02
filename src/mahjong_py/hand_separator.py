"""和了形分解（面子/雀头枚举）。

该模块将玩家手牌与副露合并后，枚举可能的 `Block` 组合（最多 4 面子 + 1 雀头），
并根据和牌张与自摸/荣和推断等待形（两面/边张/嵌张/双碰/单骑）。

分解表来源于 `mahjong-py/data/config/*_patterns.json`。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from .score_constants import WinFlag
from .score_types import Block, BlockType, MeldType, WaitType
from .utils import to_no_reddora
from .constants import Tile


def _repo_root() -> Path:
    """返回仓库根目录（.../mahjong-py）。"""
    return Path(__file__).resolve().parents[2]


def _config_dir() -> Path:
    """返回分解 patterns 的 config 目录（`mahjong-py/data/config`）。"""

    cfg = _repo_root() / "data" / "config"
    if not (cfg / "suits_patterns.json").exists() or not (cfg / "honors_patterns.json").exists():
        raise FileNotFoundError(
            "patterns json not found under mahjong-py/data/config: expected suits_patterns.json and honors_patterns.json"
        )
    return cfg


def _load_table(path: Path) -> Dict[int, List[List[Block]]]:
    """从 JSON 读取分解表：key 为 8 进制压缩计数，value 为若干 blocks 组合。"""
    table: Dict[int, List[List[Block]]] = {}
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    for v in doc:
        key = int(v["key"])
        patterns: List[List[Block]] = []
        for s in v["pattern"]:
            blocks: List[Block] = []
            for i in range(0, len(s), 2):
                min_tile = ord(s[i]) - ord("0")
                t = s[i + 1]
                if t == "k":
                    btype = BlockType.Triplet
                elif t == "s":
                    btype = BlockType.Sequence
                else:
                    btype = BlockType.Pair
                blocks.append(Block(btype, min_tile))
            patterns.append(blocks)
        table[key] = patterns

    return table


@lru_cache(maxsize=1)
def _tables() -> Tuple[Dict[int, List[List[Block]]], Dict[int, List[List[Block]]]]:
    """懒加载（带缓存）的数牌/字牌分解表。"""
    cfg = _config_dir()
    return _load_table(cfg / "suits_patterns.json"), _load_table(cfg / "honors_patterns.json")


def _hash8(counts: List[int]) -> int:
    """将计数数组按 8 进制压缩为整数 key（每张牌一位，范围 0..4）。"""
    h = 0
    for c in counts:
        h = h * 8 + c
    return h


def separate(player, win_tile: int, win_flag: int):
    """枚举玩家当前和了形的所有可能分解。

    Returns:
        List[(blocks, wait_type)]
    """
    s_tbl, z_tbl = _tables()

    blocks: List[Block] = [Block() for _ in range(5)]
    i = 0

    # add meld blocks
    for meld in player.melds:
        if meld.type == MeldType.Pong:
            blocks[i].type = BlockType.Triplet | BlockType.Open
        elif meld.type == MeldType.Chow:
            blocks[i].type = BlockType.Sequence | BlockType.Open
        elif meld.type == MeldType.ClosedKong:
            blocks[i].type = BlockType.Kong
        else:
            blocks[i].type = BlockType.Kong | BlockType.Open
        blocks[i].min_tile = to_no_reddora(meld.tiles[0])
        i += 1

    hand = player.hand

    manzu_hash = _hash8([int(x) for x in hand[0:9]])
    pinzu_hash = _hash8([int(x) for x in hand[9:18]])
    souzu_hash = _hash8([int(x) for x in hand[18:27]])
    honors_hash = _hash8([int(x) for x in hand[27:34]])

    manzu = s_tbl.get(manzu_hash, [])
    pinzu = s_tbl.get(pinzu_hash, [])
    souzu = s_tbl.get(souzu_hash, [])
    honors = z_tbl.get(honors_hash, [])

    pattern: List[Tuple[List[Block], int]] = []
    nored_win_tile = to_no_reddora(win_tile)
    _create_block_patterns(
        nored_win_tile,
        bool(win_flag & WinFlag.Tsumo),
        pattern,
        blocks,
        i,
        0,
        manzu,
        pinzu,
        souzu,
        honors,
    )

    return pattern


def _create_block_patterns(win_tile: int, tsumo: bool, pattern, blocks, i, d, manzu, pinzu, souzu, honors):
    """递归拼接三门数牌与字牌的分解，并在末端根据和牌张推断等待形。"""
    if d == 4:
        for idx, b in enumerate(blocks):
            if b.type & BlockType.Open:
                continue

            if (b.type & BlockType.Triplet) and b.min_tile == win_tile:
                _push_pattern(pattern, blocks, WaitType.TripletWait, idx, tsumo)
            elif b.type == BlockType.Sequence and b.min_tile + 1 == win_tile:
                _push_pattern(pattern, blocks, WaitType.ClosedWait, idx, tsumo)
            elif b.type == BlockType.Sequence and b.min_tile + 2 == win_tile and b.min_tile in (Tile.Manzu1, Tile.Pinzu1, Tile.Souzu1):
                _push_pattern(pattern, blocks, WaitType.EdgeWait, idx, tsumo)
            elif b.type == BlockType.Sequence and b.min_tile == win_tile and b.min_tile in (Tile.Manzu7, Tile.Pinzu7, Tile.Souzu7):
                _push_pattern(pattern, blocks, WaitType.EdgeWait, idx, tsumo)
            elif b.type == BlockType.Sequence and (b.min_tile == win_tile or b.min_tile + 2 == win_tile):
                _push_pattern(pattern, blocks, WaitType.DoubleEdgeWait, idx, tsumo)
            elif b.type == BlockType.Pair and b.min_tile == win_tile:
                _push_pattern(pattern, blocks, WaitType.PairWait, idx, tsumo)
        return

    if d == 0:
        if not manzu:
            _create_block_patterns(win_tile, tsumo, pattern, blocks, i, d + 1, manzu, pinzu, souzu, honors)
        for pat in manzu:
            for b in pat:
                blocks[i] = b
                i += 1
            _create_block_patterns(win_tile, tsumo, pattern, blocks, i, d + 1, manzu, pinzu, souzu, honors)
            i -= len(pat)
    elif d == 1:
        if not pinzu:
            _create_block_patterns(win_tile, tsumo, pattern, blocks, i, d + 1, manzu, pinzu, souzu, honors)
        for pat in pinzu:
            for b in pat:
                blocks[i] = Block(b.type, b.min_tile + 9)
                i += 1
            _create_block_patterns(win_tile, tsumo, pattern, blocks, i, d + 1, manzu, pinzu, souzu, honors)
            i -= len(pat)
    elif d == 2:
        if not souzu:
            _create_block_patterns(win_tile, tsumo, pattern, blocks, i, d + 1, manzu, pinzu, souzu, honors)
        for pat in souzu:
            for b in pat:
                blocks[i] = Block(b.type, b.min_tile + 18)
                i += 1
            _create_block_patterns(win_tile, tsumo, pattern, blocks, i, d + 1, manzu, pinzu, souzu, honors)
            i -= len(pat)
    elif d == 3:
        if not honors:
            _create_block_patterns(win_tile, tsumo, pattern, blocks, i, d + 1, manzu, pinzu, souzu, honors)
        for pat in honors:
            for b in pat:
                blocks[i] = Block(b.type, b.min_tile + 27)
                i += 1
            _create_block_patterns(win_tile, tsumo, pattern, blocks, i, d + 1, manzu, pinzu, souzu, honors)
            i -= len(pat)


def _push_pattern(pattern, blocks, wait_type, win_block_index: int, tsumo: bool):
    """将一个分解结果写入输出列表（荣和时会将和牌所在块标记为副露）。"""
    copied = [Block(b.type, b.min_tile) for b in blocks]
    if not tsumo:
        copied[win_block_index].type |= BlockType.Open
    pattern.append((copied, wait_type))
