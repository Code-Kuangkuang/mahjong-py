"""通用工具函数。

主要包含：
- 互斥 flag 检查
- 赤宝牌与普通牌的映射
- 牌的基本类型判断（幺九/字牌/断幺等）
"""

from __future__ import annotations

from .constants import Tile


# 判断是否为 0 或 2 的幂
def check_exclusive(x: int) -> bool:
    """检查位标志是否“至多只包含一个 bit”。

    用于验证多个 flag 之间是否互斥：
    - x == 0：未指定
    - x 是 2 的幂：仅指定了一个 flag
    """
    return (x == 0) or (x & (x - 1) == 0)


# 赤5 -> 普通5 的映射表
_REDDORA_MAP = {
    Tile.RedManzu5: Tile.Manzu5,
    Tile.RedPinzu5: Tile.Pinzu5,
    Tile.RedSouzu5: Tile.Souzu5,
}


# 将赤5映射为普通5，其它牌保持不变
def to_no_reddora(tile: int) -> int:
    """将赤 5 映射为对应的普通 5（其它牌不变）。"""
    return _REDDORA_MAP.get(tile, tile)


# 是否为赤宝牌
def is_reddora(tile: int) -> bool:
    """是否为赤宝牌（0m/0p/0s）。"""
    return tile >= Tile.RedManzu5


# 是否为幺九牌（数牌 1/9，不含字牌）
def is_terminal(tile: int) -> bool:
    """是否为序数幺九牌（1/9；不包含字牌）。"""
    n = tile % 9
    return tile <= Tile.Souzu9 and (n == 0 or n == 8)


# 是否为字牌
def is_honor(tile: int) -> bool:
    """是否为字牌（风牌 + 三元牌）。"""
    return Tile.East <= tile <= Tile.Red


# 是否为幺九牌或字牌
def is_terminal_or_honor(tile: int) -> bool:
    """是否为幺九牌或字牌（用于加符/役判定等）。"""
    n = tile % 9
    return tile <= Tile.Red and (n == 0 or n == 8 or Tile.East <= tile)


# 是否为断幺九（数牌 2-8）
def is_simples(tile: int) -> bool:
    """是否为断幺九牌（序数牌 2-8）。"""
    n = tile % 9
    return tile <= Tile.Souzu9 and n != 0 and n != 8
