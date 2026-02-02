"""牌的常量与基础集合。

- `Tile`：内部使用的牌 id（0..36，其中 34..36 为赤 5）
- `YAOCHUU_TILES`：幺九牌集合（1/9 + 字牌）
"""

# 牌的常量定义
class Tile:
    # 无效牌
    Null = -1
    # 万子 1-9
    Manzu1 = 0
    Manzu2 = 1
    Manzu3 = 2
    Manzu4 = 3
    Manzu5 = 4
    Manzu6 = 5
    Manzu7 = 6
    Manzu8 = 7
    Manzu9 = 8

    # 筒子 1-9
    Pinzu1 = 9
    Pinzu2 = 10
    Pinzu3 = 11
    Pinzu4 = 12
    Pinzu5 = 13
    Pinzu6 = 14
    Pinzu7 = 15
    Pinzu8 = 16
    Pinzu9 = 17

    # 索子 1-9
    Souzu1 = 18
    Souzu2 = 19
    Souzu3 = 20
    Souzu4 = 21
    Souzu5 = 22
    Souzu6 = 23
    Souzu7 = 24
    Souzu8 = 25
    Souzu9 = 26

    # 字牌（风牌 + 三元牌）
    East = 27
    South = 28
    West = 29
    North = 30
    White = 31
    Green = 32
    Red = 33

    # 赤宝牌 5
    RedManzu5 = 34
    RedPinzu5 = 35
    RedSouzu5 = 36

    # 牌的总数量（包含赤宝牌）
    Length = 37


# 幺九牌（1/9 + 字牌）
YAOCHUU_TILES = (
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


# 牌名（中文，用于 CLI 展示；不影响任何内部 id/逻辑）
# - 34 张：0..33（不含赤宝牌）
# - 37 张：0..36（包含赤宝牌 34..36）
TileNameZh34 = (
    "一万",
    "二万",
    "三万",
    "四万",
    "五万",
    "六万",
    "七万",
    "八万",
    "九万",
    "一筒",
    "二筒",
    "三筒",
    "四筒",
    "五筒",
    "六筒",
    "七筒",
    "八筒",
    "九筒",
    "一索",
    "二索",
    "三索",
    "四索",
    "五索",
    "六索",
    "七索",
    "八索",
    "九索",
    "东",
    "南",
    "西",
    "北",
    "白",
    "发",
    "中",
)

TileNameZh37 = TileNameZh34 + (
    "赤五万",
    "赤五筒",
    "赤五索",
)


def tile_id_to_zh(tile_id: int) -> str:
    """将牌 id 映射为中文牌名。

    - 0..33：返回 `TileNameZh34`
    - 34..36：返回 `TileNameZh37` 的赤宝牌名称
    """
    if 0 <= tile_id < 34:
        return TileNameZh34[tile_id]
    if 0 <= tile_id < 37:
        return TileNameZh37[tile_id]
    raise ValueError(f"tile_id out of range: {tile_id}")
