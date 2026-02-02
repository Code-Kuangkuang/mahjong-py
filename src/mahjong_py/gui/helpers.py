"""GUI helper types and functions."""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import Tile, TileNameZh34


@dataclass
class UiState:
    mode: str = "score"  # score | shanten
    target: str = "hand"  # hand | win | dora | uradora
    seat_wind: int = Tile.East
    round_wind: int = Tile.East


def _tile34_to_token(tile34: int) -> str:
    if tile34 < 0 or tile34 >= 34:
        raise ValueError(f"tile34 out of range: {tile34}")
    if tile34 <= Tile.Manzu9:
        return f"{(tile34 - Tile.Manzu1) + 1}m"
    if tile34 <= Tile.Pinzu9:
        return f"{(tile34 - Tile.Pinzu1) + 1}p"
    if tile34 <= Tile.Souzu9:
        return f"{(tile34 - Tile.Souzu1) + 1}s"
    return f"{(tile34 - Tile.East) + 1}z"


def _tiles_zh_from_34(tiles: list[int]) -> list[str]:
    return [TileNameZh34[t] for t in tiles]


def _join_list(items: list[str]) -> str:
    return "、".join(items) if items else "（无）"
