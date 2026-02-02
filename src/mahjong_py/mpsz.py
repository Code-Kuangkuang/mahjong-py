"""mpsz 字符串与手牌计数之间的转换。

mpsz 是常见的麻将手牌表示：
- 数牌用 m/p/s（万/筒/索），字牌用 z
- 同一花色的数字写在一起，例如："123m789p123s77z"
- 赤宝牌用 0m/0p/0s 表示
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .constants import Tile


def _ensure_array(hand: Iterable[int]) -> np.ndarray:
    """将输入转换为 1 维 numpy 数组，并校验长度为 34 或 37。"""
    arr = np.asarray(list(hand), dtype=np.int16)
    if arr.ndim != 1:
        raise ValueError("hand must be 1D")
    if arr.size not in (34, 37):
        raise ValueError("hand length must be 34 or 37")
    return arr


def to_mpsz(hand: Iterable[int]) -> str:
    """将手牌计数转换为 mpsz 字符串。

    支持：
    - 长度 34：不包含赤宝牌 flag（仅 0..33）
    - 长度 37：包含赤宝牌 flag（34..36），且 5 的计数包含赤 5
    """
    h = _ensure_array(hand)
    has_red = h.size == 37

    def suit_digits(start: int, red_flag_index: int | None) -> str:
        digits = []
        red = 0
        if has_red and red_flag_index is not None and h[red_flag_index] > 0:
            red = 1
            digits.append("0")

        for i in range(9):
            count = int(h[start + i])
            if i == 4 and red:
                count -= 1
            if count > 0:
                digits.append(str(i + 1) * count)
        return "".join(digits)

    s = []

    man = suit_digits(0, Tile.RedManzu5 if has_red else None)
    if man:
        s.append(man + "m")

    pin = suit_digits(9, Tile.RedPinzu5 if has_red else None)
    if pin:
        s.append(pin + "p")

    sou = suit_digits(18, Tile.RedSouzu5 if has_red else None)
    if sou:
        s.append(sou + "s")

    # honors: 1..7z
    honors = []
    for i in range(7):
        count = int(h[27 + i])
        if count > 0:
            honors.append(str(i + 1) * count)
    if honors:
        s.append("".join(honors) + "z")

    return "".join(s)


def from_mpsz(tiles: str) -> np.ndarray:
    """解析 mpsz 字符串为长度 37 的手牌计数（含赤宝牌 flag）。

    - 赤 5 用 0m/0p/0s 表示
    - 解析后会同时增加赤宝牌 flag（34..36）与对应 5 的计数
    """
    hand = np.zeros(37, dtype=np.int16)

    suit = None
    for ch in reversed(tiles):
        if ch.isspace():
            continue
        c = ch.lower()
        if c in ("m", "p", "s", "z"):
            suit = c
            continue
        if not c.isdigit():
            raise ValueError(f"invalid character: {ch}")
        if suit is None:
            raise ValueError("digit before suit")

        n = int(c) - 1
        if suit == "m":
            if n == -1:
                hand[Tile.RedManzu5] += 1
                hand[Tile.Manzu5] += 1
            else:
                hand[Tile.Manzu1 + n] += 1
        elif suit == "p":
            if n == -1:
                hand[Tile.RedPinzu5] += 1
                hand[Tile.Pinzu5] += 1
            else:
                hand[Tile.Pinzu1 + n] += 1
        elif suit == "s":
            if n == -1:
                hand[Tile.RedSouzu5] += 1
                hand[Tile.Souzu5] += 1
            else:
                hand[Tile.Souzu1 + n] += 1
        elif suit == "z":
            if n < 0 or n > 6:
                raise ValueError("honors must be 1-7")
            hand[Tile.East + n] += 1

    return hand


def to_hand34(hand: Iterable[int]) -> np.ndarray:
    """将长度 37 的手牌计数转换为长度 34（赤宝牌合并进 5 的计数）。"""
    h = _ensure_array(hand)
    if h.size == 34:
        return h.astype(np.int16, copy=True)

    out = h[:34].astype(np.int16, copy=True)
    # ensure red flags don't affect counts beyond 5s (counts already include reds)
    return out
