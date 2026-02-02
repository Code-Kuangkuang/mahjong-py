"""手牌计数的哈希（用于稠密表查表）。

实现与 C++ 侧一致：
- 数牌（9 张）按 5 进制编码
- 字牌（7 张）按 5 进制编码
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def suits_hash(hand34: np.ndarray, start: int) -> int:
    """计算数牌 9 张的 5 进制哈希（start 为花色起点：0/9/18）。"""
    h = 0
    for i in range(9):
        h = 5 * h + int(hand34[start + i])
    return h


@njit(cache=True)
def honors_hash(hand34: np.ndarray) -> int:
    """计算字牌 7 张的 5 进制哈希。"""
    h = 0
    base = 27
    for i in range(7):
        h = 5 * h + int(hand34[base + i])
    return h
