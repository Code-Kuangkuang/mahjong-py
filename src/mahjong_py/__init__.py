"""mahjong-py 对外 API 汇总。

该包提供：
- mpsz（手牌字符串）与 34/37 计数形式的转换
- 向听数计算
- 进张/无效牌（必要牌/不必要牌）计算
- 点数计算（役/符/番/满贯以上与最终点数）
"""

from .mpsz import from_mpsz, to_hand34, to_mpsz
from .shanten import ShantenFlag, shanten
from .necessary import necessary_tiles
from .unnecessary import unnecessary_tiles
from .expected_score_calculator import (
    ExpectedScoreCalculator,
    ExpectedScoreConfig,
    ExpectedScoreStat,
)
from .score_calculator import ScoreCalculator
from .score_types import Block, Meld, Player, Round, Result, hand_from_tiles
from .score_constants import WinFlag, Yaku, ScoreTitle, YakuName, YakuNameZh

__all__ = [
    "from_mpsz",
    "to_hand34",
    "to_mpsz",
    "ShantenFlag",
    "shanten",
    "necessary_tiles",
    "unnecessary_tiles",
    "ExpectedScoreCalculator",
    "ExpectedScoreConfig",
    "ExpectedScoreStat",
    "ScoreCalculator",
    "Block",
    "Meld",
    "Player",
    "Round",
    "Result",
    "hand_from_tiles",
    "WinFlag",
    "Yaku",
    "ScoreTitle",
    "YakuName",
    "YakuNameZh",
]
