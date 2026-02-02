from __future__ import annotations

"""计分相关常量定义。

- `WinFlag`：和了方式/特殊情况的标志位
- `Yaku`：役种标志位（普通役 + 役满）
- `YakuHan`：役种番数（门前/副露两套番数）
- `YakuName`：役种英文名（用于对外展示；中文说明见注释）
"""

from typing import Dict, Tuple

from .constants import Tile


class WinFlag:
    # 胡牌标志
    Null = 0
    Tsumo = 1 << 1                  # 自摸和了
    Riichi = 1 << 2                 # 立直成立
    Ippatsu = 1 << 3                # 一发成立
    RobbingAKong = 1 << 4           # 抢杠成立
    AfterAKong = 1 << 5             # 杠上开花成立
    UnderTheSea = 1 << 6            # 海底捞月成立
    UnderTheRiver = 1 << 7          # 河底捞鱼成立
    DoubleRiichi = 1 << 8           # 双立直成立
    NagashiMangan = 1 << 9          # 流局满贯成立
    BlessingOfHeaven = 1 << 10      # 天和成立
    BlessingOfEarth = 1 << 11       # 地和成立
    HandOfMan = 1 << 12             # 人和成立


YakuList = int


class Yaku:
    # 役种
    Null = 0
    Tsumo = 1 << 0                              # 自摸
    Riichi = 1 << 1                             # 立直
    Ippatsu = 1 << 2                            # 一发
    Tanyao = 1 << 3                             # 断幺
    Pinfu = 1 << 4                              # 平胡
    PureDoubleSequence = 1 << 5                 # 一杯口
    RobbingAKong = 1 << 6                       # 抢杠
    AfterAKong = 1 << 7                         # 杠上开花
    UnderTheSea = 1 << 8                        # 海底捞月
    UnderTheRiver = 1 << 9                      # 河底捞鱼
    Dora = 1 << 10                              # 宝牌
    UraDora = 1 << 11                           # 里宝牌
    RedDora = 1 << 12                           # 红宝牌
    WhiteDragon = 1 << 13                       # 白板
    GreenDragon = 1 << 14                       # 发财
    RedDragon = 1 << 15                         # 红中
    SelfWindEast = 1 << 16                      # 自风东
    SelfWindSouth = 1 << 17                     # 自风南
    SelfWindWest = 1 << 18                      # 自风西
    SelfWindNorth = 1 << 19                     # 自风北
    RoundWindEast = 1 << 20                     # 场风东
    RoundWindSouth = 1 << 21                    # 场风南
    RoundWindWest = 1 << 22                     # 场风西
    RoundWindNorth = 1 << 23                    # 场风北
    DoubleRiichi = 1 << 24                      # 双立直
    SevenPairs = 1 << 25                        # 七对子
    AllTriplets = 1 << 26                       # 对对和
    ThreeConcealedTriplets = 1 << 27            # 三暗刻
    TripleTriplets = 1 << 28                    # 三色同刻
    MixedTripleSequence = 1 << 29               # 三色同顺
    AllTerminalsAndHonors = 1 << 30             # 混老头
    PureStraight = 1 << 31                      # 一气通贯
    HalfOutsideHand = 1 << 32                   # 混全带幺九
    LittleThreeDragons = 1 << 33                # 小三元
    ThreeKongs = 1 << 34                        # 三杠子
    HalfFlush = 1 << 35                         # 混一色
    FullyOutsideHand = 1 << 36                  # 纯全带幺九
    TwicePureDoubleSequence = 1 << 37           # 两杯口
    NagashiMangan = 1 << 38                     # 流局满贯
    FullFlush = 1 << 39                         # 清一色
    BlessingOfHeaven = 1 << 40                  # 天和
    BlessingOfEarth = 1 << 41                   # 地和
    HandOfMan = 1 << 42                         # 人和
    AllGreen = 1 << 43                          # 绿一色
    BigThreeDragons = 1 << 44                   # 大三元
    LittleFourWinds = 1 << 45                   # 小四喜
    AllHonors = 1 << 46                         # 字一色
    ThirteenOrphans = 1 << 47                   # 国士无双
    NineGates = 1 << 48                         # 九莲宝灯
    FourConcealedTriplets = 1 << 49             # 四暗刻
    AllTerminals = 1 << 50                      # 清老头
    FourKongs = 1 << 51                         # 四杠子
    SingleWaitFourConcealedTriplets = 1 << 52    # 四暗刻单骑
    BigFourWinds = 1 << 53                      # 大四喜
    TrueNineGates = 1 << 54                     # 纯正九莲宝灯
    ThirteenWaitThirteenOrphans = 1 << 55       # 国士无双十三面


YakuName: Dict[int, str] = {
    Yaku.Null: "Null",
    Yaku.Tsumo: "Win by self-draw",
    Yaku.Riichi: "Riichi",
    Yaku.Ippatsu: "Ippatsu",
    Yaku.Tanyao: "All Simples",
    Yaku.Pinfu: "Pinfu",
    Yaku.PureDoubleSequence: "Pure Double Sequence",
    Yaku.RobbingAKong: "Robbing a Kong",
    Yaku.AfterAKong: "After a Kong",
    Yaku.UnderTheSea: "Under the Sea",
    Yaku.UnderTheRiver: "Under the River",
    Yaku.Dora: "Dora",
    Yaku.UraDora: "Ura Dora",
    Yaku.RedDora: "Red Dora",
    Yaku.WhiteDragon: "White Dragon",
    Yaku.GreenDragon: "Green Dragon",
    Yaku.RedDragon: "Red Dragon",
    Yaku.SelfWindEast: "Self Wind East",
    Yaku.SelfWindSouth: "Self Wind South",
    Yaku.SelfWindWest: "Self Wind West",
    Yaku.SelfWindNorth: "Self Wind North",
    Yaku.RoundWindEast: "Round Wind East",
    Yaku.RoundWindSouth: "Round Wind South",
    Yaku.RoundWindWest: "Round Wind West",
    Yaku.RoundWindNorth: "Round Wind North",
    Yaku.DoubleRiichi: "Double Riichi",
    Yaku.SevenPairs: "Seven Pairs",
    Yaku.AllTriplets: "All Triplets",
    Yaku.ThreeConcealedTriplets: "Three Concealed Triplets",
    Yaku.TripleTriplets: "Triple Triplets",
    Yaku.MixedTripleSequence: "Mixed Triple Sequence",
    Yaku.AllTerminalsAndHonors: "All Terminals and Honors",
    Yaku.PureStraight: "Pure Straight",
    Yaku.HalfOutsideHand: "Half Outside Hand",
    Yaku.LittleThreeDragons: "Little Three Dragons",
    Yaku.ThreeKongs: "Three Kongs",
    Yaku.HalfFlush: "Half Flush",
    Yaku.FullyOutsideHand: "Fully Outside Hand",
    Yaku.TwicePureDoubleSequence: "Twice Pure Double Sequence",
    Yaku.NagashiMangan: "Nagashi Mangan",
    Yaku.FullFlush: "Full Flush",
    Yaku.BlessingOfHeaven: "Blessing of Heaven",
    Yaku.BlessingOfEarth: "Blessing of Earth",
    Yaku.HandOfMan: "Hand of Man",
    Yaku.AllGreen: "All Green",
    Yaku.BigThreeDragons: "Big Three Dragons",
    Yaku.LittleFourWinds: "Little Four Winds",
    Yaku.AllHonors: "All Honors",
    Yaku.ThirteenOrphans: "Thirteen Orphans",
    Yaku.NineGates: "Nine Gates",
    Yaku.FourConcealedTriplets: "Four Concealed Triplets",
    Yaku.AllTerminals: "All Terminals",
    Yaku.FourKongs: "Four Kongs",
    Yaku.SingleWaitFourConcealedTriplets: "Single Wait Four Concealed Triplets",
    Yaku.BigFourWinds: "Big Four Winds",
    Yaku.TrueNineGates: "True Nine Gates",
    Yaku.ThirteenWaitThirteenOrphans: "Thirteen Wait Thirteen Orphans",
}


# 中文役名（用于展示；不影响现有英文 `YakuName`）
YakuNameZh: Dict[int, str] = {
    Yaku.Null: "无",
    Yaku.Tsumo: "门前清自摸和",
    Yaku.Riichi: "立直",
    Yaku.Ippatsu: "一发",
    Yaku.Tanyao: "断幺九",
    Yaku.Pinfu: "平和",
    Yaku.PureDoubleSequence: "一杯口",
    Yaku.RobbingAKong: "抢杠",
    Yaku.AfterAKong: "岭上开花",
    Yaku.UnderTheSea: "海底捞月",
    Yaku.UnderTheRiver: "河底捞鱼",
    Yaku.Dora: "宝牌",
    Yaku.UraDora: "里宝牌",
    Yaku.RedDora: "赤宝牌",
    Yaku.WhiteDragon: "役牌 白",
    Yaku.GreenDragon: "役牌 发",
    Yaku.RedDragon: "役牌 中",
    Yaku.SelfWindEast: "自风 东",
    Yaku.SelfWindSouth: "自风 南",
    Yaku.SelfWindWest: "自风 西",
    Yaku.SelfWindNorth: "自风 北",
    Yaku.RoundWindEast: "场风 东",
    Yaku.RoundWindSouth: "场风 南",
    Yaku.RoundWindWest: "场风 西",
    Yaku.RoundWindNorth: "场风 北",
    Yaku.DoubleRiichi: "两立直",
    Yaku.SevenPairs: "七对子",
    Yaku.AllTriplets: "对对和",
    Yaku.ThreeConcealedTriplets: "三暗刻",
    Yaku.TripleTriplets: "三色同刻",
    Yaku.MixedTripleSequence: "三色同顺",
    Yaku.AllTerminalsAndHonors: "混老头",
    Yaku.PureStraight: "一气通贯",
    Yaku.HalfOutsideHand: "混全带幺九",
    Yaku.LittleThreeDragons: "小三元",
    Yaku.ThreeKongs: "三杠子",
    Yaku.HalfFlush: "混一色",
    Yaku.FullyOutsideHand: "纯全带幺九",
    Yaku.TwicePureDoubleSequence: "二杯口",
    Yaku.NagashiMangan: "流局满贯",
    Yaku.FullFlush: "清一色",
    Yaku.BlessingOfHeaven: "天和",
    Yaku.BlessingOfEarth: "地和",
    Yaku.HandOfMan: "人和",
    Yaku.AllGreen: "绿一色",
    Yaku.BigThreeDragons: "大三元",
    Yaku.LittleFourWinds: "小四喜",
    Yaku.AllHonors: "字一色",
    Yaku.ThirteenOrphans: "国士无双",
    Yaku.NineGates: "九莲宝灯",
    Yaku.FourConcealedTriplets: "四暗刻",
    Yaku.AllTerminals: "清老头",
    Yaku.FourKongs: "四杠子",
    Yaku.SingleWaitFourConcealedTriplets: "四暗刻单骑",
    Yaku.BigFourWinds: "大四喜",
    Yaku.TrueNineGates: "纯正九莲宝灯",
    Yaku.ThirteenWaitThirteenOrphans: "国士无双十三面",
}


YakuHan: Dict[int, Tuple[int, int]] = {
    Yaku.Null: (0, 0),
    Yaku.Tsumo: (1, 0),                     # 门前清限定
    Yaku.Riichi: (1, 0),                    # 门前清限定
    Yaku.Ippatsu: (1, 0),                   # 门前清限定
    Yaku.Tanyao: (1, 1),
    Yaku.Pinfu: (1, 0),                     # 门前清限定
    Yaku.PureDoubleSequence: (1, 0),        # 门前清限定
    Yaku.RobbingAKong: (1, 1),
    Yaku.AfterAKong: (1, 1),
    Yaku.UnderTheSea: (1, 1),
    Yaku.UnderTheRiver: (1, 1),
    Yaku.Dora: (0, 0),
    Yaku.UraDora: (0, 0),
    Yaku.RedDora: (0, 0),
    Yaku.WhiteDragon: (1, 1),
    Yaku.GreenDragon: (1, 1),
    Yaku.RedDragon: (1, 1),
    Yaku.SelfWindEast: (1, 1),
    Yaku.SelfWindSouth: (1, 1),
    Yaku.SelfWindWest: (1, 1),
    Yaku.SelfWindNorth: (1, 1),
    Yaku.RoundWindEast: (1, 1),
    Yaku.RoundWindSouth: (1, 1),
    Yaku.RoundWindWest: (1, 1),
    Yaku.RoundWindNorth: (1, 1),
    Yaku.DoubleRiichi: (2, 0),              # 门前清限定
    Yaku.SevenPairs: (2, 0),                # 门前清限定
    Yaku.AllTriplets: (2, 2),
    Yaku.ThreeConcealedTriplets: (2, 2),
    Yaku.TripleTriplets: (2, 2),
    Yaku.MixedTripleSequence: (2, 1),       # 副露减番
    Yaku.AllTerminalsAndHonors: (2, 2),
    Yaku.PureStraight: (2, 1),              # 副露减番
    Yaku.HalfOutsideHand: (2, 1),           # 副露减番
    Yaku.LittleThreeDragons: (2, 2),
    Yaku.ThreeKongs: (2, 2),
    Yaku.HalfFlush: (3, 2),                 # 副露减番
    Yaku.FullyOutsideHand: (3, 2),          # 副露减番
    Yaku.TwicePureDoubleSequence: (3, 0),   # 门前清限定
    Yaku.NagashiMangan: (0, 0),             # 满贯处理
    Yaku.FullFlush: (6, 5),                 # 副露减番
    Yaku.BlessingOfHeaven: (1, 0),          # 役满
    Yaku.BlessingOfEarth: (1, 0),           # 役满
    Yaku.HandOfMan: (1, 0),                 # 役满
    Yaku.AllGreen: (1, 0),                  # 役满
    Yaku.BigThreeDragons: (1, 0),           # 役满
    Yaku.LittleFourWinds: (1, 0),           # 役满
    Yaku.AllHonors: (1, 0),                 # 役满
    Yaku.ThirteenOrphans: (1, 0),           # 役满
    Yaku.NineGates: (1, 0),                 # 役满
    Yaku.FourConcealedTriplets: (1, 0),     # 役满
    Yaku.AllTerminals: (1, 0),              # 役满
    Yaku.FourKongs: (1, 0),                 # 役满
    Yaku.SingleWaitFourConcealedTriplets: (2, 0),   # 两倍役满
    Yaku.BigFourWinds: (2, 0),                      # 两倍役满
    Yaku.TrueNineGates: (2, 0),                     # 两倍役满
    Yaku.ThirteenWaitThirteenOrphans: (2, 0),       # 两倍役满
}

# 普通役种
NormalYaku = (
    Yaku.Tsumo
    | Yaku.Riichi
    | Yaku.Ippatsu
    | Yaku.Tanyao
    | Yaku.Pinfu
    | Yaku.PureDoubleSequence
    | Yaku.RobbingAKong
    | Yaku.AfterAKong
    | Yaku.UnderTheSea
    | Yaku.UnderTheRiver
    | Yaku.Dora
    | Yaku.UraDora
    | Yaku.RedDora
    | Yaku.WhiteDragon
    | Yaku.GreenDragon
    | Yaku.RedDragon
    | Yaku.SelfWindEast
    | Yaku.SelfWindSouth
    | Yaku.SelfWindWest
    | Yaku.SelfWindNorth
    | Yaku.RoundWindEast
    | Yaku.RoundWindSouth
    | Yaku.RoundWindWest
    | Yaku.RoundWindNorth
    | Yaku.DoubleRiichi
    | Yaku.SevenPairs
    | Yaku.AllTriplets
    | Yaku.ThreeConcealedTriplets
    | Yaku.TripleTriplets
    | Yaku.MixedTripleSequence
    | Yaku.AllTerminalsAndHonors
    | Yaku.PureStraight
    | Yaku.HalfOutsideHand
    | Yaku.LittleThreeDragons
    | Yaku.ThreeKongs
    | Yaku.HalfFlush
    | Yaku.FullyOutsideHand
    | Yaku.TwicePureDoubleSequence
    | Yaku.FullFlush
)


# 役满表
Yakuman = (
    Yaku.BlessingOfHeaven
    | Yaku.BlessingOfEarth
    | Yaku.HandOfMan
    | Yaku.AllGreen
    | Yaku.BigThreeDragons
    | Yaku.LittleFourWinds
    | Yaku.AllHonors
    | Yaku.ThirteenOrphans
    | Yaku.NineGates
    | Yaku.FourConcealedTriplets
    | Yaku.AllTerminals
    | Yaku.FourKongs
    | Yaku.SingleWaitFourConcealedTriplets
    | Yaku.BigFourWinds
    | Yaku.TrueNineGates
    | Yaku.ThirteenWaitThirteenOrphans
)


class ScoreTitle:
    Null = -1
    Mangan = 0              # 满贯
    Haneman = 1             # 跳满
    Baiman = 2              # 倍满
    Sanbaiman = 3           # 三倍满
    CountedYakuman = 4      # 累计役满
    Yakuman = 5             # 役满
    DoubleYakuman = 6       # 双倍役满
    TripleYakuman = 7       # 三倍役满
    QuadrupleYakuman = 8    # 四倍役满
    QuintupleYakuman = 9    # 五倍役满
    SextupleYakuman = 10    # 六倍役满

# dora -> dora 指示牌
ToIndicator = (
    Tile.Manzu9,
    Tile.Manzu1,
    Tile.Manzu2,
    Tile.Manzu3,
    Tile.Manzu4,
    Tile.Manzu5,
    Tile.Manzu6,
    Tile.Manzu7,
    Tile.Manzu8,
    Tile.Pinzu9,
    Tile.Pinzu1,
    Tile.Pinzu2,
    Tile.Pinzu3,
    Tile.Pinzu4,
    Tile.Pinzu5,
    Tile.Pinzu6,
    Tile.Pinzu7,
    Tile.Pinzu8,
    Tile.Souzu9,
    Tile.Souzu1,
    Tile.Souzu2,
    Tile.Souzu3,
    Tile.Souzu4,
    Tile.Souzu5,
    Tile.Souzu6,
    Tile.Souzu7,
    Tile.Souzu8,
    Tile.North,
    Tile.East,
    Tile.South,
    Tile.West,
    Tile.Red,
    Tile.White,
    Tile.Green,
    Tile.Manzu4,
    Tile.Pinzu4,
    Tile.Souzu4,
)


# dora指示牌 -> dora
ToDora = (
    Tile.Manzu2,
    Tile.Manzu3,
    Tile.Manzu4,
    Tile.Manzu5,
    Tile.Manzu6,
    Tile.Manzu7,
    Tile.Manzu8,
    Tile.Manzu9,
    Tile.Manzu1,
    Tile.Pinzu2,
    Tile.Pinzu3,
    Tile.Pinzu4,
    Tile.Pinzu5,
    Tile.Pinzu6,
    Tile.Pinzu7,
    Tile.Pinzu8,
    Tile.Pinzu9,
    Tile.Pinzu1,
    Tile.Souzu2,
    Tile.Souzu3,
    Tile.Souzu4,
    Tile.Souzu5,
    Tile.Souzu6,
    Tile.Souzu7,
    Tile.Souzu8,
    Tile.Souzu9,
    Tile.Souzu1,
    Tile.South,
    Tile.West,
    Tile.North,
    Tile.East,
    Tile.Green,
    Tile.Red,
    Tile.White,
    Tile.Manzu6,
    Tile.Pinzu6,
    Tile.Souzu6,
)
