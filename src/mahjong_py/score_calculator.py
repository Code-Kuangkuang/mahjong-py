"""点数计算（役判定 + 符/番 + 点数汇总）。

入口：`ScoreCalculator.calc` / `ScoreCalculator.calc_fast`
"""

from __future__ import annotations

import math
from typing import List, Tuple

from .constants import Tile
from .hand_separator import separate
from .score_constants import (
    NormalYaku,
    ScoreTitle,
    ToDora,
    WinFlag,
    Yakuman,
    Yaku,
    YakuHan,
)
from .score_table import (
    AboveMangan,
    BelowMangan,
    IsMangan,
    RonDiscarderToDealer,
    RonDiscarderToPlayer,
    TsumoDealerToPlayer,
    TsumoPlayerToDealer,
    TsumoPlayerToPlayer,
    fu_to_index,
)
from .score_types import Block, BlockType, MeldType, Player, Result, Round, WaitType, RuleFlag
from .shanten import ShantenFlag, shanten
from .utils import check_exclusive, is_reddora, is_terminal_or_honor, to_no_reddora


class ScoreCalculator:
    """日麻计分核心逻辑。

    该类主要负责：
    - 参数校验（自摸/立直等标记的互斥关系）
    - 非形状型役（天和、地和、国士/九莲等）的判定
    - 形状型役（平和、三暗刻等）的判定与最优形选择
    - 符/番统计与点数（含本场/供托）的汇总

    说明：本实现用 bit flag 表示役种，并在必要处用位运算与预计算表加速。
    """

    @staticmethod
    def calc(round_: Round, player: Player, win_tile: int, win_flag: int) -> Result:
        """计分入口。

        - 先做参数与手牌一致性检查
        - 再检查是否为和了形（向听数必须为 -1）
        - 最后走快速计分路径
        """
        ok, err = ScoreCalculator.check_arguments(player, win_tile, win_flag)
        if not ok:
            return Result(False, player, win_tile, win_flag, err_msg=err)

        shanten_type, sht = shanten(player.hand[:34], player.num_melds(), ShantenFlag.All)
        if sht != -1:
            return Result(False, player, win_tile, win_flag, err_msg="The hand is not winning form.")

        return ScoreCalculator.calc_fast(round_, player, win_tile, win_flag, shanten_type)

    @staticmethod
    def calc_fast(round_: Round, player: Player, win_tile: int, win_flag: int, shanten_type: int) -> Result:
        """在已知该手为和了形时进行计分。

        先判定无需分解牌型即可确定的役（以及役满类），
        再进行分解（separate）寻找最优形来计算形状型役与符。
        """
        yaku_list = ScoreCalculator.check_not_pattern_yaku(round_, player, win_tile, win_flag, shanten_type)

        if yaku_list & Yaku.NagashiMangan:
            return ScoreCalculator.aggregate_yakuman(round_, player, win_tile, win_flag, Yaku.NagashiMangan)
        elif yaku_list & Yakuman:
            return ScoreCalculator.aggregate_yakuman(round_, player, win_tile, win_flag, yaku_list & Yakuman)

        pattern_yaku, fu, blocks, wait_type = ScoreCalculator.check_pattern_yaku(
            round_, player, win_tile, win_flag, shanten_type
        )
        yaku_list |= pattern_yaku

        if yaku_list == 0:
            return Result(False, player, win_tile, win_flag, err_msg="No yaku is established.")

        return ScoreCalculator.aggregate_normal(round_, player, win_tile, win_flag, yaku_list, fu, blocks, wait_type)

    @staticmethod
    def check_arguments(player: Player, win_tile: int, win_flag: int) -> Tuple[bool, str]:
        """参数合法性检查。

        - 胡牌牌必须出现在手牌/副露合并前的手牌计数中
        - 若干 win_flag 之间必须互斥（例如立直/两立直）
        - 立直相关要求门前清
        - 海底/岭上等要求自摸
        """
        if player.hand[to_no_reddora(win_tile)] == 0:
            return False, f"Win tile {win_tile} is not in the hand."

        if not check_exclusive(win_flag & (WinFlag.Riichi | WinFlag.DoubleRiichi)):
            return False, "Only one of Riichi or Double Riichi may be specified."

        if not check_exclusive(
            win_flag
            & (
                WinFlag.RobbingAKong
                | WinFlag.AfterAKong
                | WinFlag.UnderTheSea
                | WinFlag.UnderTheRiver
            )
        ):
            return False, "Only one of RobbingAKong, AfterAKong, UnderTheSea, UnderTheRiver may be specified."

        if not check_exclusive(
            win_flag & (WinFlag.BlessingOfHeaven | WinFlag.BlessingOfEarth | WinFlag.HandOfMan)
        ):
            return False, "Only one of BlessingOfHeaven, BlessingOfEarth, HandOfMan may be specified."

        if (win_flag & (WinFlag.Riichi | WinFlag.DoubleRiichi)) and not player.is_closed():
            return False, "Riichi/DoubleRiichi requires closed hand."

        if (win_flag & WinFlag.Ippatsu) and not (win_flag & (WinFlag.Riichi | WinFlag.DoubleRiichi)):
            return False, "Ippatsu requires Riichi/DoubleRiichi."

        if (win_flag & (WinFlag.UnderTheSea | WinFlag.AfterAKong)) and not (win_flag & WinFlag.Tsumo):
            return False, "UnderTheSea/AfterAKong requires Tsumo."

        return True, ""

    @staticmethod
    def aggregate_yakuman(round_: Round, player: Player, win_tile: int, win_flag: int, yaku_list: int) -> Result:
        """役满/流局满贯的汇总与点数计算。"""
        yaku_han_list: List[Tuple[int, int]] = []
        is_dealer = player.wind == Tile.East

        if yaku_list & Yaku.NagashiMangan:
            yaku_han_list.append((Yaku.NagashiMangan, 0))
            score_title = ScoreTitle.Mangan
            score = ScoreCalculator.calc_score(is_dealer, True, round_.honba, round_.kyotaku, score_title)
            return Result(True, player, win_tile, win_flag, yaku_han_list, 0, 0, score_title, score)

        # 统计役满倍数：不同役满可叠加（双倍/三倍役满等）
        n = 0
        y = 1 << 40
        while y <= (1 << 55):
            if (yaku_list & y) and (y & Yakuman):
                n += YakuHan.get(y, (0, 0))[0]
                yaku_han_list.append((y, YakuHan.get(y, (0, 0))[0]))
            y <<= 1

        score_title = ScoreCalculator.get_score_title_yakuman(n)
        score = ScoreCalculator.calc_score(is_dealer, bool(win_flag & WinFlag.Tsumo), round_.honba, round_.kyotaku, score_title)
        return Result(True, player, win_tile, win_flag, yaku_han_list, 0, 0, score_title, score)

    @staticmethod
    def aggregate_normal(
        round_: Round,
        player: Player,
        win_tile: int,
        win_flag: int,
        yaku_list: int,
        fu: int,
        blocks: List[Block],
        wait_type: int,
    ) -> Result:
        """普通役（非役满）汇总：统计番/符，计算点数并返回详细分解信息。"""
        han = 0
        yaku_han_list: List[Tuple[int, int]] = []
        y = 1
        while y <= (1 << 39):
            if (yaku_list & y) and (NormalYaku & y):
                yaku_han = YakuHan[y][0] if player.is_closed() else YakuHan[y][1]
                yaku_han_list.append((y, yaku_han))
                han += yaku_han
            y <<= 1

        num_doras = ScoreCalculator.count_dora(player.hand, player.melds, round_.dora_indicators)
        if num_doras:
            yaku_han_list.append((Yaku.Dora, num_doras))
            han += num_doras

        num_uradoras = ScoreCalculator.count_dora(player.hand, player.melds, round_.uradora_indicators)
        if num_uradoras:
            yaku_han_list.append((Yaku.UraDora, num_uradoras))
            han += num_uradoras

        num_reddoras = ScoreCalculator.count_reddora(bool(round_.rules & RuleFlag.RedDora), player.hand, player.melds)
        if num_reddoras:
            yaku_han_list.append((Yaku.RedDora, num_reddoras))
            han += num_reddoras

        score_title = ScoreCalculator.get_score_title(fu, han)
        is_dealer = player.wind == Tile.East
        score = ScoreCalculator.calc_score(is_dealer, bool(win_flag & WinFlag.Tsumo), round_.honba, round_.kyotaku, score_title, han, fu)

        yaku_han_list.sort(key=lambda x: x[0])

        return Result(True, player, win_tile, win_flag, yaku_han_list, han, fu, score_title, score, blocks, wait_type)

    @staticmethod
    def calc_fu(blocks: List[Block], wait_type: int, is_closed: bool, is_tsumo: bool, is_pinfu: bool, round_wind: int, seat_wind: int) -> int:
        """计算符。

        关键点：
        - 平和自摸门前：固定 20 符
        - 平和荣和（并且不是门前）：固定 30 符（该规则与实现保持一致）
        - 门前荣和 +10 符，自摸 +2 符
        - 边张/嵌张/单骑等加符
        - 刻子/杠子符按明暗与幺九牌翻倍
        - 役牌/自风/场风雀头加符
        """
        if is_pinfu and is_tsumo and is_closed:
            return 20
        if is_pinfu and (not is_tsumo) and (not is_closed):
            return 30

        fu = 20
        if is_closed and not is_tsumo:
            fu += 10
        elif is_tsumo:
            fu += 2

        if wait_type in (WaitType.ClosedWait, WaitType.EdgeWait, WaitType.PairWait):
            fu += 2

        for block in blocks:
            if block.type & (BlockType.Triplet | BlockType.Kong):
                if block.type == (BlockType.Triplet | BlockType.Open):
                    block_fu = 2
                elif block.type == BlockType.Triplet:
                    block_fu = 4
                elif block.type == (BlockType.Kong | BlockType.Open):
                    block_fu = 8
                else:
                    block_fu = 16

                fu += block_fu * 2 if is_terminal_or_honor(block.min_tile) else block_fu
            elif block.type & BlockType.Pair:
                if block.min_tile == seat_wind and block.min_tile == round_wind:
                    fu += 4
                elif block.min_tile == seat_wind or block.min_tile == round_wind or block.min_tile >= Tile.White:
                    fu += 2

        return fu

    @staticmethod
    def check_not_pattern_yaku(round_: Round, player: Player, win_tile: int, win_flag: int, shanten_type: int) -> int:
        """判定不依赖具体分解形（blocks）的役。

        这些役可以在“合并手牌计数”层面确定：
        - 立直/一发/海底/河底等和了方式类
        - 国士无双、九莲宝灯、清一色等牌种结构类
        - 多数役满
        """
        yaku_list = 0

        if (win_flag & WinFlag.Tsumo) and player.is_closed():
            yaku_list |= Yaku.Tsumo
        if win_flag & WinFlag.Riichi:
            yaku_list |= Yaku.Riichi
        if win_flag & WinFlag.Ippatsu:
            yaku_list |= Yaku.Ippatsu
        if win_flag & WinFlag.RobbingAKong:
            yaku_list |= Yaku.RobbingAKong
        if win_flag & WinFlag.AfterAKong:
            yaku_list |= Yaku.AfterAKong
        if win_flag & WinFlag.UnderTheSea:
            yaku_list |= Yaku.UnderTheSea
        if win_flag & WinFlag.UnderTheRiver:
            yaku_list |= Yaku.UnderTheRiver
        if win_flag & WinFlag.DoubleRiichi:
            yaku_list |= Yaku.DoubleRiichi
        if win_flag & WinFlag.NagashiMangan:
            yaku_list |= Yaku.NagashiMangan
        if win_flag & WinFlag.BlessingOfHeaven:
            yaku_list |= Yaku.BlessingOfHeaven
        if win_flag & WinFlag.BlessingOfEarth:
            yaku_list |= Yaku.BlessingOfEarth
        if win_flag & WinFlag.HandOfMan:
            yaku_list |= Yaku.HandOfMan

        merged = ScoreCalculator.merge_hand(player)
        nored_win_tile = to_no_reddora(win_tile)

        if shanten_type & ShantenFlag.Regular:
            yaku_list |= ScoreCalculator.check_all_green(merged)
            yaku_list |= ScoreCalculator.check_three_dragons(merged)
            yaku_list |= ScoreCalculator.check_four_winds(merged)
            yaku_list |= ScoreCalculator.check_all_honors(merged)
            yaku_list |= ScoreCalculator.check_four_concealed_triplets(player, merged, nored_win_tile, win_flag)
            yaku_list |= ScoreCalculator.check_all_terminals(merged)
            yaku_list |= ScoreCalculator.check_kongs(player)
            yaku_list |= ScoreCalculator.check_nine_gates(player, merged, nored_win_tile)
            yaku_list |= ScoreCalculator.check_tanyao(player, merged, bool(round_.rules & RuleFlag.OpenTanyao))
            yaku_list |= ScoreCalculator.check_flush(merged)
            yaku_list |= ScoreCalculator.check_value_tile(round_, player, merged)
        elif shanten_type & ShantenFlag.SevenPairs:
            yaku_list |= Yaku.SevenPairs
            yaku_list |= ScoreCalculator.check_all_honors(merged)
            yaku_list |= ScoreCalculator.check_all_terminals(merged)
            yaku_list |= ScoreCalculator.check_tanyao(player, merged, bool(round_.rules & RuleFlag.OpenTanyao))
            yaku_list |= ScoreCalculator.check_flush(merged)
        else:
            if ScoreCalculator.check_thirteen_wait_thirteen_orphans(merged, nored_win_tile):
                yaku_list |= Yaku.ThirteenWaitThirteenOrphans
            else:
                yaku_list |= Yaku.ThirteenOrphans

        return yaku_list

    @staticmethod
    def check_pattern_yaku(round_: Round, player: Player, win_tile: int, win_flag: int, shanten_type: int) -> Tuple[int, int, List[Block], int]:
        """判定依赖牌型分解（blocks）的役，并选择“番数优先，其次符数”的最优形。

        返回： (役列表, 符, 最优 blocks, wait_type)
        """
        if shanten_type == ShantenFlag.SevenPairs:
            # 七对：符固定 25，等待形视为单骑
            return 0, 25, [], WaitType.PairWait

        pattern_yaku = [
            Yaku.Pinfu,
            Yaku.PureDoubleSequence,
            Yaku.AllTriplets,
            Yaku.ThreeConcealedTriplets,
            Yaku.TripleTriplets,
            Yaku.MixedTripleSequence,
            Yaku.PureStraight,
            Yaku.HalfOutsideHand,
            Yaku.ThreeKongs,
            Yaku.FullyOutsideHand,
            Yaku.TwicePureDoubleSequence,
        ]

        # separate 会枚举可能的面子/雀头组合与等待形
        pattern = separate(player, win_tile, win_flag)
        if not pattern:
            return 0, 0, [], WaitType.Null

        max_han = 0
        max_fu = 0
        max_idx = 0
        max_yaku_list = 0

        for idx, (blocks, wait_type) in enumerate(pattern):
            yaku_list = 0
            is_pinfu = ScoreCalculator.check_pinfu(blocks, wait_type, round_.wind, player.wind)

            if player.is_closed():
                if is_pinfu:
                    yaku_list |= Yaku.Pinfu
                yaku_list |= ScoreCalculator.check_pure_double_sequence(blocks)

            if ScoreCalculator.check_pure_straight(blocks):
                yaku_list |= Yaku.PureStraight
            elif ScoreCalculator.check_triple_triplets(blocks):
                yaku_list |= Yaku.TripleTriplets
            elif ScoreCalculator.check_mixed_triple_sequence(blocks):
                yaku_list |= Yaku.MixedTripleSequence

            yaku_list |= ScoreCalculator.check_outside_hand(blocks)
            yaku_list |= ScoreCalculator.check_all_triplets(blocks)
            yaku_list |= ScoreCalculator.check_three_concealed_triplets(blocks)

            han = 0
            for y in pattern_yaku:
                if yaku_list & y:
                    han += YakuHan[y][0] if player.is_closed() else YakuHan[y][1]

            fu = ScoreCalculator.calc_fu(
                blocks,
                wait_type,
                player.is_closed(),
                bool(win_flag & WinFlag.Tsumo),
                is_pinfu,
                round_.wind,
                player.wind,
            )

            if (han > max_han) or (han == max_han and fu > max_fu):
                max_han = han
                max_fu = fu
                max_idx = idx
                max_yaku_list = yaku_list

        # 符按规则向上取整到 10 的倍数
        max_fu = int(math.ceil(max_fu / 10.0) * 10)

        return max_yaku_list, max_fu, pattern[max_idx][0], pattern[max_idx][1]

    @staticmethod
    def check_pinfu(blocks: List[Block], wait_type: int, round_wind: int, seat_wind: int) -> bool:
        """平和判定。

        条件（按本实现）：
        - 两面听（DoubleEdgeWait）
        - 全部为顺子 + 雀头，且无刻子/杠子
        - 雀头不能是役牌（场风/自风/三元牌）
        """
        if wait_type != WaitType.DoubleEdgeWait:
            return False
        for b in blocks:
            if b.type & (BlockType.Triplet | BlockType.Kong):
                return False
            if (b.type & BlockType.Pair) and (b.min_tile == round_wind or b.min_tile == seat_wind or b.min_tile >= Tile.White):
                return False
        return True

    @staticmethod
    def check_pure_double_sequence(blocks: List[Block]) -> int:
        """一杯口/二杯口判定（门前限定的番种由外层控制）。"""
        count = [0] * 34
        for b in blocks:
            if b.type & BlockType.Sequence:
                count[b.min_tile] += 1
        num_double = 0
        for x in count:
            if x == 4:
                num_double += 2
            elif x >= 2:
                num_double += 1
        if num_double == 1:
            return Yaku.PureDoubleSequence
        if num_double == 2:
            return Yaku.TwicePureDoubleSequence
        return 0

    @staticmethod
    def check_all_triplets(blocks: List[Block]) -> int:
        """对对和判定（全部面子为刻子/杠子）。"""
        for b in blocks:
            if b.type & BlockType.Sequence:
                return 0
        return Yaku.AllTriplets

    @staticmethod
    def check_three_concealed_triplets(blocks: List[Block]) -> int:
        """三暗刻判定。

        注意：这里“暗刻/暗杠”的区分由上游分解与 BlockType 语义决定。
        本函数按块类型统计 Triplet/Kong 的数量。
        """
        num_triplets = 0
        for b in blocks:
            if b.type == BlockType.Triplet or b.type == BlockType.Kong:
                num_triplets += 1
        return Yaku.ThreeConcealedTriplets if num_triplets == 3 else 0

    @staticmethod
    def check_triple_triplets(blocks: List[Block]) -> bool:
        """三色同刻判定（同数牌在三门各有刻子/杠子）。"""
        count = [0] * 34
        for b in blocks:
            if b.type & (BlockType.Triplet | BlockType.Kong):
                count[b.min_tile] += 1
        for i in range(9):
            if count[i] and count[i + 9] and count[i + 18]:
                return True
        return False

    @staticmethod
    def check_mixed_triple_sequence(blocks: List[Block]) -> bool:
        """三色同顺判定（同序列在三门各有顺子）。"""
        count = [0] * 34
        for b in blocks:
            if b.type & BlockType.Sequence:
                count[b.min_tile] += 1
        for i in range(9):
            if count[i] and count[i + 9] and count[i + 18]:
                return True
        return False

    @staticmethod
    def check_pure_straight(blocks: List[Block]) -> bool:
        """一气通贯判定（同一门 123 + 456 + 789）。"""
        count = [0] * 34
        for b in blocks:
            if b.type & BlockType.Sequence:
                count[b.min_tile] += 1
        return (
            (count[Tile.Manzu1] and count[Tile.Manzu4] and count[Tile.Manzu7])
            or (count[Tile.Pinzu1] and count[Tile.Pinzu4] and count[Tile.Pinzu7])
            or (count[Tile.Souzu1] and count[Tile.Souzu4] and count[Tile.Souzu7])
        )

    @staticmethod
    def check_outside_hand(blocks: List[Block]) -> int:
        """混全带幺九/纯全带幺九判定。

        - 若含字牌并满足每组都带幺九，则为混全带幺九
        - 不含字牌且每组都带幺九，则为纯全带幺九
        """
        honor_block = False
        sequence_block = False
        for b in blocks:
            if b.type & BlockType.Sequence:
                if b.min_tile not in (
                    Tile.Manzu1,
                    Tile.Manzu7,
                    Tile.Pinzu1,
                    Tile.Pinzu7,
                    Tile.Souzu1,
                    Tile.Souzu7,
                ):
                    return 0
                sequence_block = True
            else:
                if b.min_tile not in (
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
                ):
                    return 0
                if b.min_tile >= Tile.East:
                    honor_block = True

        if honor_block and sequence_block:
            return Yaku.HalfOutsideHand
        if (not honor_block) and sequence_block:
            return Yaku.FullyOutsideHand
        return 0

    @staticmethod
    def merge_hand(player: Player) -> Tuple[list, int, int, int, int]:
        """将手牌与副露合并为计数数组，并生成用于快速判断的 8 进制压缩表示。

        返回：
        - hand: 长度 34 的计数
        - manzu/pinzu/souzu/honors: 以 8 进制“每张牌 3bit”压缩后的整数
        """
        hand = [int(x) for x in player.hand]
        for meld in player.melds:
            min_tile = to_no_reddora(meld.tiles[0])
            if meld.type == MeldType.Chow:
                hand[min_tile] += 1
                hand[min_tile + 1] += 1
                hand[min_tile + 2] += 1
            else:
                hand[min_tile] += 3

        manzu = 0
        for i in range(9):
            manzu = manzu * 8 + hand[i]
        pinzu = 0
        for i in range(9, 18):
            pinzu = pinzu * 8 + hand[i]
        souzu = 0
        for i in range(18, 27):
            souzu = souzu * 8 + hand[i]
        honors = 0
        for i in range(27, 34):
            honors = honors * 8 + hand[i]

        return hand, manzu, pinzu, souzu, honors

    @staticmethod
    def check_all_green(merged) -> int:
        _, manzu, pinzu, souzu, honors = merged
        # 绿一色允许的索子：2/3/4/6/8，以及发；其余位置为掩码过滤
        souzu_mask = 0b111_000_000_000_111_000_111_000_111
        honors_mask = 0b111_111_111_111_111_000_111
        return 0 if (manzu or pinzu or (souzu & souzu_mask) or (honors & honors_mask)) else Yaku.AllGreen

    @staticmethod
    def check_three_dragons(merged) -> int:
        """小三元/大三元判定。"""
        hand, *_ = merged
        total = hand[Tile.White] + hand[Tile.Green] + hand[Tile.Red]
        if total == 8:
            return Yaku.LittleThreeDragons
        if total == 9:
            return Yaku.BigThreeDragons
        return 0

    @staticmethod
    def check_four_winds(merged) -> int:
        """小四喜/大四喜判定。"""
        hand, *_ = merged
        total = hand[Tile.East] + hand[Tile.South] + hand[Tile.West] + hand[Tile.North]
        if total == 11:
            return Yaku.LittleFourWinds
        if total == 12:
            return Yaku.BigFourWinds
        return 0

    @staticmethod
    def check_all_honors(merged) -> int:
        """字一色判定（仅字牌）。"""
        _, manzu, pinzu, souzu, _ = merged
        return 0 if (manzu or pinzu or souzu) else Yaku.AllHonors

    @staticmethod
    def check_four_concealed_triplets(player: Player, merged, win_tile: int, win_flag: int) -> int:
        """四暗刻/四暗刻单骑判定。

        本实现要求：
        - 自摸
        - 门前清
        """
        if not (win_flag & WinFlag.Tsumo):
            return 0
        if not player.is_closed():
            return 0

        hand, *_ = merged
        num_triplets = 0
        num_pairs = 0
        single_wait = False
        for i in range(34):
            if hand[i] == 3:
                num_triplets += 1
            elif hand[i] == 2:
                num_pairs += 1
                single_wait = (i == win_tile)

        if num_triplets == 4 and num_pairs == 1:
            return Yaku.SingleWaitFourConcealedTriplets if single_wait else Yaku.FourConcealedTriplets
        return 0

    @staticmethod
    def check_all_terminals(merged) -> int:
        _, manzu, pinzu, souzu, honors = merged
        # terminals_mask 表示序数牌 2~8（非幺九）的位段；若出现则不是清幺九系
        terminals_mask = 0b000_111_111_111_111_111_111_111_000
        if (manzu | pinzu | souzu) & terminals_mask:
            return 0
        return Yaku.AllTerminalsAndHonors if honors else Yaku.AllTerminals

    @staticmethod
    def check_kongs(player: Player) -> int:
        """三杠子/四杠子判定（基于副露中的杠子数量）。"""
        num_kongs = 0
        for meld in player.melds:
            if meld.type >= MeldType.ClosedKong:
                num_kongs += 1
        if num_kongs == 4:
            return Yaku.FourKongs
        if num_kongs == 3:
            return Yaku.ThreeKongs
        return 0

    @staticmethod
    def check_nine_gates(player: Player, merged, win_tile: int) -> int:
        """九莲宝灯/纯正九莲判定（门前清一色 + 特定牌姿）。"""
        if player.melds:
            return 0

        hand, manzu, pinzu, souzu, _ = merged
        # tile1: 压缩表示中“减去一张对应牌”的值（按 8 进制每位 3bit）
        tile1 = [
            1 << 24, 1 << 21, 1 << 18, 1 << 15, 1 << 12, 1 << 9, 1 << 6, 1 << 3, 1,
            1 << 24, 1 << 21, 1 << 18, 1 << 15, 1 << 12, 1 << 9, 1 << 6, 1 << 3, 1,
            1 << 24, 1 << 21, 1 << 18, 1 << 15, 1 << 12, 1 << 9, 1 << 6, 1 << 3, 1,
        ]
        # 标准九莲的压缩掩码：1112345678999（其中 1 与 9 为 3 张）
        mask = 0b011_001_001_001_001_001_001_001_011

        is_valid = False
        is_true = False
        if win_tile <= Tile.Manzu9:
            is_valid = (
                hand[Tile.Manzu1] >= 3 and hand[Tile.Manzu2] and hand[Tile.Manzu3] and hand[Tile.Manzu4]
                and hand[Tile.Manzu5] and hand[Tile.Manzu6] and hand[Tile.Manzu7] and hand[Tile.Manzu8]
                and hand[Tile.Manzu9] >= 3
            )
            is_true = (manzu - tile1[win_tile]) == mask
        elif win_tile <= Tile.Pinzu9:
            is_valid = (
                hand[Tile.Pinzu1] >= 3 and hand[Tile.Pinzu2] and hand[Tile.Pinzu3] and hand[Tile.Pinzu4]
                and hand[Tile.Pinzu5] and hand[Tile.Pinzu6] and hand[Tile.Pinzu7] and hand[Tile.Pinzu8]
                and hand[Tile.Pinzu9] >= 3
            )
            is_true = (pinzu - tile1[win_tile]) == mask
        elif win_tile <= Tile.Souzu9:
            is_valid = (
                hand[Tile.Souzu1] >= 3 and hand[Tile.Souzu2] and hand[Tile.Souzu3] and hand[Tile.Souzu4]
                and hand[Tile.Souzu5] and hand[Tile.Souzu6] and hand[Tile.Souzu7] and hand[Tile.Souzu8]
                and hand[Tile.Souzu9] >= 3
            )
            is_true = (souzu - tile1[win_tile]) == mask

        if is_valid:
            return Yaku.TrueNineGates if is_true else Yaku.NineGates
        return 0

    @staticmethod
    def check_thirteen_wait_thirteen_orphans(merged, win_tile: int) -> bool:
        """国士无双十三面听（十三面待ち）判定。"""
        tile1 = [
            1 << 24, 1 << 21, 1 << 18, 1 << 15, 1 << 12, 1 << 9, 1 << 6, 1 << 3, 1,
            1 << 24, 1 << 21, 1 << 18, 1 << 15, 1 << 12, 1 << 9, 1 << 6, 1 << 3, 1,
            1 << 24, 1 << 21, 1 << 18, 1 << 15, 1 << 12, 1 << 9, 1 << 6, 1 << 3, 1,
            1 << 18, 1 << 15, 1 << 12, 1 << 9, 1 << 6, 1 << 3, 1,
        ]
        # terminals_mask: 每门幺九（1 与 9）各 1 张
        terminals_mask = 0b001_000_000_000_000_000_000_000_001
        # honors_mask: 七种字牌各 1 张
        honors_mask = 0b001_001_001_001_001_001_001

        _, manzu, pinzu, souzu, honors = merged
        if win_tile <= Tile.Manzu9:
            return (manzu - tile1[win_tile] == terminals_mask) and (pinzu == terminals_mask) and (souzu == terminals_mask) and (honors == honors_mask)
        if win_tile <= Tile.Pinzu9:
            return (manzu == terminals_mask) and (pinzu - tile1[win_tile] == terminals_mask) and (souzu == terminals_mask) and (honors == honors_mask)
        if win_tile <= Tile.Souzu9:
            return (manzu == terminals_mask) and (pinzu == terminals_mask) and (souzu - tile1[win_tile] == terminals_mask) and (honors == honors_mask)
        return (manzu == terminals_mask) and (pinzu == terminals_mask) and (souzu == terminals_mask) and (honors - tile1[win_tile] == honors_mask)

    @staticmethod
    def check_tanyao(player: Player, merged, rule_open_tanyao: bool) -> int:
        """断幺九判定（是否允许副露断幺九由规则控制）。"""
        _, manzu, pinzu, souzu, honors = merged
        if (not rule_open_tanyao) and (not player.is_closed()):
            return 0
        # terminals_mask: 序数牌 1 与 9 的位段
        terminals_mask = 0b111_000_000_000_000_000_000_000_111
        if (manzu & terminals_mask) or (pinzu & terminals_mask) or (souzu & terminals_mask) or honors:
            return 0
        return Yaku.Tanyao

    @staticmethod
    def check_flush(merged) -> int:
        """清一色/混一色判定。

        - 仅一门序数牌：清一色
        - 一门序数牌 + 字牌：混一色
        """
        _, manzu, pinzu, souzu, honors = merged
        is_flush = (manzu != 0) + (pinzu != 0) + (souzu != 0) == 1
        if is_flush:
            return Yaku.HalfFlush if honors else Yaku.FullFlush
        return 0

    @staticmethod
    def check_value_tile(round_: Round, player: Player, merged) -> int:
        """役牌判定（白/发/中、场风、自风）。"""
        hand, *_ = merged
        yaku_list = 0
        if hand[Tile.White] == 3:
            yaku_list |= Yaku.WhiteDragon
        if hand[Tile.Green] == 3:
            yaku_list |= Yaku.GreenDragon
        if hand[Tile.Red] == 3:
            yaku_list |= Yaku.RedDragon

        if hand[round_.wind] == 3:
            if round_.wind == Tile.East:
                yaku_list |= Yaku.RoundWindEast
            elif round_.wind == Tile.South:
                yaku_list |= Yaku.RoundWindSouth
            elif round_.wind == Tile.West:
                yaku_list |= Yaku.RoundWindWest
            elif round_.wind == Tile.North:
                yaku_list |= Yaku.RoundWindNorth

        if hand[player.wind] == 3:
            if player.wind == Tile.East:
                yaku_list |= Yaku.SelfWindEast
            elif player.wind == Tile.South:
                yaku_list |= Yaku.SelfWindSouth
            elif player.wind == Tile.West:
                yaku_list |= Yaku.SelfWindWest
            elif player.wind == Tile.North:
                yaku_list |= Yaku.SelfWindNorth

        return yaku_list

    @staticmethod
    def count_dora(hand, melds, indicators: List[int]) -> int:
        """统计宝牌数量（含手牌与副露，赤宝牌不在此处统计）。"""
        num = 0
        for t in indicators:
            dora = ToDora[t]
            num += int(hand[dora])
            for meld in melds:
                for tile in meld.tiles:
                    num += 1 if to_no_reddora(tile) == dora else 0
        return num

    @staticmethod
    def count_reddora(rule_reddora: bool, hand, melds) -> int:
        """统计赤宝牌数量（规则关闭时直接为 0）。"""
        if not rule_reddora:
            return 0
        num = int(hand[Tile.RedManzu5] + hand[Tile.RedPinzu5] + hand[Tile.RedSouzu5])
        for meld in melds:
            for tile in meld.tiles:
                if is_reddora(tile):
                    num += 1
                    break
        return num

    @staticmethod
    def get_up_scores(round_: Round, player: Player, result: Result, win_flag: int, n: int) -> List[int]:
        """返回当前和牌结果额外增加 0..n 番时的总得点。

        期望值计算会用它估计里宝牌带来的升番收益。
        """
        if not result.success:
            return []
        if result.score_title >= ScoreTitle.CountedYakuman:
            return [result.score[0]]

        scores = [0] * (n + 1)
        is_dealer = player.wind == Tile.East
        is_tsumo = bool(win_flag & WinFlag.Tsumo)
        for i in range(n + 1):
            han = result.han + i
            score_title = ScoreCalculator.get_score_title(result.fu, han)
            scores[i] = ScoreCalculator.calc_score(
                is_dealer,
                is_tsumo,
                round_.honba,
                round_.kyotaku,
                score_title,
                han,
                result.fu,
            )[0]
        return scores

    @staticmethod
    def get_score_title(fu: int, han: int) -> int:
        """根据符/番确定是否为满贯以上，以及对应的段位（满贯/跳满/倍满/三倍满/累计役满）。"""
        fu_idx = fu_to_index(fu)
        han_idx = han - 1
        if han < 5:
            return ScoreTitle.Mangan if IsMangan[fu_idx][han_idx] else ScoreTitle.Null
        if han == 5:
            return ScoreTitle.Mangan
        if han <= 7:
            return ScoreTitle.Haneman
        if han <= 10:
            return ScoreTitle.Baiman
        if han <= 12:
            return ScoreTitle.Sanbaiman
        return ScoreTitle.CountedYakuman

    @staticmethod
    def get_score_title_yakuman(n: int) -> int:
        """根据役满倍数 n 返回役满段位（役满/双倍役满/…）。"""
        if n == 1:
            return ScoreTitle.Yakuman
        if n == 2:
            return ScoreTitle.DoubleYakuman
        if n == 3:
            return ScoreTitle.TripleYakuman
        if n == 4:
            return ScoreTitle.QuadrupleYakuman
        if n == 5:
            return ScoreTitle.QuintupleYakuman
        if n == 6:
            return ScoreTitle.SextupleYakuman
        return ScoreTitle.Null

    @staticmethod
    def calc_score(is_dealer: bool, is_tsumo: bool, honba: int, kyotaku: int, score_title: int, han: int = 0, fu: int = 0) -> List[int]:
        """根据番/符与是否满贯以上，计算最终点数。

        返回格式：
        - 自摸： [总得点, (庄家支付), (闲家支付)]（庄家自摸时为 [总得点, 闲家支付]）
        - 荣和： [总得点, 放铳者支付]
        总得点包含：供托（kyotaku*1000） + 本场（honba）的加点。
        """
        fu_idx = fu_to_index(fu)
        han_idx = han - 1

        if is_tsumo and is_dealer:
            player_payment = (
                BelowMangan[TsumoPlayerToDealer][fu_idx][han_idx]
                if score_title == ScoreTitle.Null
                else AboveMangan[TsumoPlayerToDealer][score_title]
            ) + 100 * honba
            score = 1000 * kyotaku + player_payment * 3
            return [score, player_payment]

        if is_tsumo and not is_dealer:
            dealer_payment = (
                BelowMangan[TsumoDealerToPlayer][fu_idx][han_idx]
                if score_title == ScoreTitle.Null
                else AboveMangan[TsumoDealerToPlayer][score_title]
            ) + 100 * honba
            player_payment = (
                BelowMangan[TsumoPlayerToPlayer][fu_idx][han_idx]
                if score_title == ScoreTitle.Null
                else AboveMangan[TsumoPlayerToPlayer][score_title]
            ) + 100 * honba
            score = 1000 * kyotaku + dealer_payment + player_payment * 2
            return [score, dealer_payment, player_payment]

        if (not is_tsumo) and is_dealer:
            payment = (
                BelowMangan[RonDiscarderToDealer][fu_idx][han_idx]
                if score_title == ScoreTitle.Null
                else AboveMangan[RonDiscarderToDealer][score_title]
            ) + 300 * honba
            score = 1000 * kyotaku + payment
            return [score, payment]

        payment = (
            BelowMangan[RonDiscarderToPlayer][fu_idx][han_idx]
            if score_title == ScoreTitle.Null
            else AboveMangan[RonDiscarderToPlayer][score_title]
        ) + 300 * honba
        score = 1000 * kyotaku + payment
        return [score, payment]
