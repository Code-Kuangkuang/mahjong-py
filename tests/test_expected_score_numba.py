from __future__ import annotations

import pytest

from mahjong_py.constants import Tile
from mahjong_py.expected_score_calculator import (
    ExpectedScoreCalculator,
    ExpectedScoreConfig,
)
from mahjong_py.expected_score_numba import ExpectedScoreNumbaCalculator
from mahjong_py.mpsz import from_mpsz
from mahjong_py.score_types import Player, Round, RuleFlag
from mahjong_py.shanten import ShantenFlag


def _assert_stats_match(old_stats, new_stats):
    assert len(new_stats) == len(old_stats)
    for old, new in zip(old_stats, new_stats):
        assert new.tile == old.tile
        assert new.shanten == old.shanten
        assert new.necessary_tiles == old.necessary_tiles
        assert new.tenpai_prob == pytest.approx(old.tenpai_prob)
        assert new.win_prob == pytest.approx(old.win_prob)
        assert new.exp_score == pytest.approx(old.exp_score)


def test_numba_expected_score_matches_python_sample():
    player = Player(hand=from_mpsz("222567m345p33667s"), wind=Tile.East)
    round_ = Round(
        rules=RuleFlag.OpenTanyao | RuleFlag.RedDora,
        wind=Tile.East,
        dora_indicators=[Tile.East],
    )
    config = ExpectedScoreConfig(
        t_min=1,
        t_max=18,
        extra=1,
        shanten_type=ShantenFlag.All,
        enable_reddora=True,
        enable_uradora=True,
        enable_shanten_down=True,
        enable_tegawari=True,
        enable_riichi=False,
    )

    old_stats, old_searched = ExpectedScoreCalculator.calc(config, round_, player)
    new_stats, new_searched = ExpectedScoreNumbaCalculator.calc(config, round_, player)

    assert new_searched == old_searched == 256
    _assert_stats_match(old_stats, new_stats)


def test_numba_expected_score_matches_python_tenpai_case():
    player = Player(hand=from_mpsz("123m123p123s11z77z"), wind=Tile.East)
    round_ = Round(wind=Tile.East)
    config = ExpectedScoreConfig(
        t_max=3,
        extra=1,
        shanten_type=ShantenFlag.All,
        enable_uradora=False,
        enable_shanten_down=False,
        enable_tegawari=False,
    )

    old_stats, old_searched = ExpectedScoreCalculator.calc(config, round_, player)
    new_stats, new_searched = ExpectedScoreNumbaCalculator.calc(config, round_, player)

    assert new_searched == old_searched == 3
    _assert_stats_match(old_stats, new_stats)


def test_numba_expected_score_matches_python_user_case():
    player = Player(hand=from_mpsz("06778p1122345s77z"), wind=Tile.West)
    round_ = Round(
        rules=RuleFlag.OpenTanyao | RuleFlag.RedDora,
        wind=Tile.East,
        dora_indicators=[Tile.South],
    )
    config = ExpectedScoreConfig(
        t_min=1,
        t_max=18,
        extra=1,
        shanten_type=ShantenFlag.All,
        enable_reddora=True,
        enable_uradora=True,
        enable_shanten_down=True,
        enable_tegawari=True,
        enable_riichi=False,
    )

    old_stats, old_searched = ExpectedScoreCalculator.calc(config, round_, player)
    new_stats, new_searched = ExpectedScoreNumbaCalculator.calc(config, round_, player)

    assert new_searched == old_searched == 4471
    _assert_stats_match(old_stats, new_stats)
