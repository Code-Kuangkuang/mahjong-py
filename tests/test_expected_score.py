from __future__ import annotations

import pytest

from mahjong_py.constants import Tile
from mahjong_py.expected_score_calculator import (
    ExpectedScoreCalculator,
    ExpectedScoreConfig,
)
from mahjong_py.mpsz import from_mpsz
from mahjong_py.score_types import Player, Round, RuleFlag
from mahjong_py.server import calc
from mahjong_py.shanten import ShantenFlag


def test_expected_score_tenpai_hand_has_probabilities():
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

    stats, searched = ExpectedScoreCalculator.calc(config, round_, player)

    assert searched == 3
    assert len(stats) == 1
    stat = stats[0]
    assert stat.tile == Tile.Null
    assert stat.shanten == 0
    assert stat.necessary_tiles == [(Tile.East, 2), (Tile.Red, 2)]
    assert stat.tenpai_prob[1] == 1.0
    assert 0.0 < stat.win_prob[1] <= 1.0
    assert stat.exp_score[1] > 0.0


def test_calc_endpoint_returns_expected_score_stats():
    payload = {
        "enable_reddora": True,
        "enable_uradora": False,
        "enable_shanten_down": True,
        "enable_tegawari": True,
        "enable_riichi": False,
        "round_wind": Tile.East,
        "dora_indicators": [],
        "hand": [
            Tile.Manzu1,
            Tile.Manzu2,
            Tile.Manzu3,
            Tile.Pinzu1,
            Tile.Pinzu2,
            Tile.Pinzu3,
            Tile.Souzu1,
            Tile.Souzu2,
            Tile.Souzu3,
            Tile.East,
            Tile.East,
            Tile.Red,
            Tile.Red,
        ],
        "melds": [],
        "seat_wind": Tile.East,
        "version": "0.1.0",
    }

    result = calc(payload)

    assert result["success"] is True
    response = result["response"]
    assert response["shanten"]["all"] == 0
    assert response["searched"] > 0
    assert response["time"] >= 0
    assert response["config"]["calc_stats"] is True
    assert len(response["stats"]) == 1
    stat = response["stats"][0]
    assert stat["tile"] == Tile.Null
    assert stat["necessary_tiles"] == [
        {"tile": Tile.East, "count": 2},
        {"tile": Tile.Red, "count": 2},
    ]
    assert stat["tenpai_prob"][1] == 1.0
    assert stat["win_prob"][1] > 0.0
    assert stat["exp_score"][1] > 0.0


def test_expected_score_matches_cpp_sample_probabilities():
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

    stats, searched = ExpectedScoreCalculator.calc(config, round_, player)

    assert searched == 256
    assert [stat.tile for stat in stats] == [
        Tile.Manzu2,
        Tile.Manzu5,
        Tile.Manzu6,
        Tile.Manzu7,
        Tile.Pinzu3,
        Tile.Pinzu4,
        Tile.Pinzu5,
        Tile.Souzu3,
        Tile.Souzu6,
        Tile.Souzu7,
    ]

    # The C++ Stat arrays are 1-based by turn; index 0 is intentionally unused.
    first = stats[0]
    assert first.tenpai_prob[1] == pytest.approx(0.8972, abs=5e-5)
    assert first.win_prob[1] == pytest.approx(0.3417, abs=5e-5)
    assert first.exp_score[1] == pytest.approx(2913.96, abs=0.5)
    assert first.tenpai_prob[17] == pytest.approx(0.1346, abs=5e-5)
    assert first.win_prob[17] == pytest.approx(0.0)
    assert first.exp_score[17] == pytest.approx(0.0)
    assert first.tenpai_prob[18] == pytest.approx(0.0)

    tenpai_discard = stats[8]
    assert tenpai_discard.tenpai_prob[1] == pytest.approx(1.0)
    assert tenpai_discard.win_prob[1] == pytest.approx(0.7170, abs=5e-5)
    assert tenpai_discard.exp_score[1] == pytest.approx(6427.14, abs=0.5)
    assert tenpai_discard.tenpai_prob[18] == pytest.approx(1.0)
    assert tenpai_discard.win_prob[18] == pytest.approx(0.0)
