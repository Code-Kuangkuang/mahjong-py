from __future__ import annotations

from pathlib import Path
import os

from mahjong_py.score_calculator import ScoreCalculator
from mahjong_py.score_constants import WinFlag, Yaku
from mahjong_py.score_types import Player, Round, hand_from_tiles


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cpp_testcase(name: str) -> Path:
    return Path(__file__).resolve().parent / "testcase" / name


def _load_cases(path: Path, limit: int | None = None):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            parts = [int(x) for x in line.split()]
            tiles = parts[:14]
            win_tile = parts[14]
            expected = parts[15] == 1
            yield tiles, win_tile, expected


def _make_player(tiles):
    hand = hand_from_tiles(tiles)
    return Player(hand=hand, wind=0, melds=[])


def _yakuman_limit() -> int:
    return int(os.environ.get("MAHJONG_PY_YAKUMAN_LIMIT", "2000"))


def test_all_green():
    for tiles, win_tile, expected in _load_cases(
        _cpp_testcase("test_score_calculator_all_green.txt"), _yakuman_limit()
    ):
        player = _make_player(tiles)
        merged = ScoreCalculator.merge_hand(player)
        actual = ScoreCalculator.check_all_green(merged) != 0
        assert actual == expected


def test_big_three_dragons():
    for tiles, win_tile, expected in _load_cases(
        _cpp_testcase("test_score_calculator_big_three_dragons.txt"), _yakuman_limit()
    ):
        player = _make_player(tiles)
        merged = ScoreCalculator.merge_hand(player)
        actual = ScoreCalculator.check_three_dragons(merged) == Yaku.BigThreeDragons
        assert actual == expected


def test_little_four_winds():
    for tiles, win_tile, expected in _load_cases(
        _cpp_testcase("test_score_calculator_little_four_winds.txt"), _yakuman_limit()
    ):
        player = _make_player(tiles)
        merged = ScoreCalculator.merge_hand(player)
        actual = ScoreCalculator.check_four_winds(merged) == Yaku.LittleFourWinds
        assert actual == expected


def test_all_honors():
    for tiles, win_tile, expected in _load_cases(
        _cpp_testcase("test_score_calculator_all_honors.txt"), _yakuman_limit()
    ):
        player = _make_player(tiles)
        merged = ScoreCalculator.merge_hand(player)
        actual = ScoreCalculator.check_all_honors(merged) != 0
        assert actual == expected


def test_nine_gates():
    for tiles, win_tile, expected in _load_cases(
        _cpp_testcase("test_score_calculator_nine_gates.txt"), _yakuman_limit()
    ):
        player = _make_player(tiles)
        merged = ScoreCalculator.merge_hand(player)
        actual = ScoreCalculator.check_nine_gates(player, merged, win_tile) == Yaku.NineGates
        assert actual == expected


def test_true_nine_gates():
    for tiles, win_tile, expected in _load_cases(
        _cpp_testcase("test_score_calculator_true_nine_gates.txt"), _yakuman_limit()
    ):
        player = _make_player(tiles)
        merged = ScoreCalculator.merge_hand(player)
        actual = ScoreCalculator.check_nine_gates(player, merged, win_tile) == Yaku.TrueNineGates
        assert actual == expected


def test_four_concealed_triplets():
    for tiles, win_tile, expected in _load_cases(
        _cpp_testcase("test_score_calculator_four_concealed_triplets.txt"), _yakuman_limit()
    ):
        player = _make_player(tiles)
        merged = ScoreCalculator.merge_hand(player)
        actual = ScoreCalculator.check_four_concealed_triplets(player, merged, win_tile, WinFlag.Tsumo) != 0
        assert actual == expected


def test_single_wait_four_concealed_triplets():
    for tiles, win_tile, expected in _load_cases(
        _cpp_testcase("test_score_calculator_single_wait_four_concealed_triplets.txt"),
        _yakuman_limit(),
    ):
        player = _make_player(tiles)
        merged = ScoreCalculator.merge_hand(player)
        actual = ScoreCalculator.check_four_concealed_triplets(player, merged, win_tile, WinFlag.Tsumo) == Yaku.SingleWaitFourConcealedTriplets
        assert actual == expected


def test_all_terminals():
    for tiles, win_tile, expected in _load_cases(
        _cpp_testcase("test_score_calculator_all_terminals.txt"), _yakuman_limit()
    ):
        player = _make_player(tiles)
        merged = ScoreCalculator.merge_hand(player)
        actual = ScoreCalculator.check_all_terminals(merged) != 0
        assert actual == expected


def test_big_four_winds():
    for tiles, win_tile, expected in _load_cases(
        _cpp_testcase("test_score_calculator_big_four_winds.txt"), _yakuman_limit()
    ):
        player = _make_player(tiles)
        merged = ScoreCalculator.merge_hand(player)
        actual = ScoreCalculator.check_four_winds(merged) == Yaku.BigFourWinds
        assert actual == expected


def test_thirteen_wait_thirteen_orphans():
    for tiles, win_tile, expected in _load_cases(
        _cpp_testcase("test_score_calculator_thirteen_wait_thirteen_orphans.txt"), _yakuman_limit()
    ):
        player = _make_player(tiles)
        merged = ScoreCalculator.merge_hand(player)
        actual = ScoreCalculator.check_thirteen_wait_thirteen_orphans(merged, win_tile)
        assert actual == expected
