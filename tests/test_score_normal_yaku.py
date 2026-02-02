from __future__ import annotations

from pathlib import Path

import json

from mahjong_py.score_calculator import ScoreCalculator
from mahjong_py.score_types import Meld, Player, Round, hand_from_tiles


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cpp_testcase(name: str) -> Path:
    return Path(__file__).resolve().parent / "testcase" / name


def _load_cases(limit: int = 200):
    path = _cpp_testcase("test_score_normal_yaku.json")
    decoder = json.JSONDecoder()
    buf = ""
    in_array = False
    count = 0

    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk

            while True:
                buf = buf.lstrip()
                if not in_array:
                    if not buf:
                        break
                    if buf[0] == "[":
                        in_array = True
                        buf = buf[1:]
                    else:
                        raise ValueError("Invalid JSON: expected array")

                buf = buf.lstrip()
                if buf.startswith("]"):
                    return
                if buf.startswith(","):
                    buf = buf[1:]
                    continue

                try:
                    obj, idx = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    break

                yield obj
                count += 1
                buf = buf[idx:]
                if count >= limit:
                    return


def test_score_normal_yaku_subset():
    for v in _load_cases():
        round_ = Round()
        round_.rules = 0
        if v["akadora"]:
            from mahjong_py.score_types import RuleFlag

            round_.rules |= RuleFlag.RedDora
        if v["kuitan"]:
            from mahjong_py.score_types import RuleFlag

            round_.rules |= RuleFlag.OpenTanyao

        round_.wind = int(v["bakaze"])
        round_.honba = int(v["num_tumibo"])
        round_.kyotaku = int(v["num_kyotakubo"])
        round_.set_dora([int(x) for x in v["dora_tiles"]])
        round_.set_uradora([int(x) for x in v["uradora_tiles"]])

        tiles = [int(x) for x in v["hand_tiles"]]
        melds = []
        for m in v["melded_blocks"]:
            meld = Meld(
                type=int(m["type"]),
                tiles=[int(x) for x in m["tiles"]],
                discarded_tile=int(m["discarded_tile"]),
                from_player=int(m["from"]),
            )
            melds.append(meld)

        player = Player(hand=hand_from_tiles(tiles), melds=melds, wind=int(v["zikaze"]))
        win_tile = int(v["win_tile"])
        win_flag = int(v["flag"])

        ret = ScoreCalculator.calc(round_, player, win_tile, win_flag)

        assert ret.han == int(v["han"])
        assert ret.fu == int(v["hu"])
        assert ret.score_title == int(v["score_title"])

        expected_yaku_list = [(int(x[0]), int(x[1])) for x in v["yaku_list"]]
        assert len(ret.yaku_list) == len(expected_yaku_list)
        for i, item in enumerate(expected_yaku_list):
            assert ret.yaku_list[i][0] == item[0]
            assert ret.yaku_list[i][1] == item[1]

        expected_score = [int(x) for x in v["score"]]
        assert ret.score == expected_score
