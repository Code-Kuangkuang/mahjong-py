"""命令行工具（CLI）。

提供一个简单的命令行入口：
- 计分：`python -m mahjong_py.cli HAND WIN_TILE`
- 向听：`python -m mahjong_py.cli HAND` 或 `--mode shanten`

输出均为 JSON。
"""

from __future__ import annotations

import json
from typing import List

import typer

from .mpsz import from_mpsz, to_hand34
from .score_calculator import ScoreCalculator
from .score_constants import WinFlag, YakuName, YakuNameZh
from .score_types import Meld, Player, Round
from .utils import to_no_reddora
from .constants import Tile, TileNameZh34
from .shanten import ShantenFlag, shanten, shanten_regular, shanten_seven_pairs, shanten_thirteen_orphans
from .tables import load_tables
from .necessary import necessary_tiles
from .unnecessary import unnecessary_tiles


def _parse_tile(token: str) -> int:
    """解析单张牌。

    支持：
    - 纯数字 id（0..36）
    - mpsz 形式（如 5m/0p/7z）
    """
    token = token.strip().lower()
    if token.isdigit():
        return int(token)

    if len(token) != 2:
        raise ValueError(f"Invalid tile token: {token}")

    num = token[0]
    suit = token[1]
    if num not in "0123456789" or suit not in "mpsz":
        raise ValueError(f"Invalid tile token: {token}")

    # reuse from_mpsz for single tile
    hand = from_mpsz(f"{num}{suit}")
    for i in range(37):
        if hand[i] > 0:
            return i
    raise ValueError(f"Invalid tile token: {token}")


def _parse_flags(flags: List[str]) -> int:
    """将字符串 flags 转成 WinFlag 位标志。"""
    mapping = {
        "tsumo": WinFlag.Tsumo,
        "riichi": WinFlag.Riichi,
        "driichi": WinFlag.DoubleRiichi,
        "ippatsu": WinFlag.Ippatsu,
        "rob": WinFlag.RobbingAKong,
        "afterkong": WinFlag.AfterAKong,
        "undersea": WinFlag.UnderTheSea,
        "underriver": WinFlag.UnderTheRiver,
        "nagashi": WinFlag.NagashiMangan,
        "tenhou": WinFlag.BlessingOfHeaven,
        "chihou": WinFlag.BlessingOfEarth,
        "renhou": WinFlag.HandOfMan,
    }
    val = 0
    for f in flags:
        key = f.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unknown flag: {f}")
        val |= mapping[key]
    return val


app = typer.Typer(help="麻将工具集（Python）：计分/向听等")


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


def _tile34_to_zh(tile34: int) -> str:
    if tile34 < 0 or tile34 >= 34:
        raise ValueError(f"tile34 out of range: {tile34}")
    return TileNameZh34[tile34]


def _tiles_display(tiles34: list[int], lang_key: str) -> list[str]:
    if lang_key == "zh":
        return [_tile34_to_zh(t) for t in tiles34]
    return [_tile34_to_token(t) for t in tiles34]


def _run_score(
    hand: str,
    win_tile: str,
    seat: int,
    round_wind: int,
    dora: List[str] | None,
    uradora: List[str] | None,
    honba: int,
    kyotaku: int,
    flags: List[str] | None,
    meld_json: str | None,
    meld_file: str | None,
    lang: str,
) -> None:
    hand_arr = from_mpsz(hand)
    win_tile_id = _parse_tile(win_tile)

    # 约定：hand 参数要么是 13 张（不含和牌张），要么是 14 张（已含和牌张）。
    total_tiles = int(hand_arr[:34].sum())
    if total_tiles not in (13, 14):
        raise typer.BadParameter(
            f"HAND 需要是 13 张（不含和牌张）或 14 张（已含和牌张），当前为 {total_tiles} 张。"
        )

    # 若传入 13 张，则把 win_tile 补进来（与 C++ 侧接口对齐：14 张）
    if total_tiles == 13:
        if win_tile_id == Tile.RedManzu5:
            hand_arr[Tile.RedManzu5] += 1
            hand_arr[Tile.Manzu5] += 1
        elif win_tile_id == Tile.RedPinzu5:
            hand_arr[Tile.RedPinzu5] += 1
            hand_arr[Tile.Pinzu5] += 1
        elif win_tile_id == Tile.RedSouzu5:
            hand_arr[Tile.RedSouzu5] += 1
            hand_arr[Tile.Souzu5] += 1
        else:
            hand_arr[win_tile_id] += 1

    round_ = Round()
    round_.wind = round_wind
    round_.honba = honba
    round_.kyotaku = kyotaku
    if dora:
        round_.set_dora_indicators([_parse_tile(x) for x in dora])
    if uradora:
        round_.set_uradora_indicators([_parse_tile(x) for x in uradora])

    melds = []
    if meld_file:
        with open(meld_file, "r", encoding="utf-8") as f:
            meld_data = json.load(f)
    elif meld_json:
        meld_data = json.loads(meld_json)
    else:
        meld_data = []

    if meld_data:
        for m in meld_data:
            melds.append(
                Meld(
                    type=int(m["type"]),
                    tiles=[int(x) for x in m["tiles"]],
                    discarded_tile=int(m.get("discarded_tile", -1)),
                    from_player=int(m.get("from_player", -1)),
                )
            )

    player = Player(hand=hand_arr, wind=seat, melds=melds)
    win_flag = _parse_flags(flags or [])

    res = ScoreCalculator.calc(round_, player, win_tile_id, win_flag)

    lang_key = lang.strip().lower()
    if lang_key not in ("en", "zh"):
        raise typer.BadParameter("--lang must be 'en' or 'zh'")
    yaku_name = YakuNameZh if lang_key == "zh" else YakuName

    out = {
        "success": res.success,
        "err_msg": res.err_msg,
        "han": res.han,
        "fu": res.fu,
        "score_title": res.score_title,
        "score": res.score,
        "yaku_list": [
            {"yaku": y, "name": yaku_name.get(y, str(y)), "han": h} for y, h in res.yaku_list
        ],
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))


@app.command()
def main(
    hand: str = typer.Argument(..., help="手牌（mpsz），例如 123m789p123s77z"),
    win_tile: str | None = typer.Argument(
        None,
        help="和牌（0-36 或 mpsz，如 5m/0p/7z）。省略时计算向听数",
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="运行模式：auto/score/shanten（auto: 有 WIN_TILE 则计分，否则向听）",
    ),
    seat: int = typer.Option(0, help="自风牌 id（27-30）"),
    round_wind: int = typer.Option(27, "--round", help="场风牌 id（27-30）"),
    dora: List[str] = typer.Option(
        None, help="宝牌指示（牌 id 或 mpsz，如 5m）"
    ),
    uradora: List[str] = typer.Option(
        None, help="里宝牌指示（牌 id 或 mpsz，如 5m）"
    ),
    honba: int = typer.Option(0, help="本场数"),
    kyotaku: int = typer.Option(0, help="立直棒数"),
    flags: List[str] = typer.Option(
        None,
        help=(
            "和牌标志：tsumo/riichi/driichi/ippatsu/rob/afterkong/"
            "undersea/underriver/nagashi/tenhou/chihou/renhou"
        ),
    ),
    meld_json: str | None = typer.Option(
        None,
        help=(
            "副露 JSON 列表，例如 "
            "[{\"type\":1,\"tiles\":[0,1,2]},"
            "{\"type\":0,\"tiles\":[13,13,13],\"discarded_tile\":13,\"from_player\":1}]"
        ),
    ),
    meld_file: str | None = typer.Option(
        None,
        help="副露 JSON 文件路径（格式同 --meld-json）",
    ),
    melds: int = typer.Option(0, "--melds", help="副露数量（向听一般形计算会扣除面子数）"),
    types: List[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="向听计算类型：regular/7pairs/orphans/all（可多次传入）",
    ),
    details: bool = typer.Option(False, "--details", help="向听模式输出进张/打张等明细"),
    lang: str = typer.Option(
        "en",
        "--lang",
        "-l",
        help="役名输出语言：en/zh",
    ),
) -> None:
    """CLI 主命令：计分/向听（由 `--mode` 或是否提供 WIN_TILE 决定）。"""

    mode_key = mode.strip().lower()
    if mode_key not in ("auto", "score", "shanten"):
        raise typer.BadParameter("--mode 必须是 auto/score/shanten")

    do_shanten = False
    if mode_key == "shanten":
        do_shanten = True
    elif mode_key == "score":
        do_shanten = False
    else:
        do_shanten = win_tile is None

    if do_shanten:
        lang_key = lang.strip().lower()
        if lang_key not in ("en", "zh"):
            raise typer.BadParameter("--lang must be 'en' or 'zh'")

        hand37 = from_mpsz(hand)
        hand34 = to_hand34(hand37)
        total_tiles = int(hand34.sum())
        if total_tiles not in (13, 14):
            raise typer.BadParameter(f"HAND 需要是 13 张或 14 张（当前为 {total_tiles} 张）。")
        if melds < 0 or melds > 4:
            raise typer.BadParameter("--melds 必须在 0..4 之间")

        type_mask = _parse_shanten_types(types)

        best_type, best = shanten(hand34, melds, type_mask)
        best_types: List[str] = []
        if best_type & ShantenFlag.Regular:
            best_types.append("regular")
        if best_type & ShantenFlag.SevenPairs:
            best_types.append("7pairs")
        if best_type & ShantenFlag.ThirteenOrphans:
            best_types.append("orphans")

        regular = None
        seven = None
        orphans = None
        if type_mask & ShantenFlag.Regular:
            suits_raw, honors_raw = load_tables()
            regular = int(shanten_regular(hand34, melds, suits_raw, honors_raw))
        if (type_mask & ShantenFlag.SevenPairs) and melds == 0:
            seven = int(shanten_seven_pairs(hand34))
        if (type_mask & ShantenFlag.ThirteenOrphans) and melds == 0:
            orphans = int(shanten_thirteen_orphans(hand34))

        out: dict = {
            "mode": "shanten",
            "tiles": total_tiles,
            "melds": melds,
            "best": best,
            "best_types": best_types,
            "regular": regular,
            "seven_pairs": seven,
            "thirteen_orphans": orphans,
        }

        # 进张/打张/听牌明细
        if details:
            if total_tiles == 13:
                _, best_s, tiles_need = necessary_tiles(hand34, melds, type_mask)
                out["necessary_tiles"] = tiles_need
                out["necessary_mpsz"] = [_tile34_to_token(t) for t in tiles_need]
                out["necessary_zh"] = [_tile34_to_zh(t) for t in tiles_need]
                out["necessary_display"] = _tiles_display(tiles_need, lang_key)

                # 听牌：向听=0 时，进张就是“和了张”
                if best == 0 and best_s == 0:
                    out["waits_tiles"] = tiles_need
                    out["waits_mpsz"] = out["necessary_mpsz"]
                    out["waits_zh"] = out["necessary_zh"]
                    out["waits_display"] = out["necessary_display"]
            else:
                # 14 张：输出“不掉向听的打张”，并可选计算“打这张后进张/听牌”
                _, best_s, discards = unnecessary_tiles(hand34, melds, type_mask)
                out["unnecessary_discards"] = discards
                out["unnecessary_discards_mpsz"] = [_tile34_to_token(t) for t in discards]
                out["unnecessary_discards_zh"] = [_tile34_to_zh(t) for t in discards]
                out["unnecessary_discards_display"] = _tiles_display(discards, lang_key)

                discard_details: dict[str, dict] = {}
                for t in discards:
                    if hand34[t] <= 0:
                        continue
                    tmp = hand34.copy()
                    tmp[t] -= 1
                    bt2, s2, needs = necessary_tiles(tmp, melds, type_mask)
                    wait_mpsz = [_tile34_to_token(x) for x in needs]
                    wait_zh = [_tile34_to_zh(x) for x in needs]
                    entry: dict = {
                        "after_shanten": int(s2),
                        "necessary_tiles": needs,
                        "necessary_mpsz": wait_mpsz,
                        "necessary_zh": wait_zh,
                        "necessary_display": _tiles_display(needs, lang_key),
                    }
                    if s2 == 0:
                        entry["waits_tiles"] = needs
                        entry["waits_mpsz"] = wait_mpsz
                        entry["waits_zh"] = wait_zh
                        entry["waits_display"] = entry["necessary_display"]
                    discard_details[_tile34_to_token(t)] = entry
                out["discard_analysis"] = discard_details

        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    # score
    if win_tile is None:
        raise typer.BadParameter("计分模式需要提供 WIN_TILE（或改用 --mode shanten）")

    _run_score(
        hand=hand,
        win_tile=win_tile,
        seat=seat,
        round_wind=round_wind,
        dora=dora,
        uradora=uradora,
        honba=honba,
        kyotaku=kyotaku,
        flags=flags,
        meld_json=meld_json,
        meld_file=meld_file,
        lang=lang,
    )


def _parse_shanten_types(types: List[str] | None) -> int:
    if not types:
        return ShantenFlag.All

    mapping = {
        "all": ShantenFlag.All,
        "regular": ShantenFlag.Regular,
        "7pairs": ShantenFlag.SevenPairs,
        "sevenpairs": ShantenFlag.SevenPairs,
        "chiitoi": ShantenFlag.SevenPairs,
        "orphans": ShantenFlag.ThirteenOrphans,
        "13orphans": ShantenFlag.ThirteenOrphans,
        "kokushi": ShantenFlag.ThirteenOrphans,
    }

    mask = 0
    for t in types:
        key = t.strip().lower()
        if key not in mapping:
            raise typer.BadParameter(
                "--type 支持：regular/7pairs/orphans/all（也可多次传入 --type）"
            )
        v = mapping[key]
        if v == ShantenFlag.All:
            return ShantenFlag.All
        mask |= v
    return mask
if __name__ == "__main__":
    app()
