"""HTTP 服务（FastAPI）。

提供：
- `/calc`：向听/进张/无效牌与简单枚举（面向前端）
- `/score`：点数计算与役种明细
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from .constants import Tile
from .score_calculator import ScoreCalculator
from .score_constants import WinFlag, YakuName
from .score_types import Meld, Player, Round, RuleFlag, hand_from_tiles
from .expected_score_calculator import (
    ExpectedScoreCalculator,
    ExpectedScoreConfig,
    ExpectedScoreStat,
)
from .expected_score_numba import ExpectedScoreNumbaCalculator
from .shanten import ShantenFlag, shanten
from .necessary import necessary_tiles
from .unnecessary import unnecessary_tiles


class MeldIn(BaseModel):
    """请求体中的副露结构（简化版）。"""
    type: int = Field(ge=0, le=4)
    tiles: List[int]


class RequestIn(BaseModel):
    """/calc 请求体。"""
    enable_reddora: bool
    enable_uradora: bool
    enable_shanten_down: bool
    enable_tegawari: bool
    enable_riichi: bool
    round_wind: int = Field(..., ge=27, le=30)
    dora_indicators: List[int] = Field(default_factory=list)
    hand: Optional[List[int]] = None
    hand_tiles: Optional[List[int]] = None
    melds: List[MeldIn] = Field(default_factory=list)
    seat_wind: int = Field(..., ge=27, le=30)
    version: Optional[str] = None
    wall: Optional[List[int]] = None
    ip: Optional[str] = None
    use_numba: bool = True  # True = use Numba accelerator


class ScoreRequestIn(BaseModel):
    """/score 请求体。"""
    hand: Optional[List[int]] = None
    hand_tiles: Optional[List[int]] = None
    melds: List[MeldIn] = Field(default_factory=list)
    seat_wind: int = Field(..., ge=27, le=30)
    round_wind: int = Field(..., ge=27, le=30)
    dora_indicators: List[int] = Field(default_factory=list)
    uradora_indicators: List[int] = Field(default_factory=list)
    win_tile: int = Field(..., ge=0, le=36)
    win_flag: int = 0
    enable_reddora: bool = True


app = FastAPI(title="mahjong-py server", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_tiles(tiles: List[int]) -> None:
    """校验牌 id 范围（0..36）。"""
    if any((t < 0 or t > 36) for t in tiles):
        raise ValueError("tile out of range")


def _validate_wall(wall: List[int]) -> None:
    """校验牌山计数。"""
    if len(wall) != 37:
        raise ValueError("wall must contain 37 counts")
    for count in wall:
        if count < 0 or count > 4:
            raise ValueError("wall count out of range")


def _create_wall(round_: Round, player: Player, enable_reddora: bool) -> List[int]:
    """根据玩家手牌/副露/宝牌指示，推算牌山剩余张数（用于有效牌枚举）。"""
    return ExpectedScoreCalculator.create_wall(round_, player, enable_reddora)


def _apply_discard(hand: List[int], tile: int) -> List[int]:
    """对计数手牌应用一次舍牌（考虑赤 5 与普通 5 的联动）。"""
    new_hand = list(hand)
    new_hand[tile] -= 1
    if tile == Tile.RedManzu5:
        new_hand[Tile.Manzu5] -= 1
    elif tile == Tile.RedPinzu5:
        new_hand[Tile.Pinzu5] -= 1
    elif tile == Tile.RedSouzu5:
        new_hand[Tile.Souzu5] -= 1
    return new_hand


def _wall_after_discard(wall: List[int], tile: int) -> List[int]:
    """更新牌山：假设打出 tile，则该牌回到可摸的“剩余池”。"""
    new_wall = list(wall)
    if tile == Tile.RedManzu5:
        new_wall[Tile.RedManzu5] += 1
        new_wall[Tile.Manzu5] += 1
    elif tile == Tile.RedPinzu5:
        new_wall[Tile.RedPinzu5] += 1
        new_wall[Tile.Pinzu5] += 1
    elif tile == Tile.RedSouzu5:
        new_wall[Tile.RedSouzu5] += 1
        new_wall[Tile.Souzu5] += 1
    else:
        new_wall[tile] += 1
    return new_wall


def _parse_request(payload: Dict[str, Any]) -> RequestIn:
    """将 dict 解析为 RequestIn，并把校验错误转成 ValueError。"""
    try:
        return RequestIn(**payload)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def _build_player(req: RequestIn) -> Player:
    """从 /calc 请求构造 Player（包含手牌与副露）。"""
    tiles = req.hand if req.hand is not None else req.hand_tiles
    if tiles is None:
        raise ValueError("hand is required")
    _validate_tiles(tiles)
    hand = hand_from_tiles(tiles)

    melds = [Meld(type=m.type, tiles=m.tiles) for m in req.melds]
    return Player(hand=hand, melds=melds, wind=req.seat_wind)


def _build_round(req: RequestIn) -> Round:
    """从 /calc 请求构造 Round（规则与场风、宝牌指示）。"""
    round_ = Round()
    round_.rules = 0
    round_.rules |= RuleFlag.RedDora if req.enable_reddora else 0
    round_.rules |= RuleFlag.OpenTanyao if True else 0
    round_.wind = req.round_wind
    round_.dora_indicators = list(req.dora_indicators)
    return round_


def _dump_expected_stats(stats: List[ExpectedScoreStat]) -> List[Dict[str, Any]]:
    """将期望值统计结果转换成 JSON 友好的结构。"""
    out = []
    for stat in stats:
        out.append(
            {
                "tile": stat.tile,
                "tenpai_prob": [min(float(x), 1.0) for x in stat.tenpai_prob],
                "win_prob": [min(float(x), 1.0) for x in stat.win_prob],
                "exp_score": [float(x) for x in stat.exp_score],
                "necessary_tiles": [
                    {"tile": tile, "count": count}
                    for tile, count in stat.necessary_tiles
                ],
                "shanten": stat.shanten,
            }
        )
    return out


@app.get("/health")
def health() -> Dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """简单的调试页面。"""
    return """
<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>mahjong-py</title>
        <style>
            body { font-family: system-ui, sans-serif; margin: 24px; }
            textarea, input { width: 100%; max-width: 900px; }
            textarea { height: 220px; }
            .row { margin-bottom: 12px; }
            pre { background: #f6f8fa; padding: 12px; border-radius: 8px; overflow: auto; }
            button { padding: 8px 14px; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
            .card { border: 1px solid #e5e7eb; padding: 12px; border-radius: 8px; }
            .muted { color: #6b7280; font-size: 12px; }
        </style>
    </head>
    <body>
        <h1>mahjong-py</h1>
        <div class=\"grid\">
            <div class=\"card\">
                <h2>/calc</h2>
                <p class=\"muted\">Shanten + expected-score placeholder</p>
                <textarea id=\"req_calc\">{
    \"enable_reddora\": true,
    \"enable_uradora\": false,
    \"enable_shanten_down\": true,
    \"enable_tegawari\": true,
    \"enable_riichi\": false,
    \"round_wind\": 27,
    \"dora_indicators\": [],
    \"hand\": [0,0,0,1,1,1,2,3,4,18,18,18,28,28],
    \"melds\": [],
    \"seat_wind\": 27,
    \"version\": \"0.1.0\"
}</textarea>
                <div class=\"row\"><button id=\"send_calc\">Send /calc</button></div>
                <pre id=\"res_calc\"></pre>
            </div>

            <div class=\"card\">
                <h2>/score</h2>
                <p class=\"muted\">Score calculation + yaku breakdown</p>
                <textarea id=\"req_score\">{
    \"hand\": [19,19,19,20,20,20,21,21,21,22,22,23,23],
    \"melds\": [],
    \"seat_wind\": 28,
    \"round_wind\": 27,
    \"dora_indicators\": [],
    \"uradora_indicators\": [],
    \"win_tile\": 22,
    \"win_flag\": 16,
    \"enable_reddora\": true
}</textarea>
                <div class=\"row\"><button id=\"send_score\">Send /score</button></div>
                <pre id=\"res_score\"></pre>
            </div>
        </div>
        <script>
            async function send(endpoint, reqId, resId) {
                const reqEl = document.getElementById(reqId);
                const resEl = document.getElementById(resId);
                try {
                    const payload = JSON.parse(reqEl.value);
                    const res = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();
                    resEl.textContent = JSON.stringify(data, null, 2);
                } catch (err) {
                    resEl.textContent = String(err);
                }
            }

            document.getElementById('send_calc').addEventListener('click', () => {
                send('/calc', 'req_calc', 'res_calc');
            });
            document.getElementById('send_score').addEventListener('click', () => {
                send('/score', 'req_score', 'res_score');
            });
        </script>
    </body>
</html>
"""


@app.post("/calc")
def calc(payload: Dict[str, Any]) -> Dict[str, Any]:
    """计算向听/进张/无效牌，并在可舍牌时枚举有效牌数量。"""
    try:
        req = _parse_request(payload)
        player = _build_player(req)
        round_ = _build_round(req)

        wall = req.wall if req.wall is not None else _create_wall(round_, player, req.enable_reddora)
        _validate_wall(wall)

        num_tiles = player.num_tiles() + player.num_melds() * 3
        if num_tiles % 3 == 0 or num_tiles > 14:
            raise ValueError("Invalid number of tiles.")

        shanten_all = shanten(player.hand[:34], player.num_melds(), ShantenFlag.All)[1]
        shanten_regular = shanten(player.hand[:34], player.num_melds(), ShantenFlag.Regular)[1]
        shanten_seven = shanten(player.hand[:34], player.num_melds(), ShantenFlag.SevenPairs)[1]
        shanten_orphans = shanten(player.hand[:34], player.num_melds(), ShantenFlag.ThirteenOrphans)[1]

        if shanten_all == -1:
            raise ValueError("手牌はすでに和了形です。")

        _, _, necessary = necessary_tiles(player.hand[:34], player.num_melds(), ShantenFlag.All)
        _, _, unnecessary = unnecessary_tiles(player.hand[:34], player.num_melds(), ShantenFlag.All)

        config = ExpectedScoreConfig(
            t_min=1,
            t_max=18,
            sum=sum(wall[:34]),
            extra=1,
            shanten_type=ShantenFlag.All,
            calc_stats=shanten_all <= 3,
            enable_reddora=req.enable_reddora,
            enable_uradora=req.enable_uradora,
            enable_shanten_down=req.enable_shanten_down,
            enable_tegawari=req.enable_tegawari,
            enable_riichi=req.enable_riichi,
            use_numba=req.use_numba,
        )
        if shanten_all == 0:
            config.enable_shanten_down = False
            config.enable_tegawari = False

        start_ns = time.perf_counter_ns()
        if config.use_numba and not player.melds:
            stats, searched = ExpectedScoreNumbaCalculator.calc(config, round_, player, wall)
        else:
            stats, searched = ExpectedScoreCalculator.calc(config, round_, player, wall)
        elapsed_us = (time.perf_counter_ns() - start_ns) // 1000

        discard_options = []
        if num_tiles % 3 == 2:
            hand_list = [int(x) for x in player.hand]
            for tile in range(37):
                if hand_list[tile] <= 0:
                    continue

                next_hand = _apply_discard(hand_list, tile)
                shanten_after = shanten(next_hand[:34], player.num_melds(), ShantenFlag.All)[1]
                _, _, next_necessary = necessary_tiles(next_hand[:34], player.num_melds(), ShantenFlag.All)

                wall_after = _wall_after_discard(wall, tile)
                eff_tiles = []
                eff_count = 0
                for t in next_necessary:
                    count = wall_after[t]
                    eff_tiles.append({"tile": t, "count": count})
                    eff_count += count

                discard_options.append(
                    {
                        "discard_tile": tile,
                        "shanten": shanten_after,
                        "necessary_tiles": eff_tiles,
                        "effective_count": eff_count,
                    }
                )

            discard_options.sort(key=lambda x: (-x["effective_count"], x["discard_tile"]))

        response = {
            "shanten": {
                "all": shanten_all,
                "regular": shanten_regular,
                "seven_pairs": shanten_seven,
                "thirteen_orphans": shanten_orphans,
            },
            "necessary_tiles": necessary,
            "unnecessary_tiles": unnecessary,
            "discard_options": discard_options,
            "stats": _dump_expected_stats(stats),
            "searched": searched,
            "time": elapsed_us,
            "config": {
                "t_min": config.t_min,
                "t_max": config.t_max,
                "sum": config.sum,
                "extra": config.extra,
                "shanten_type": config.shanten_type,
                "calc_stats": config.calc_stats,
                "use_numba": config.use_numba,
                "num_tiles": num_tiles,
            },
        }

        return {
            "success": True,
            "request": payload,
            "response": response,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "request": payload,
            "err_msg": str(exc),
        }


@app.post("/score")
def score(payload: Dict[str, Any]) -> Dict[str, Any]:
    """点数计算接口：返回番/符/点数与役种明细。"""
    try:
        req = ScoreRequestIn(**payload)

        tiles = req.hand if req.hand is not None else req.hand_tiles
        if tiles is None:
            raise ValueError("hand is required")
        _validate_tiles(tiles)

        tiles = list(tiles)
        if len(tiles) % 3 == 1:
            tiles.append(req.win_tile)

        melds = [Meld(type=m.type, tiles=m.tiles) for m in req.melds]
        player = Player(hand=hand_from_tiles(tiles), melds=melds, wind=req.seat_wind)

        round_ = Round()
        round_.rules = 0
        round_.rules |= RuleFlag.RedDora if req.enable_reddora else 0
        round_.rules |= RuleFlag.OpenTanyao
        round_.wind = req.round_wind
        round_.dora_indicators = list(req.dora_indicators)
        round_.uradora_indicators = list(req.uradora_indicators)

        result = ScoreCalculator.calc(round_, player, req.win_tile, req.win_flag)

        return {
            "success": result.success,
            "request": payload,
            "response": {
                "err_msg": result.err_msg,
                "han": result.han,
                "fu": result.fu,
                "score_title": result.score_title,
                "score": result.score,
                "yaku_list": [
                    {"yaku": y, "name": YakuName.get(y, str(y)), "han": h}
                    for y, h in result.yaku_list
                ],
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "request": payload,
            "err_msg": str(exc),
        }
