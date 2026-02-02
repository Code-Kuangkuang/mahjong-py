"""Maliang GUI for mahjong_py.

Provides a clickable tile UI to build a hand and compute score/shanten.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk

import numpy as np

from maliang.core import containers
from maliang.toolbox import enhanced
from maliang.standard import dialogs, widgets

from ..constants import Tile, tile_id_to_zh
from ..utils import to_no_reddora
from ..mpsz import to_mpsz
from ..score_calculator import ScoreCalculator
from ..score_constants import WinFlag, YakuNameZh
from ..score_types import Player, Round
from ..shanten import ShantenFlag, shanten
from ..necessary import necessary_tiles
from ..unnecessary import unnecessary_tiles
from .helpers import UiState, _join_list
from .layout import LayoutMixin
from .result_panel import ResultPanelMixin


class MahjongGui(LayoutMixin, ResultPanelMixin):
    def __init__(self) -> None:
        self.state = UiState()
        self.hand_tiles: list[int] = []  # store tile ids 0..36 (reds are 34..36)
        self.counts = np.zeros(37, dtype=np.int16)
        self.win_tile: int | None = None
        self.dora_tiles: list[int] = []
        self.uradora_tiles: list[int] = []
        self.tile_image_size = (50, 70)
        self.result_tile_size = (32, 44)
        self.img_cache: dict[int, enhanced.PhotoImage] = {}
        self.img_cache_small: dict[int, enhanced.PhotoImage] = {}
        self.hand_slot_items: list[int] = []
        self.hand_slot_rects: list[int] = []
        self.hand_slot_origin = (150, 360)
        self.hand_slot_size = self.tile_image_size
        self.hand_slot_gap = 8
        self.result_origin = (620, 470)
        self.result_size = (540, 380)
        self.result_items: list[int] = []
        self.palette_buttons: list[widgets.Button] = []
        self.palette_layout: list[tuple[widgets.Button, int, int]] = []
        self.palette_label: widgets.Label | None = None
        self.seat_wind_buttons: list[widgets.Button] = []
        self.round_wind_buttons: list[widgets.Button] = []

        self.tk = containers.Tk(size=(1400, 900), title="Mahjong GUI")
        self.cv = containers.Canvas(self.tk)
        self.cv.place(x=0, y=0, width=1380, height=880)

        self.v_scroll = tk.Scrollbar(self.tk, orient="vertical", command=self.cv.yview)
        self.h_scroll = tk.Scrollbar(self.tk, orient="horizontal", command=self.cv.xview)
        self.v_scroll.place(x=1380, y=0, width=20, height=880)
        self.h_scroll.place(x=0, y=880, width=1380, height=20)
        self.cv.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.cv.configure(scrollregion=(0, 0, 1900, 1400))

        self.cv.bind_all("<MouseWheel>", self._on_mousewheel)
        self.cv.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.cv.bind_all("<Button-5>", self._on_mousewheel_linux)

        self.tk.bind("<Configure>", self._on_resize)

        self._load_tile_images()
        self._build_layout()
        self._refresh_display()

    def _fit_image(self, img: enhanced.PhotoImage, size: tuple[int, int]) -> enhanced.PhotoImage:
        try:
            w, h = img.width(), img.height()
        except Exception:
            return img
        if not w or not h:
            return img

        target_w, target_h = size
        if w == target_w and h == target_h:
            return img

        try:
            scale_x = target_w / w
            scale_y = target_h / h
            if hasattr(img, "scale") and callable(getattr(img, "scale")):
                return img.scale(scale_x, scale_y)
        except Exception:
            return img
        return img

    def _load_tile_images(self) -> None:
        base = None
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "photo" / "img"
            if candidate.exists():
                base = candidate
                break
        if base is None:
            base = Path(__file__).resolve().parents[4] / "photo" / "img"

        def load(tid: int, rel: str) -> None:
            path = base / rel
            if not path.exists():
                return
            try:
                img = enhanced.PhotoImage(file=str(path), master=self.tk)
                img_big = self._fit_image(img, self.tile_image_size)
                self.img_cache[tid] = img_big
                img_small = self._fit_image(img, self.result_tile_size)
                self.img_cache_small[tid] = img_small
            except Exception:
                return

        for i in range(1, 10):
            load(Tile.Manzu1 + (i - 1), f"manzu/{i}m.png")
            load(Tile.Pinzu1 + (i - 1), f"pinzu/{i}p.png")
            load(Tile.Souzu1 + (i - 1), f"souzu/{i}s.png")

        load(Tile.RedManzu5, "manzu/0m.png")
        load(Tile.RedPinzu5, "pinzu/0p.png")
        load(Tile.RedSouzu5, "souzu/0s.png")

        load(Tile.East, "honors/east.png")
        load(Tile.South, "honors/south.png")
        load(Tile.West, "honors/west.png")
        load(Tile.North, "honors/north.png")
        load(Tile.White, "honors/white.png")
        load(Tile.Green, "honors/green.png")
        load(Tile.Red, "honors/red.png")

    def _set_mode(self, mode: str) -> None:
        self.state.mode = mode
        if mode == "shanten" and self.state.target == "win":
            self.state.target = "hand"
        self._refresh_display()

    def _set_target(self, target: str) -> None:
        self.state.target = target
        self._refresh_display()

    def _set_visible(self, widget, visible: bool) -> None:
        try:
            widget.forget(not visible)
        except Exception:
            pass

    def _on_tsumo(self, value: bool) -> None:
        if value:
            if self.chk_ron.get():
                self.chk_ron.set(False)
        else:
            if not self.chk_ron.get():
                self.chk_ron.set(True)
        self._refresh_display()

    def _on_ron(self, value: bool) -> None:
        if value:
            if self.chk_tsumo.get():
                self.chk_tsumo.set(False)
        else:
            if not self.chk_tsumo.get():
                self.chk_tsumo.set(True)
        self._refresh_display()

    def _set_seat_wind(self, tile_id: int) -> None:
        self.state.seat_wind = tile_id
        self.lbl_seat.set(tile_id_to_zh(tile_id))

    def _set_round_wind(self, tile_id: int) -> None:
        self.state.round_wind = tile_id
        self.lbl_round.set(tile_id_to_zh(tile_id))

    def _clear_win(self) -> None:
        self.win_tile = None
        if self.state.target == "win":
            self.state.target = "hand"
        self._refresh_display()

    def _clear_dora(self) -> None:
        self.dora_tiles.clear()
        if self.state.target == "dora":
            self.state.target = "hand"
        self._refresh_display()

    def _clear_uradora(self) -> None:
        self.uradora_tiles.clear()
        if self.state.target == "uradora":
            self.state.target = "hand"
        self._refresh_display()

    def _add_tile(self, tile_id: int) -> None:
        if self.state.target == "win":
            self.win_tile = tile_id
            self.state.target = "hand"
            self._refresh_display()
            return

        if self.state.target == "dora":
            if len(self.dora_tiles) >= 5:
                dialogs.TkMessage(message="宝牌指示牌最多 5 张。", icon="warning", master=self.tk)
                return
            self.dora_tiles.append(tile_id)
            self._refresh_display()
            return

        if self.state.target == "uradora":
            if len(self.uradora_tiles) >= 5:
                dialogs.TkMessage(message="里宝指示牌最多 5 张。", icon="warning", master=self.tk)
                return
            self.uradora_tiles.append(tile_id)
            self._refresh_display()
            return

        # limit: 14 tiles in hand
        if int(self.counts[:34].sum()) >= 14:
            dialogs.TkMessage(message="手牌最多 14 张。", icon="warning", master=self.tk)
            return

        base_id = tile_id
        if tile_id in (Tile.RedManzu5, Tile.RedPinzu5, Tile.RedSouzu5):
            # only one red 5 each suit
            if self.counts[tile_id] >= 1:
                dialogs.TkMessage(message="赤5每种最多 1 张。", icon="warning", master=self.tk)
                return
            if tile_id == Tile.RedManzu5:
                base_id = Tile.Manzu5
            elif tile_id == Tile.RedPinzu5:
                base_id = Tile.Pinzu5
            else:
                base_id = Tile.Souzu5

        if self.counts[base_id] >= 4:
            dialogs.TkMessage(message="同一张牌最多 4 张。", icon="warning", master=self.tk)
            return

        self.hand_tiles.append(tile_id)
        if tile_id in (Tile.RedManzu5, Tile.RedPinzu5, Tile.RedSouzu5):
            self.counts[tile_id] += 1
            self.counts[base_id] += 1
        else:
            self.counts[tile_id] += 1

        self._refresh_display()

    def _undo(self) -> None:
        if not self.hand_tiles:
            return
        self._remove_at(len(self.hand_tiles) - 1)
        self._refresh_display()

    def _clear(self) -> None:
        self.hand_tiles.clear()
        self.counts[:] = 0
        self.win_tile = None
        self.dora_tiles.clear()
        self.uradora_tiles.clear()
        self._refresh_display()

    def _remove_at(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.hand_tiles):
            return
        tile_id = self.hand_tiles.pop(idx)
        if tile_id in (Tile.RedManzu5, Tile.RedPinzu5, Tile.RedSouzu5):
            base_id = Tile.Manzu5 if tile_id == Tile.RedManzu5 else Tile.Pinzu5 if tile_id == Tile.RedPinzu5 else Tile.Souzu5
            self.counts[tile_id] -= 1
            self.counts[base_id] -= 1
        else:
            self.counts[tile_id] -= 1
        self._refresh_display()

    def _refresh_display(self) -> None:
        # mode buttons
        if self.state.mode == "score":
            self.btn_mode_score.set("计分✅")
            self.btn_mode_shanten.set("向听")
        else:
            self.btn_mode_score.set("计分")
            self.btn_mode_shanten.set("向听✅")

        # target buttons
        if self.state.target == "hand":
            self.btn_target_hand.set("手牌✅")
            self.btn_target_win.set("和牌")
        elif self.state.target == "win":
            self.btn_target_hand.set("手牌")
            self.btn_target_win.set("和牌✅")
        elif self.state.target == "dora":
            self.btn_target_hand.set("手牌")
            self.btn_target_win.set("和牌")
            self.btn_dora_target.set("设置✅")
            self.btn_uradora_target.set("设置")
        elif self.state.target == "uradora":
            self.btn_target_hand.set("手牌")
            self.btn_target_win.set("和牌")
            self.btn_dora_target.set("设置")
            self.btn_uradora_target.set("设置✅")
        else:
            self.btn_target_hand.set("手牌")
            self.btn_target_win.set("和牌")

        if self.state.target not in ("dora", "uradora"):
            self.btn_dora_target.set("设置")
            self.btn_uradora_target.set("设置")

        hand_mpsz = to_mpsz(self.counts)
        self.lbl_hand.set(f"{hand_mpsz}  （{int(self.counts[:34].sum())} 张）")

        # hand zh line is hidden in UI
        zh = [tile_id_to_zh(t) for t in self.hand_tiles]
        self.lbl_hand_zh.set(_join_list(zh))

        # update hand slot images
        for i in range(14):
            if i < len(self.hand_tiles):
                tid = self.hand_tiles[i]
                img = self.img_cache.get(tid)
                self.cv.itemconfigure(self.hand_slot_items[i], image=img if img is not None else "")
                self.cv.itemconfigure(self.hand_slot_rects[i], outline="#4CAF50")
            else:
                self.cv.itemconfigure(self.hand_slot_items[i], image="")
                self.cv.itemconfigure(self.hand_slot_rects[i], outline="#cfcfcf")

        if self.win_tile is None:
            self.lbl_win.set("（未选）")
        else:
            self.lbl_win.set(tile_id_to_zh(self.win_tile))

        # dora/uradora display
        dora_zh = [tile_id_to_zh(t) for t in self.dora_tiles]
        uradora_zh = [tile_id_to_zh(t) for t in self.uradora_tiles]
        self.lbl_dora.set(_join_list(dora_zh))
        self.lbl_uradora.set(_join_list(uradora_zh))

        # show/hide by mode
        is_score = self.state.mode == "score"
        self._set_visible(self.btn_target_win, is_score)
        self._set_visible(self.btn_target_hand, not is_score)

        # hide winds in shanten mode
        self._set_visible(self.lbl_seat_title, is_score)
        self._set_visible(self.lbl_seat, is_score)
        self._set_visible(self.lbl_round_title, is_score)
        self._set_visible(self.lbl_round, is_score)
        for btn in self.seat_wind_buttons:
            self._set_visible(btn, is_score)
        for btn in self.round_wind_buttons:
            self._set_visible(btn, is_score)

        self._set_visible(self.lbl_flags1, is_score)
        self._set_visible(self.chk_tsumo, is_score)
        self._set_visible(self.txt_tsumo, is_score)
        self._set_visible(self.chk_ron, is_score)
        self._set_visible(self.txt_ron, is_score)

        self._set_visible(self.chk_riichi, is_score)
        self._set_visible(self.txt_riichi, is_score)
        self._set_visible(self.chk_driichi, is_score)
        self._set_visible(self.txt_driichi, is_score)
        self._set_visible(self.chk_ippatsu, is_score)
        self._set_visible(self.txt_ippatsu, is_score)

        self._set_visible(self.lbl_dora_title, is_score)
        self._set_visible(self.btn_dora_target, is_score)
        self._set_visible(self.btn_dora_clear, is_score)
        self._set_visible(self.lbl_dora, is_score)
        self._set_visible(self.lbl_uradora_title, is_score)
        self._set_visible(self.btn_uradora_target, is_score)
        self._set_visible(self.btn_uradora_clear, is_score)
        self._set_visible(self.lbl_uradora, is_score)

        self._set_visible(self.lbl_win_title, is_score)
        self._set_visible(self.btn_win_target, is_score)
        self._set_visible(self.btn_win_clear, is_score)
        self._set_visible(self.lbl_win, is_score)
        self._set_visible(self.lbl_hand_zh, False)

        # conditional flags based on tsumo/ron
        tsumo = self.chk_tsumo.get()
        ron = self.chk_ron.get()

        show_haitei = is_score and tsumo
        show_houtei = is_score and ron
        show_robbing = is_score and ron

        self._set_visible(self.chk_haitei, show_haitei)
        self._set_visible(self.txt_haitei, show_haitei)
        self._set_visible(self.chk_houtei, show_houtei)
        self._set_visible(self.txt_houtei, show_houtei)
        self._set_visible(self.chk_robbing, show_robbing)
        self._set_visible(self.txt_robbing, show_robbing)

        self._set_visible(self.chk_rinshan, is_score)
        self._set_visible(self.txt_rinshan, is_score)

        # reflow vertical layout (avoid blank gaps)
        base_y = 95 if is_score else 95
        row3_y = 95 if is_score else 95
        row4_y = 140 if is_score else 95
        row5_y = 180 if is_score else 120
        row6_y = 225 if is_score else 120

        if is_score:
            # Row 3 (tsumo/ron)
            self._moveto_widget(self.lbl_flags1, 20, row3_y)
            self._moveto_widget(self.chk_tsumo, 120, row3_y + 5)
            self._moveto_widget(self.txt_tsumo, 150, row3_y + 3)
            self._moveto_widget(self.chk_ron, 220, row3_y + 5)
            self._moveto_widget(self.txt_ron, 250, row3_y + 3)

            # Row 4 (detail flags)
            self._moveto_widget(self.chk_riichi, 120, row4_y + 5)
            self._moveto_widget(self.txt_riichi, 150, row4_y + 3)
            self._moveto_widget(self.chk_driichi, 220, row4_y + 5)
            self._moveto_widget(self.txt_driichi, 250, row4_y + 3)
            self._moveto_widget(self.chk_ippatsu, 320, row4_y + 5)
            self._moveto_widget(self.txt_ippatsu, 350, row4_y + 3)

            self._moveto_widget(self.chk_haitei, 420, row4_y + 5)
            self._moveto_widget(self.txt_haitei, 450, row4_y + 3)
            self._moveto_widget(self.chk_houtei, 540, row4_y + 5)
            self._moveto_widget(self.txt_houtei, 570, row4_y + 3)
            self._moveto_widget(self.chk_rinshan, 660, row4_y + 5)
            self._moveto_widget(self.txt_rinshan, 690, row4_y + 3)
            self._moveto_widget(self.chk_robbing, 780, row4_y + 5)
            self._moveto_widget(self.txt_robbing, 810, row4_y + 3)

            # Row 5 (dora)
            self._moveto_widget(self.lbl_dora_title, 20, row5_y)
            self._moveto_widget(self.btn_dora_target, 140, row5_y)
            self._moveto_widget(self.btn_dora_clear, 240, row5_y)
            self._moveto_widget(self.lbl_dora, 310, row5_y)
            self._moveto_widget(self.lbl_uradora_title, 640, row5_y)
            self._moveto_widget(self.btn_uradora_target, 760, row5_y)
            self._moveto_widget(self.btn_uradora_clear, 860, row5_y)
            self._moveto_widget(self.lbl_uradora, 930, row5_y)

            # Row 6 (actions)
            self._moveto_widget(self.btn_calc, 20, row6_y)
            self._moveto_widget(self.btn_undo, 120, row6_y)
            self._moveto_widget(self.btn_clear, 220, row6_y)
        else:
            # Actions move up
            self._moveto_widget(self.btn_calc, 20, row6_y)
            self._moveto_widget(self.btn_undo, 120, row6_y)
            self._moveto_widget(self.btn_clear, 220, row6_y)

        # Hand section and result/palette positions
        hand_y = 260 if is_score else 200
        self._moveto_widget(self.lbl_hand, 150, hand_y)
        self._moveto_widget(self.lbl_hand_zh, 150, hand_y + 35)
        self._move_hand_slots(hand_y + 60)
        self._moveto_widget(self.lbl_hand, 150, hand_y)

        if is_score:
            self._moveto_widget(self.lbl_win_title, 680, hand_y)
            self._moveto_widget(self.btn_win_target, 760, hand_y)
            self._moveto_widget(self.btn_win_clear, 830, hand_y)
            self._moveto_widget(self.lbl_win, 900, hand_y)

        palette_y = 520 if is_score else 430
        if self.palette_label is not None:
            self._moveto_widget(self.palette_label, 20, palette_y)
        for btn, row, col in self.palette_layout:
            x = 20 + col * (self.tile_image_size[0] + 8)
            y = (palette_y + 40) + row * (self.tile_image_size[1] + 8)
            self._moveto_widget(btn, x, y)

        result_y = palette_y
        self._move_result_panel(self.result_origin[0], result_y)

    def _calculate(self) -> None:
        if self.state.mode == "shanten":
            self._calc_shanten()
        else:
            self._calc_score()

    def _on_resize(self, _event) -> None:
        try:
            w = self.tk.winfo_width()
            h = self.tk.winfo_height()
        except Exception:
            return

        canvas_w = max(200, w - 20)
        canvas_h = max(200, h - 20)
        self.cv.place(x=0, y=0, width=canvas_w, height=canvas_h)

        self.v_scroll.place(x=canvas_w, y=0, width=20, height=canvas_h)
        self.h_scroll.place(x=0, y=canvas_h, width=canvas_w, height=20)

    def _on_mousewheel(self, event) -> None:
        # Windows uses event.delta; positive means scroll up
        if event.delta == 0:
            return
        self.cv.yview_scroll(int(-event.delta / 120), "units")

    def _on_mousewheel_linux(self, event) -> None:
        # Linux uses Button-4 (up) and Button-5 (down)
        if event.num == 4:
            self.cv.yview_scroll(-1, "units")
        elif event.num == 5:
            self.cv.yview_scroll(1, "units")

    def _calc_shanten(self) -> None:
        total_tiles = int(self.counts[:34].sum())
        if total_tiles not in (13, 14):
            dialogs.TkMessage(message="向听需要 13 张或 14 张手牌。", icon="warning", master=self.tk)
            return

        hand34 = self.counts[:34].copy()
        best_type, best = shanten(hand34, 0, ShantenFlag.All)
        types = []
        if best_type & ShantenFlag.Regular:
            types.append("一般形")
        if best_type & ShantenFlag.SevenPairs:
            types.append("七对子")
        if best_type & ShantenFlag.ThirteenOrphans:
            types.append("国士无双")

        if total_tiles == 13:
            _, best_s, tiles_need = necessary_tiles(hand34, 0, ShantenFlag.All)
            self._render_shanten_result(best=best, types=types, tiles_need=tiles_need)
        else:
            _, best_s, discards = unnecessary_tiles(hand34, 0, ShantenFlag.All)
            discard_rows: list[tuple[int, list[int], int]] = []
            for t in discards:
                if hand34[t] <= 0:
                    continue
                tmp = hand34.copy()
                tmp[t] -= 1
                _, s2, needs = necessary_tiles(tmp, 0, ShantenFlag.All)

                total_waits = 0
                for w in needs:
                    total_waits += max(0, 4 - int(tmp[w]))
                discard_rows.append((t, needs, total_waits))

            self._render_shanten_result(best=best, types=types, discards=discard_rows)

    def _calc_score(self) -> None:
        if self.win_tile is None:
            dialogs.TkMessage(message="请先选择和牌。", icon="warning", master=self.tk)
            return

        total_tiles = int(self.counts[:34].sum())
        if total_tiles not in (13, 14):
            dialogs.TkMessage(message="计分需要 13 张或 14 张手牌。", icon="warning", master=self.tk)
            return

        hand_arr = self.counts.copy()
        if total_tiles == 13:
            # add win tile
            win_id = self.win_tile
            if win_id in (Tile.RedManzu5, Tile.RedPinzu5, Tile.RedSouzu5):
                base_id = Tile.Manzu5 if win_id == Tile.RedManzu5 else Tile.Pinzu5 if win_id == Tile.RedPinzu5 else Tile.Souzu5
                hand_arr[win_id] += 1
                hand_arr[base_id] += 1
            else:
                hand_arr[win_id] += 1

        round_ = Round()
        round_.wind = self.state.round_wind
        if self.dora_tiles:
            round_.set_dora_indicators([int(to_no_reddora(t)) for t in self.dora_tiles])
        if self.uradora_tiles:
            round_.set_uradora_indicators([int(to_no_reddora(t)) for t in self.uradora_tiles])
        player = Player(hand=hand_arr, wind=self.state.seat_wind, melds=[])

        flags = 0
        if self.chk_tsumo.get():
            flags |= WinFlag.Tsumo
        # ron is implicit when not tsumo
        if self.chk_riichi.get():
            flags |= WinFlag.Riichi
        if self.chk_driichi.get():
            flags |= WinFlag.DoubleRiichi
        if self.chk_ippatsu.get():
            flags |= WinFlag.Ippatsu
        if self.chk_haitei.get():
            flags |= WinFlag.UnderTheSea
        if self.chk_houtei.get():
            flags |= WinFlag.UnderTheRiver
        if self.chk_rinshan.get():
            flags |= WinFlag.AfterAKong
        if self.chk_robbing.get():
            flags |= WinFlag.RobbingAKong

        res = ScoreCalculator.calc(round_, player, self.win_tile, flags)
        if not res.success:
            self._render_score_result([f"计算失败：{res.err_msg}"])
            return

        yaku_lines = []
        for y, h in res.yaku_list:
            name = YakuNameZh.get(y, str(y))
            yaku_lines.append(f"- {name} ×{h}翻")

        lines = [
            f"番：{res.han}  符：{res.fu}",
            f"点数：{res.score}",
            "役种：" + ("\n".join(yaku_lines) if yaku_lines else "（无）"),
        ]

        self._render_score_result(lines)

    def run(self) -> None:
        self.tk.center()
        self.tk.mainloop()


def main() -> None:
    app = MahjongGui()
    app.run()


if __name__ == "__main__":
    main()
