"""GUI layout mixin for MahjongGui."""

from __future__ import annotations

from typing import Callable

from maliang.standard import widgets

from ..constants import Tile


class LayoutMixin:
    def _build_layout(self) -> None:
        # Title
        widgets.Text(self.cv, (20, 15), text="麻将计算器 GUI（Maliang）", fontsize=20, weight="bold")

        # Row 1: Mode
        self.lbl_mode = widgets.Label(self.cv, (20, 55), size=(90, 28), text="模式")
        self.btn_mode_score = widgets.Button(
            self.cv, (120, 55), size=(80, 28), text="计分",
            command=lambda: self._set_mode("score"),
        )
        self.btn_mode_shanten = widgets.Button(
            self.cv, (210, 55), size=(80, 28), text="向听",
            command=lambda: self._set_mode("shanten"),
        )

        # Row 2: Target (only show relevant one per mode)
        self.lbl_target = widgets.Label(self.cv, (320, 55), size=(110, 28), text="点击目标")
        self.btn_target_hand = widgets.Button(
            self.cv, (440, 55), size=(100, 28), text="手牌",
            command=lambda: self._set_target("hand"),
        )
        self.btn_target_win = widgets.Button(
            self.cv, (550, 55), size=(100, 28), text="和牌",
            command=lambda: self._set_target("win"),
        )

        # Wind selectors (always visible)
        self.lbl_seat_title = widgets.Label(self.cv, (680, 55), size=(80, 28), text="自风")
        self.lbl_seat = widgets.Label(self.cv, (760, 55), size=(60, 28), text="东")
        self.seat_wind_buttons = self._build_wind_buttons(830, 55, setter=self._set_seat_wind)

        self.lbl_round_title = widgets.Label(self.cv, (680, 90), size=(80, 28), text="场风")
        self.lbl_round = widgets.Label(self.cv, (760, 90), size=(60, 28), text="东")
        self.round_wind_buttons = self._build_wind_buttons(830, 90, setter=self._set_round_wind)

        # Row 3: Flags (tsumo/ron)
        self.lbl_flags1 = widgets.Label(self.cv, (20, 95), size=(90, 28), text="标志")
        self.chk_tsumo = widgets.CheckBox(self.cv, (120, 100), length=24, command=self._on_tsumo)
        self.txt_tsumo = widgets.Text(self.cv, (150, 98), text="自摸")
        self.chk_ron = widgets.CheckBox(self.cv, (220, 100), length=24, command=self._on_ron)
        self.txt_ron = widgets.Text(self.cv, (250, 98), text="荣和")

        # Row 4: Flags (details)
        self.chk_riichi = widgets.CheckBox(self.cv, (120, 140), length=24)
        self.txt_riichi = widgets.Text(self.cv, (150, 138), text="立直")
        self.chk_driichi = widgets.CheckBox(self.cv, (220, 140), length=24)
        self.txt_driichi = widgets.Text(self.cv, (250, 138), text="双立直")
        self.chk_ippatsu = widgets.CheckBox(self.cv, (320, 140), length=24)
        self.txt_ippatsu = widgets.Text(self.cv, (350, 138), text="一发")

        self.chk_haitei = widgets.CheckBox(self.cv, (420, 140), length=24)
        self.txt_haitei = widgets.Text(self.cv, (450, 138), text="海底捞月")
        self.chk_houtei = widgets.CheckBox(self.cv, (540, 140), length=24)
        self.txt_houtei = widgets.Text(self.cv, (570, 138), text="河底捞鱼")
        self.chk_rinshan = widgets.CheckBox(self.cv, (660, 140), length=24)
        self.txt_rinshan = widgets.Text(self.cv, (690, 138), text="岭上开花")
        self.chk_robbing = widgets.CheckBox(self.cv, (780, 140), length=24)
        self.txt_robbing = widgets.Text(self.cv, (810, 138), text="抢杠")

        # Row 5: Dora / Uradora targets
        self.lbl_dora_title = widgets.Label(self.cv, (20, 180), size=(110, 28), text="宝牌指示")
        self.btn_dora_target = widgets.Button(
            self.cv, (140, 180), size=(90, 28), text="设置",
            command=lambda: self._set_target("dora"),
        )
        self.btn_dora_clear = widgets.Button(
            self.cv, (240, 180), size=(60, 28), text="清空",
            command=self._clear_dora,
        )
        self.lbl_dora = widgets.Label(self.cv, (310, 180), size=(310, 28), text="（无）")

        self.lbl_uradora_title = widgets.Label(self.cv, (640, 180), size=(110, 28), text="里宝指示")
        self.btn_uradora_target = widgets.Button(
            self.cv, (760, 180), size=(90, 28), text="设置",
            command=lambda: self._set_target("uradora"),
        )
        self.btn_uradora_clear = widgets.Button(
            self.cv, (860, 180), size=(60, 28), text="清空",
            command=self._clear_uradora,
        )
        self.lbl_uradora = widgets.Label(self.cv, (930, 180), size=(240, 28), text="（无）")

        # Row 6: Actions
        self.btn_calc = widgets.Button(
            self.cv, (20, 225), size=(90, 28), text="计算",
            command=self._calculate,
        )
        self.btn_undo = widgets.Button(
            self.cv, (120, 225), size=(90, 28), text="撤销",
            command=self._undo,
        )
        self.btn_clear = widgets.Button(
            self.cv, (220, 225), size=(90, 28), text="清空",
            command=self._clear,
        )

        # Hand / win tile display
        widgets.Label(self.cv, (20, 300), size=(120, 28), text="门前手牌")
        self.lbl_hand = widgets.Text(self.cv, (150, 300), size=(500, 28), text="")
        self.lbl_hand_zh = widgets.Text(self.cv, (150, 335), size=(500, 28), text="")

        self._build_hand_slots(*self.hand_slot_origin)

        self.lbl_win_title = widgets.Label(self.cv, (680, 300), size=(80, 28), text="和牌")
        self.btn_win_target = widgets.Button(
            self.cv, (760, 300), size=(60, 28), text="设置",
            command=lambda: self._set_target("win"),
        )
        self.btn_win_clear = widgets.Button(
            self.cv, (830, 300), size=(60, 28), text="清空",
            command=self._clear_win,
        )
        self.lbl_win = widgets.Label(self.cv, (900, 300), size=(200, 28), text="（未选）")

        # Tile palette
        self.palette_label = widgets.Label(self.cv, (20, 470), size=(120, 28), text="点击选牌")
        self._build_tile_palette(x=20, y=510)

        # Result panel
        self.lbl_result_title = widgets.Label(self.cv, self.result_origin, size=(120, 28), text="结果")
        self.result_panel = self.cv.create_rectangle(
            self.result_origin[0],
            self.result_origin[1] + 35,
            self.result_origin[0] + self.result_size[0],
            self.result_origin[1] + 35 + self.result_size[1],
            outline="#cfcfcf",
            width=1,
        )

        # default: ron
        self.chk_ron.set(True)

    def _build_wind_buttons(self, x: int, y: int, *, setter: Callable[[int], None]) -> list[widgets.Button]:
        names = ["东", "南", "西", "北"]
        ids = [Tile.East, Tile.South, Tile.West, Tile.North]
        buttons: list[widgets.Button] = []
        for i, (n, tid) in enumerate(zip(names, ids)):
            btn = widgets.Button(
                self.cv, (x + i * 50, y), size=(45, 28), text=n,
                command=lambda t=tid: setter(t),
            )
            buttons.append(btn)
        return buttons

    def _build_tile_palette(self, x: int, y: int) -> None:
        btn_w, btn_h = self.tile_image_size
        gap = 8
        self.palette_buttons.clear()
        self.palette_layout.clear()

        def row(tokens: list[tuple[str, int]], row_idx: int) -> None:
            for col_idx, (label, tid) in enumerate(tokens):
                img = self.img_cache.get(tid)
                btn = widgets.Button(
                    self.cv,
                    (x + col_idx * (btn_w + gap), y + row_idx * (btn_h + gap)),
                    size=(btn_w, btn_h),
                    text=label if img is None else "",
                    image=img,
                    command=lambda t=tid: self._add_tile(t),
                )
                self.palette_buttons.append(btn)
                self.palette_layout.append((btn, row_idx, col_idx))

        man = [(f"{i}m", Tile.Manzu1 + (i - 1)) for i in range(1, 10)]
        pin = [(f"{i}p", Tile.Pinzu1 + (i - 1)) for i in range(1, 10)]
        sou = [(f"{i}s", Tile.Souzu1 + (i - 1)) for i in range(1, 10)]
        hon = [(f"{i}z", Tile.East + (i - 1)) for i in range(1, 8)]
        red = [("0m", Tile.RedManzu5), ("0p", Tile.RedPinzu5), ("0s", Tile.RedSouzu5)]

        row(man, 0)
        row(pin, 1)
        row(sou, 2)
        row(hon, 3)
        row(red, 4)

    def _build_hand_slots(self, x: int, y: int) -> None:
        slot_w, slot_h = self.hand_slot_size
        gap = self.hand_slot_gap
        self.hand_slot_items.clear()
        self.hand_slot_rects.clear()

        for i in range(14):
            sx = x + i * (slot_w + gap)
            rect = self.cv.create_rectangle(
                sx,
                y,
                sx + slot_w,
                y + slot_h,
                outline="#cfcfcf",
                width=1,
            )
            img_item = self.cv.create_image(
                sx + slot_w / 2,
                y + slot_h / 2,
                image="",
                anchor="center",
            )
            tag = f"hand_slot_{i}"
            self.cv.itemconfigure(rect, tags=(tag,))
            self.cv.itemconfigure(img_item, tags=(tag,))
            self.cv.tag_bind(tag, "<Button-1>", lambda e, idx=i: self._remove_at(idx))
            self.hand_slot_rects.append(rect)
            self.hand_slot_items.append(img_item)

    def _moveto_widget(self, widget, x: int, y: int) -> None:
        try:
            widget.moveto(x, y)
            return
        except Exception:
            pass
        try:
            widget.move(x - widget.position[0], y - widget.position[1])
        except Exception:
            pass

    def _move_hand_slots(self, new_y: int) -> None:
        old_x, old_y = self.hand_slot_origin
        dy = new_y - old_y
        if dy == 0:
            return
        for rect in self.hand_slot_rects:
            self.cv.move(rect, 0, dy)
        for item in self.hand_slot_items:
            self.cv.move(item, 0, dy)
        self.hand_slot_origin = (old_x, new_y)

    def _move_result_panel(self, new_x: int, new_y: int) -> None:
        old_x, old_y = self.result_origin
        dx, dy = new_x - old_x, new_y - old_y
        if dx == 0 and dy == 0:
            return
        try:
            self.lbl_result_title.moveto(new_x, new_y)
        except Exception:
            try:
                self.lbl_result_title.move(dx, dy)
            except Exception:
                pass
        x0 = new_x
        y0 = new_y + 35
        self.cv.coords(
            self.result_panel,
            x0,
            y0,
            x0 + self.result_size[0],
            y0 + self.result_size[1],
        )
        for item in self.result_items:
            self.cv.move(item, dx, dy)
        self.result_origin = (new_x, new_y)
