"""Result panel rendering mixin for MahjongGui."""

from __future__ import annotations

from .helpers import _join_list


class ResultPanelMixin:
    def _clear_result_items(self) -> None:
        for item in self.result_items:
            try:
                self.cv.delete(item)
            except Exception:
                pass
        self.result_items.clear()

    def _draw_result_text(self, x: int, y: int, text: str, *, size: int = 14) -> int:
        item = self.cv.create_text(
            x, y,
            text=text,
            anchor="nw",
            font=("Microsoft YaHei", size),
        )
        self.result_items.append(item)
        return item

    def _draw_tile_row(self, x: int, y: int, tiles: list[int], *, max_w: int) -> int:
        cur_x, cur_y = x, y
        w, h = self.result_tile_size
        gap = 8
        for t in tiles:
            img = self.img_cache_small.get(t)
            if img is None:
                continue
            if cur_x + w > x + max_w:
                cur_x = x
                cur_y += h + gap
            item = self.cv.create_image(cur_x + w / 2, cur_y + h / 2, image=img)
            self.result_items.append(item)
            cur_x += w + gap
        return cur_y + h + gap + 6

    def _render_shanten_result(
        self,
        *,
        best: int,
        types: list[str],
        tiles_need: list[int] | None = None,
        discards: list[tuple[int, list[int], int]] | None = None,
    ) -> None:
        self._clear_result_items()
        x0, y0 = self.result_origin[0], self.result_origin[1] + 40
        max_w = self.result_size[0] - 20

        y = y0
        self._draw_result_text(x0 + 10, y, f"向听数：{best}")
        y += 30
        self._draw_result_text(x0 + 10, y, f"最优类型：{_join_list(types)}")
        y += 34

        if tiles_need is not None:
            self._draw_result_text(x0 + 10, y, "进张：")
            y += 26
            y = self._draw_tile_row(x0 + 10, y, tiles_need, max_w=max_w)
            if best == 0:
                self._draw_result_text(x0 + 10, y, "听牌：")
                y += 26
                y = self._draw_tile_row(x0 + 10, y, tiles_need, max_w=max_w)

        if discards:
            self._draw_result_text(x0 + 10, y, "切牌后的进张：")
            y += 30

            left_x = x0 + 10
            right_x = x0 + 120
            count_x = x0 + 440
            right_w = max_w - (right_x - (x0 + 10))

            self._draw_result_text(left_x, y, "切牌")
            self._draw_result_text(right_x, y, "进张")
            self._draw_result_text(count_x, y, "总数")
            y += 26

            for d_tile, waits, total_waits in discards:
                row_start = y
                img = self.img_cache_small.get(d_tile)
                if img is not None:
                    item = self.cv.create_image(left_x + 18, row_start + self.result_tile_size[1] / 2, image=img)
                    self.result_items.append(item)
                self._draw_result_text(count_x, row_start, f"{total_waits}")

                y = self._draw_tile_row(right_x, row_start + 2, waits, max_w=right_w)

                min_y = row_start + self.result_tile_size[1] + 26
                if y < min_y:
                    y = min_y
                y += 12

        # resize result panel height to fit content
        self._set_result_panel_height(max(120, y - y0 + 10))
        self._fit_result_panel_to_items()

    def _render_score_result(self, lines: list[str]) -> None:
        self._clear_result_items()
        x0, y0 = self.result_origin[0], self.result_origin[1] + 40
        y = y0
        for line in lines:
            self._draw_result_text(x0 + 10, y, line)
            y += 26
        self._set_result_panel_height(max(120, y - y0 + 10))
        self._fit_result_panel_to_items()

    def _set_result_panel_height(self, height: int) -> None:
        x0 = self.result_origin[0]
        y0 = self.result_origin[1] + 35
        self.result_size = (self.result_size[0], height)
        self.cv.coords(
            self.result_panel,
            x0,
            y0,
            x0 + self.result_size[0],
            y0 + height,
        )

    def _fit_result_panel_to_items(self) -> None:
        if not self.result_items:
            return
        try:
            bbox = self.cv.bbox(*self.result_items)
        except Exception:
            return
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        x0 = self.result_origin[0]
        y0 = self.result_origin[1] + 35
        pad = 16
        new_w = max(self.result_size[0], int((x2 - x0) + pad))
        new_h = max(self.result_size[1], int((y2 - y0) + pad))
        if new_w != self.result_size[0] or new_h != self.result_size[1]:
            self.result_size = (new_w, new_h)
            self.cv.coords(
                self.result_panel,
                x0,
                y0,
                x0 + new_w,
                y0 + new_h,
            )
