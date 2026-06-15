# 期望值性能优化记录

本节记录对期望值计算的实验优化过程。当前默认服务路径仍是 `src/mahjong_py/expected_score_calculator.py`，`POST /calc` 仍由 `src/mahjong_py/server.py` 调用旧实现；优化版放在独立文件 `src/mahjong_py/expected_score_numba.py`，通过 `tests/test_expected_score_numba.py` 与旧实现做数值一致性对照。

## 有用的路径

- `src/mahjong_py/expected_score_numba.py`：实验版主入口 `ExpectedScoreNumbaCalculator`，没有接入默认 API。当前主要优化都集中在这里。
- `_ArrayGraph` + `_calc_stats_array`：把原来的 Python 对象图改成数组图，并把倒推听牌率、和牌率、点数期望的动态规划下沉到 Numba。
- `_necessary_mask_numba` / `_unnecessary_mask_numba`：把进张、打张候选从 Python list/set 形态改成 bit mask，并复用 `necessary.py`、`unnecessary.py` 的查表热路径。
- `_calc_score_fast_or_none` / `_check_pattern_yaku_fast` / `_evaluate_blocks_fast`：保留原计分规则，但减少 `ScoreCalculator` 中多次重复遍历 block 的成本。
- `_score_title_cached` / `_score_total_cached` / `_get_up_scores_cached`：缓存番符到得点、里宝牌升番得点数组。用户牌例中同一番符组合会反复出现，这条路径有效。
- `reserve_edge`：把"检查边是否存在"和"登记边 key"合并，减少图构建阶段重复构造 `(source, target)` key。
- `_separate_closed_fast` + `_hand_hashes_numba`：闭门手牌专用分解入口，避免 `hand_separator.separate` 每次切片转 list 再 hash 四门。
- `_check_not_pattern_yaku_numba` + `_normal_han_closed_numba`：把非分解役判断和闭门普通役番数累加下沉到 Numba。对当前 4471 节点用户牌例收益最明显。

## 已经证明收益不好的路径

- 对 `_necessary_mask` / `_unnecessary_mask` 加 Python dict 缓存：key 构造成本高，命中率不足，用户牌例反而变慢。
- 对打点评估加 `score_cache = {(hand_counts, shanten_type, win_tile, riichi): score}`：用户牌例实测 1627 次调用、0 次命中，已移除。
- 用 packed big-int 替代 tuple 作为缓存 key：构造 packed key 的成本没有被命中收益抵消，已回退。
- Python 层手写 `_check_not_pattern_yaku_fast_or_none`：虽然少调了一些 `ScoreCalculator` 函数，但仍在解释器里扫 34 张牌，整体比原函数慢。只有 Numba 版本值得保留。
- 盲目加跨节点缓存：当前图搜索中很多状态只访问一次，缓存前必须先测命中率，否则容易只增加 dict 成本。
- bit iteration 替代 range 遍历 37 张牌：`_iter_bits` + `_wall_counts_to_mask` 的 Python while 循环反而比直接 `for tile in range(37)` 更慢。
- 整数编码替代 tuple 作为 hand_counts 缓存 key：`_encode_hand_counts` 用 Python 循环编码 37 张牌比 C 级 `tuple(hand_counts)` 更慢。
- draw/discard 操作内联：替换 `ExpectedScoreCalculator._draw/_discard` 方法调用为内联代码，对性能无明显提升。
- `_distance` 计算内联：使用 generator + sum 替代显式循环，比直接方法调用更慢，已回退。

## 2026-06-08 新增优化（第二轮）

- `_get_tables()` 模块缓存：避免每次调用 `load_tables()` 重复加载 memmap 文件。
- `_expand_red_mask()` 位运算：原来用 dict 迭代展开赤宝牌，改用位运算直接检查特定位置，避免 dict 迭代和 tuple 构造开销。
- 边去重从 `tuple` 改为 `int`：`edge_keys: set[tuple[int, int]]` → `set[int]`，key = `(source << 16) | target`，消除 tuple 对象分配和哈希开销。
- Dora 计数内联：添加 `_count_dora_closed_hand()` 直接内联 dora 计数，替换 `ScoreCalculator.count_dora()` 和 `count_reddora()` 调用，闭门手无副露时消除方法调用和 melds 迭代开销。

## 2026-06-08 新增优化（第三轮）

- `_SEPARATE_CACHE` 手牌分解缓存：对 `_separate_closed_fast` 结果按 (hashes, win_tile, tsumo) 缓存。搜索 4471 节点共调用 1543 次分解，命中率高，**约 12% 性能提升**。
- `hand_origin` 预计算为 tuple：避免搜索过程中重复 `tuple(list)` 转换。
- `_PATTERN_CACHE` 计分役缓存：对 `_check_pattern_yaku_fast` 结果按 (hashes, win_tile, win_flag, shanten_type, round_wind, seat_wind) 缓存，**约 18% 性能提升**。

## 当前基准结论

- 参考牌例：`06778p1122345s77z`，西家，东场，宝牌指示牌南，赤宝牌/里宝牌/手变/向听回退开启。
- 搜索节点数保持 `4471`，结果与旧实现一致。
- 旧 Python 实现中位数约 280ms；实验版中位数约 70.5ms，**提速约 4.0 倍**。具体数值受机器负载影响，应看同环境多次中位数。
- 原 `mahjong-cpp` C++ 实现仍明显更快，实验版只是把 Python 实现的主要热点先压下来。

## 下一步优化建议（按优先级）

1. **图构建递归改迭代**：当前 `_draw_node` / `_discard_node` 仍是 Python 递归，递归调用本身占总时间约 40%（120ms/300ms）。已尝试用显式栈队列实现迭代版本，但因状态同步复杂（player.hand / hand_counts / wall_counts 三者需保持一致）且 bug 难定位，暂未成功。后续可考虑用 Cython 或 Rust 编写图的展开逻辑。
2. **候选牌迭代向量化**：已分析，prange 不可用 — 循环体内每次迭代修改共享状态（hand_counts / wall_counts），必须顺序执行，无法并行。37 张牌遍历本身开销仅约 3ms（占总 1%），不是瓶颈。
3. **接入 API 加开关**：已实现。`ExpectedScoreConfig.use_numba = True` 启用 Numba 加速版。`POST /calc` 请求体中 `use_numba: true` 字段启用，默认为 `True`。副露手牌自动回退到 Python 实现。