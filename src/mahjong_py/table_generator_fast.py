"""稀疏查表数据生成器（并行版）。

该脚本用于离线生成 `suits_table.bin` 与 `honors_table.bin`，
供运行时转换为稠密 memmap 并用于向听/进张/无效牌查询。
"""

from __future__ import annotations

import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from heapq import heappop, heappush
from itertools import combinations_with_replacement, product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import typer

from .tables import _repo_root

# 配置部分
KeyType = int
ValueType = List[int]
VALUES_PER_KEY = 10

# -----------------------------------------------------------------------------
# 基础工具函数
# -----------------------------------------------------------------------------

def _default_out_dir() -> Path:
    """默认输出目录：mahjong-py/data/config。"""
    # 默认输出到 mahjong-py/data/config
    return _repo_root() / "data" / "config"

def _suits_hash(pattern: Sequence[int]) -> int:
    # 以 5 进制编码数牌计数
    h = 0
    for x in pattern:
        h = 5 * h + int(x)
    return h

def _honors_hash(pattern: Sequence[int]) -> int:
    # 以 5 进制编码字牌计数
    h = 0
    for x in pattern:
        h = 5 * h + int(x)
    return h

def _index_from_win_tiles(num_win_tiles: int) -> int:
    # 索引计算：每 3 张牌一个槽位，余数占用额外槽位
    return num_win_tiles // 3 + (5 if num_win_tiles % 3 != 0 else 0)

# -----------------------------------------------------------------------------
# 形态生成器 (Generator Functions)
# -----------------------------------------------------------------------------

def list_suits_win_patterns() -> List[List[int]]:
    """列举数牌（单一花色）所有和牌型计数模式。"""
    # 列举数牌的和牌型
    patterns = []
    num_shuntsu_patterns = range(5)     # 顺子数量 0-4
    num_koutsu_patterns = range(5)      # 刻子数量 0-4
    num_head_patterns = range(2)        # 雀头数量 0-1
    shuntsu_positions = range(7)        # 顺子起始牌 （0-6）1-7
    koutsu_positions = range(9)         # 刻子起始牌 （0-8）1-9
    head_positions = range(9)           # 雀头起始牌 （0-8）1-9

    for num_shuntsu, num_koutsu, num_head in product(
        num_shuntsu_patterns, num_koutsu_patterns, num_head_patterns
    ):
        if num_shuntsu + num_koutsu > 4:
            continue
        for shuntsu_pos in combinations_with_replacement(shuntsu_positions, num_shuntsu):
            for koutsu_pos in combinations_with_replacement(koutsu_positions, num_koutsu):
                for head_pos in combinations_with_replacement(head_positions, num_head):
                    pattern = [0] * 9
                    for i in shuntsu_pos:
                        pattern[i] += 1
                        pattern[i + 1] += 1
                        pattern[i + 2] += 1
                    for i in koutsu_pos:
                        pattern[i] += 3
                    for i in head_pos:
                        pattern[i] += 2
                    
                    if all(x <= 4 for x in pattern):
                        # 合法性判断，每张牌不能超过4张
                        patterns.append(pattern)
    return patterns

def list_honors_win_patterns() -> List[List[int]]:
    """列举字牌所有和牌型计数模式。"""
    # 列举字牌的和牌型
    patterns = []
    num_koutsu_patterns = range(5)      # 刻子数量 0-4
    num_head_patterns = range(2)        # 雀头数量 0-1
    koutsu_positions = range(7)         # 刻子起始牌 （0-6）1-7
    head_positions = range(7)           # 雀头起始牌 （0-6）1-7

    for num_koutsu, num_head in product(num_koutsu_patterns, num_head_patterns):
        for koutsu_pos in combinations_with_replacement(koutsu_positions, num_koutsu):
            for head_pos in combinations_with_replacement(head_positions, num_head):
                pattern = [0] * 7
                for i in koutsu_pos:
                    pattern[i] += 3
                for i in head_pos:
                    pattern[i] += 2
                if all(x <= 4 for x in pattern):
                    patterns.append(pattern)
    return patterns

def list_suits_patterns() -> List[List[int]]:
    """列举数牌所有可达形（每张 0..4，且总牌数 <= 14）。"""
    # 列举数牌可达形（总牌数 <= 14）
    patterns = []
    # 穷举所有可能的数牌手牌（总数<=14）
    for pattern in product(range(5), repeat=9):
        if sum(pattern) <= 14:
            patterns.append(list(pattern))
    return patterns

def list_honors_patterns() -> List[List[int]]:
    """列举字牌所有可达形（每张 0..4，且总牌数 <= 14）。"""
    # 列举字牌可达形（总牌数 <= 14）
    patterns = []
    for pattern in product(range(5), repeat=7):
        if sum(pattern) <= 14:
            patterns.append(list(pattern))
    return patterns

# -----------------------------------------------------------------------------
# 核心计算逻辑 (Worker Function)
# -----------------------------------------------------------------------------

def process_chunk(args: Tuple) -> Path:
    """单个进程处理一部分 patterns 的计算任务。"""
    patterns_chunk, win_patterns, is_suits, tmp_path = args
    table: Dict[KeyType, ValueType] = {}
    
    # 预计算 win_patterns 的 sum 和 index，减少内层循环计算量
    # 结构: [(pattern_list, sum, index), ...]
    optimized_win_patterns = []
    for wp in win_patterns:
        s = sum(wp)
        idx = _index_from_win_tiles(s)
        optimized_win_patterns.append((wp, s, idx))

    # 初始化最大距离 (2^31 - 1)
    MAX_DIST = 2147483647

    for pattern in patterns_chunk:
        key = _suits_hash(pattern) if is_suits else _honors_hash(pattern)
        
        # 初始化 10 个槽位： [distance, wait, desc] 压缩在一起
        # 初始时 distance 极大，wait/desc 为 0
        # 我们用一个临时结构存储： values[index] = (min_dist, wait_bits, desc_bits)
        temp_values = [[MAX_DIST, 0, 0] for _ in range(VALUES_PER_KEY)]
        
        p_len = len(pattern)

        for win_pattern, wp_sum, index in optimized_win_patterns:
            # --- 核心优化：计算距离 ---
            distance = 0
            for i in range(p_len):
                if win_pattern[i] > pattern[i]:
                    distance += win_pattern[i] - pattern[i]
            
            # --- 核心优化：单次遍历合并逻辑 ---
            current_entry = temp_values[index]
            
            if distance < current_entry[0]:
                # 发现更优路径：重置距离，重新计算 wait/desc
                wait = 0
                desc = 0
                for i in range(p_len):
                    if win_pattern[i] > pattern[i]:
                        wait |= 1 << i
                    elif win_pattern[i] < pattern[i]:
                        desc |= 1 << i
                
                current_entry[0] = distance
                current_entry[1] = wait
                current_entry[2] = desc
                
            elif distance == current_entry[0]:
                # 距离相同：合并 wait/desc
                for i in range(p_len):
                    if win_pattern[i] > pattern[i]:
                        current_entry[1] |= 1 << i
                    elif win_pattern[i] < pattern[i]:
                        current_entry[2] |= 1 << i
        
        # 将 temp_values 压缩回 int 列表
        # format: dist | (wait << 4) | (desc << 13)
        final_values = []
        for dist, wait, desc in temp_values:
            packed = (dist & 0b1111) | (wait << 4) | (desc << 13)
            final_values.append(packed)
            
        table[key] = final_values

    sorted_records = sorted(table.items(), key=lambda kv: kv[0])
    write_sparse_table(tmp_path, sorted_records)
    return tmp_path

# -----------------------------------------------------------------------------
# 主控制逻辑 (Manager)
# -----------------------------------------------------------------------------

def _read_record(stream, record_struct: struct.Struct) -> Tuple[int, List[int]] | None:
    # 读取单条记录（key + 10 值）
    chunk = stream.read(record_struct.size)
    if not chunk:
        return None
    if len(chunk) != record_struct.size:
        raise IOError("Truncated record in temp table.")
    key, *vals = record_struct.unpack(chunk)
    return int(key), [int(v) for v in vals]


def merge_sorted_tables(temp_paths: List[Path], out_path: Path) -> None:
    """将多个已排序的临时表文件进行多路归并，输出最终稀疏表。"""
    # 多路归并各分块的已排序结果
    record_struct = struct.Struct("<" + "i" * (1 + VALUES_PER_KEY))
    streams = [p.open("rb") for p in temp_paths]
    try:
        heap = []
        for idx, f in enumerate(streams):
            rec = _read_record(f, record_struct)
            if rec is not None:
                heappush(heap, (rec[0], idx, rec[1]))

        with out_path.open("wb") as out:
            while heap:
                key, idx, values = heappop(heap)
                out.write(record_struct.pack(int(key), *[int(v) for v in values]))
                rec = _read_record(streams[idx], record_struct)
                if rec is not None:
                    heappush(heap, (rec[0], idx, rec[1]))
    finally:
        for f in streams:
            f.close()


def generate_table_parallel(
    patterns: List[List[int]],
    win_patterns: List[List[int]],
    is_suits: bool,
    out_path: Path,
    workers: int,
    show_progress: bool,
) -> None:
    """并行生成稀疏查表数据。"""
    
    print(f"Start generating {out_path.name}...")
    print(f"  Total Patterns: {len(patterns)}")
    print(f"  Win Patterns:   {len(win_patterns)}")
    
    # 1. 准备并行任务
    num_workers = max(1, workers)
    chunk_size = len(patterns) // num_workers + 1
    chunks = [patterns[i:i + chunk_size] for i in range(0, len(patterns), chunk_size)]
    
    print(f"  Spawning {num_workers} workers...")
    start_time = time.time()
    
    # 2. 并行计算，每个分块写临时表
    tmp_dir = out_path.parent / f".tmp_{out_path.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_paths = [tmp_dir / f"chunk_{i:04d}.bin" for i in range(len(chunks))]

    tasks = [(chunk, win_patterns, is_suits, tmp_paths[i]) for i, chunk in enumerate(chunks)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_chunk, tasks)
        for i, _ in enumerate(results):
            if show_progress:
                elapsed = time.time() - start_time
                print(f"  [Worker {i+1}/{len(chunks)}] Finished. Elapsed: {elapsed:.2f}s")

    # 3. 归并写入最终文件
    print("  Merging and writing to disk...")
    merge_sorted_tables(tmp_paths, out_path)

    for p in tmp_paths:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    
    total_time = time.time() - start_time
    print(f"Done! {out_path.name} generated in {total_time:.2f}s.\n")


def write_sparse_table(path: Path, records: Iterable[Tuple[KeyType, ValueType]]) -> None:
    # 以稀疏格式写入：key + 10 个值
    path.parent.mkdir(parents=True, exist_ok=True)
    # 格式: Key(int32) + 10 * Value(int32)
    record_struct = struct.Struct("<" + "i" * (1 + VALUES_PER_KEY))
    
    with path.open("wb") as f:
        for key, values in records:
            if len(values) != VALUES_PER_KEY:
                raise ValueError(f"Expected {VALUES_PER_KEY} values, got {len(values)}")
            f.write(record_struct.pack(int(key), *[int(v) for v in values]))


app = typer.Typer(help="Generate shanten tables (Parallel Optimized).")


@app.command()
def main(
    out_dir: Path = typer.Option(
        _default_out_dir(), "--out-dir", help="输出目录"
    ),
    workers: int = typer.Option(
        os.cpu_count() or 4, "--workers", help="并发进程数"
    ),
    no_progress: bool = typer.Option(False, "--no-progress", help="关闭进度输出"),
    suits: bool = typer.Option(False, "--suits", help="只生成数牌表"),
    honors: bool = typer.Option(False, "--honors", help="只生成字牌表"),
) -> None:
    # 进度开关
    show_progress = not no_progress

    # 默认全部生成
    if not suits and not honors:
        suits = True
        honors = True

    # 1. 生成字牌表
    if honors:
        honors_patterns = list_honors_patterns()
        honors_win = list_honors_win_patterns()
        generate_table_parallel(
            honors_patterns,
            honors_win,
            is_suits=False,
            out_path=out_dir / "honors_table.bin",
            workers=workers,
            show_progress=show_progress,
        )

    # 2. 生成数牌表 (这是最耗时的部分)
    if suits:
        suits_patterns = list_suits_patterns()
        suits_win = list_suits_win_patterns()
        generate_table_parallel(
            suits_patterns,
            suits_win,
            is_suits=True,
            out_path=out_dir / "suits_table.bin",
            workers=workers,
            show_progress=show_progress,
        )

if __name__ == "__main__":
    app()