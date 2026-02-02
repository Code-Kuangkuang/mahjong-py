"""向听/进张相关的稠密查表加载与生成。

运行时会尝试从 `MAHJONG_CONFIG_DIR`（或默认目录）读取稀疏表 bin，
并按需生成稠密 memmap（uint32）以便 O(1) 下标访问。
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Final, Tuple

import numpy as np

SUITS_TABLE_SIZE: Final[int] = 1_943_751  # 数牌表大小
HONORS_TABLE_SIZE: Final[int] = 77_751    # 字牌表大小
VALUES_PER_KEY: Final[int] = 10           # 每个 key 对应的 10 个槽位

# 稠密表布局：uint32[table_size, 10]，每个 uint32 打包如下：
# - bits 0..3   : 距离（4 bits）
# - bits 4..12  : 听牌位图（9 bits）
# - bits 13..21 : 舍牌位图（9 bits）


def _repo_root() -> Path:
    # .../mahjong-py/src/mahjong_py/tables.py -> parents[2] 为 .../mahjong-py
    return Path(__file__).resolve().parents[2]


def _py_config_dir() -> Path:
    # Python 项目内配置目录
    return _repo_root() / "data" / "config"


def _generated_dir() -> Path:
    # 生成的稠密表保存目录
    return _repo_root() / "data" / "generated"


def _dense_path(name: str) -> Path:
    # 稠密表文件路径
    return _generated_dir() / name


def build_dense_table(bin_path: Path, out_path: Path, table_size: int) -> np.memmap:
    # 将稀疏 bin 转成稠密 memmap（便于 O(1) 下标访问）
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mm = np.memmap(
        out_path,
        mode="w+",
        dtype=np.uint32,
        shape=(table_size, VALUES_PER_KEY),
    )
    mm[:] = 0

    record_struct = struct.Struct("<i" + "I" * VALUES_PER_KEY)

    with bin_path.open("rb") as f:
        while True:
            chunk = f.read(record_struct.size)
            if not chunk:
                break
            if len(chunk) != record_struct.size:
                raise IOError(f"Truncated record in {bin_path}.")

            key, *vals = record_struct.unpack(chunk)
            if key < 0 or key >= table_size:
                raise ValueError(f"Key out of range: {key} (size={table_size})")
            mm[key, :] = vals

    mm.flush()
    return mm


def _is_valid_dense(path: Path, table_size: int) -> bool:
    # 校验稠密表文件尺寸是否匹配
    if not path.exists():
        return False
    expected_bytes = table_size * VALUES_PER_KEY * np.dtype(np.uint32).itemsize
    try:
        return path.stat().st_size == expected_bytes
    except OSError:
        return False


_TABLES: Tuple[np.ndarray, np.ndarray] | None = None


def load_tables() -> Tuple[np.ndarray, np.ndarray]:
    """加载（或生成）稠密 memmap 表。

    返回：
        (suits_raw, honors_raw)：均为 `uint32` 的 ndarray/memmap，形状为 (size, 10)
        - suits_raw：数牌表
        - honors_raw：字牌表
    """
    global _TABLES
    if _TABLES is not None:
        return _TABLES

    config_dir = Path(os.environ.get("MAHJONG_CONFIG_DIR", str(_py_config_dir())))

    suits_bin = config_dir / "suits_table.bin"
    honors_bin = config_dir / "honors_table.bin"

    suits_dense = _dense_path("suits_u32.dat")
    honors_dense = _dense_path("honors_u32.dat")

    if not _is_valid_dense(suits_dense, SUITS_TABLE_SIZE):
        build_dense_table(suits_bin, suits_dense, SUITS_TABLE_SIZE)
    if not _is_valid_dense(honors_dense, HONORS_TABLE_SIZE):
        build_dense_table(honors_bin, honors_dense, HONORS_TABLE_SIZE)

    suits_raw = np.memmap(
        suits_dense,
        mode="r",
        dtype=np.uint32,
        shape=(SUITS_TABLE_SIZE, VALUES_PER_KEY),
    )
    honors_raw = np.memmap(
        honors_dense,
        mode="r",
        dtype=np.uint32,
        shape=(HONORS_TABLE_SIZE, VALUES_PER_KEY),
    )

    _TABLES = (suits_raw, honors_raw)
    return _TABLES
