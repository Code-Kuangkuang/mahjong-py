# mahjong-py

`mahjong-py` 是基于 `mahjong-cpp` 迁移和扩展的 Python 版日麻计算核心，提供向听、进张/打张、役种计分、何切概率统计和点数期望计算。项目可通过 Python API、CLI、FastAPI 服务和 GUI 使用。

## 功能概览

- 向听数：一般形 / 七对子 / 国士无双。
- 进张与打张：有效牌、无效牌、候选舍牌有效枚数。
- 计分：役种、翻数、符数、点数，支持副露、宝牌、里宝牌、赤宝牌和常见和牌 flag。
- 何切期望值：返回每个候选切牌的听牌概率、和牌概率、点数期望、有效牌列表和搜索节点数。
- Numba 加速期望值：`/calc` 默认对无副露手牌使用 `ExpectedScoreNumbaCalculator`，副露手牌自动回退到 Python 实现。
- FastAPI 服务：提供 `/calc`、`/score`、`/health` 和内置调试页。
- CLI：`mahjong-score` 支持计分与向听分析。
- GUI：基于 Maliang 的点选牌面界面。

## 安装与测试

Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

GUI 需要额外安装可选依赖：

```powershell
python -m pip install -e ".[gui]"
```

## 数据表

项目依赖两类数据：

- `data/config/*_patterns.json`：手牌分解 pattern 配置，随仓库提交。
- `data/config/uradora.bin`：里宝牌概率表，随仓库提交。
- `data/generated/*.dat`：运行时稠密 memmap 表，体积较大，不提交 Git。

如果新环境没有 `data/generated/*.dat`，运行时会尝试根据 `data/config/suits_table.bin` 和 `data/config/honors_table.bin` 生成。若这两个稀疏表也不存在，可先离线生成：

```powershell
python -m mahjong_py.table_generator_fast --workers 8
```

生成后的 `data/generated/` 属于本地缓存，不需要提交。

## CLI 用法

计分示例：

```powershell
mahjong-score 123m789p123s77z 7z --flags tsumo
```

向听示例：

```powershell
mahjong-score 123m456p789s11z --mode shanten
```

带副露示例：

```powershell
mahjong-score 3s4s22z 5s --flags tsumo undersea --meld-file examples/melds_open.json
```

常用参数：

- `--mode`: `auto` / `score` / `shanten`
- `--seat` / `--round`: 自风/场风，`27=东`、`28=南`、`29=西`、`30=北`
- `--dora` / `--uradora`: 宝牌指示牌，支持牌 id 或 mpsz，如 `5m`
- `--honba` / `--kyotaku`
- `--flags`: `tsumo`、`riichi`、`driichi`、`ippatsu`、`rob`、`afterkong`、`undersea`、`underriver`、`nagashi`、`tenhou`、`chihou`、`renhou`
- `--meld-json` / `--meld-file`: 副露 JSON

## FastAPI 服务

启动：

```powershell
uvicorn mahjong_py.server:app --host 0.0.0.0 --port 8000
```

接口：

- `GET /health`：健康检查。
- `POST /calc`：向听、进张/打张、何切概率和点数期望。
- `POST /score`：役种、翻数、符数和点数。

`/calc` 请求支持的核心字段：

- `hand` 或 `hand_tiles`：手牌 tile id 列表。
- `melds`：副露列表。
- `seat_wind` / `round_wind`：自风/场风。
- `dora_indicators`：宝牌指示牌。
- `enable_reddora` / `enable_uradora` / `enable_shanten_down` / `enable_tegawari` / `enable_riichi`：期望值计算开关。
- `use_numba`：是否启用 Numba 加速，默认 `true`；副露手牌会自动回退到 Python 实现。

`/calc` 响应中的 `stats` 包含每个候选切牌的：

- `tenpai_prob`：各巡目听牌概率。
- `win_prob`：各巡目和牌概率。
- `exp_score`：各巡目点数期望。
- `necessary_tiles`：当前候选下的有效牌。
- `shanten`：切牌后的向听数。

## GUI

启动 GUI：

```powershell
python -m mahjong_py.gui
```

功能：

- 牌面图片点击选牌。
- 宝牌/里宝牌指示牌设置。
- 结果面板展示向听、进张、计分结果。

## 期望值性能优化

期望值计算已从原始 Python 图结构逐步迁移到数组图和 Numba 热路径。当前 `/calc` 默认使用 Numba 加速版处理无副露手牌，并保留 Python 实现作为兼容回退。

## 许可证

本项目基于 `mahjong-cpp`（GPLv3）修改而来，遵循 **GNU GPL v3** 许可证。
