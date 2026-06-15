# mahjong-py

本项目是 `mahjong-cpp` 的 **Python 版本**，核心算法保持一致，便于脚本调用、接口服务与 GUI 交互。

## 功能概览

- 向听数：一般形 / 七对子 / 国士无双
- 进张 / 打张 / 听牌分析
- 计分：役种、翻数、符数、点数
- CLI 命令行
- FastAPI 服务
- GUI（Maliang）点选牌面

## 快速开始（Windows）

在项目根目录执行：

1. 创建并激活虚拟环境
2. 安装依赖
3. 运行测试

（首次运行会在 `data/generated/` 生成表，耗时稍长）

## CLI 用法

计分（示例）：

`mahjong-score 123m789p123s77z 7z --flags tsumo`

常用参数：

- `--mode`: auto/score/shanten
- `--seat` / `--round`: 自风/场风（27=东，28=南，29=西，30=北）
- `--dora` / `--uradora`: **宝牌指示牌**（牌 id 或 mpsz，如 5m）
- `--honba` / `--kyotaku`
- `--flags`: tsumo/riichi/driichi/ippatsu/rob/afterkong/undersea/underriver/nagashi/tenhou/chihou/renhou
- `--meld-json` / `--meld-file`: 副露 JSON

带副露示例：

`mahjong-score 3s4s22z 5s --flags tsumo undersea --meld-file examples/melds_open.json`

## GUI

运行 GUI：

`python -m mahjong_py.gui`

特性：

- 牌面图片点击选牌
- 宝牌/里宝牌指示牌设置
- 结果面板清晰展示（支持自动扩展）

## FastAPI 服务

启动：

`uvicorn mahjong_py.server:app --host 0.0.0.0 --port 8000`

接口：

- `POST /calc`：向听与进张
- `POST /score`：役/翻/符/点数

`/calc` 会返回 `stats`，包含每个候选打牌的听牌概率、和牌概率、点数期望、有效牌列表与搜索节点数。

## 数据与配置

- `data/config/`：规则、pattern 配置（JSON）与里宝牌统计表（`uradora.bin`）
- `data/generated/`：运行时生成的表（memmap）

## 期望值性能优化记录

详见 [docs/expected-score-optimization.md](docs/expected-score-optimization.md)。

## 许可证

本项目基于 `mahjong-cpp`（GPLv3）修改而来，遵循 **GNU GPL v3** 许可证。
