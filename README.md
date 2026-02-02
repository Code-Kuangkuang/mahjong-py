# mahjong-py

This folder is a **pure Python + Numba** port scaffold of the core algorithms from the sibling project `mahjong-cpp`.

## What works (initial scaffold)

- Table loader (reads `../mahjong-cpp/data/config/{suits_table.bin,honors_table.bin}` and builds dense `numpy.memmap` files)
- Shanten (regular / seven pairs / thirteen orphans)
- Necessary tiles / Unnecessary tiles (based on the same DP merge logic as C++)
- Score calculation (core yaku + fu/han + score table)

## Score CLI

After installing the package in editable mode, use:

mahjong-score 123m789p123s77z 7z --flags tsumo

Options:
- --seat: seat wind tile id (27=East, 28=South, 29=West, 30=North)
- --round: round wind tile id (27..30)
- --dora / --uradora: dora indicator tiles (tile id or mpsz like 5m)
- --honba / --kyotaku
- --flags: tsumo/riichi/driichi/ippatsu/rob/afterkong/undersea/underriver/nagashi/tenhou/chihou/renhou
- --meld-json: JSON list of melds
- --meld-file: path to JSON list of melds

Example with open melds (meld JSON in examples/melds_open.json):

mahjong-score 3s4s22z 5s --flags tsumo undersea --meld-file examples/melds_open.json

## FastAPI server

Run the server:

uvicorn mahjong_py.server:app --host 0.0.0.0 --port 8000

POST /calc accepts the same request JSON shape as mahjong-cpp (subset for now). It returns:

- success: true/false
- request: echoed payload
- response: shanten + (placeholder) stats/config

POST /score calculates yaku/han/fu/score. Example payload:

{
	"hand": [0,0,0,1,1,1,2,3,4,18,18,18,28,28],
	"melds": [],
	"seat_wind": 27,
	"round_wind": 27,
	"dora_indicators": [],
	"uradora_indicators": [],
	"win_tile": 28,
	"win_flag": 2,
	"enable_reddora": true
}

## Quick start

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[dev]
pytest
```

The first `pytest` run may take longer because it will build the dense memmap tables under `data/generated/`.
