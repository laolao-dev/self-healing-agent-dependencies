# Agent dependency regression experiments

A small research snapshot exploring how tool description and schema changes affect agent workflows, and whether baseline traces can select a smaller regression suite.

## Included

- V1: four synthetic workflows, two using a real MCP stdio tool. A description mutation preserves ordinary code tests while breaking two behavior checks. A prescribed description rollback restores them.
- V2: 24 filesystem cases on the real Deep Agents graph, using a deterministic replay model; three controlled interface mutations and trace-based selection.
- V3: portable stochastic comparison/selection library and its tests, plus a historical result summary. The original provenance-bound local-model driver is excluded because it depends on private paths and artifacts.

## Run

Python 3.12 or newer. Use a disposable virtual environment.

```sh
python -m venv .venv
# Activate .venv using your shell's usual command.
python -m pip install -r requirements-v1.txt
python v1/run_experiment.py
python -m pip install -r requirements-v2-v3.txt
python v2/smoke.py
python -m pytest v2/tests -q
python -m pytest v3/tests -q
```

V1 creates synthetic reports in `v1/artifacts`. Optional `--abl-root PATH` invokes an independently installed ABL CLI; ABL is neither modified nor bundled. Without that option, ABL verification is explicitly NOT_RUN. V2's smoke exercises real tool dispatch without paid inference. The legacy V2 full driver also needs separate Deep Agents and ABL checkouts; the portable smoke is the supported shortest path.

## Evidence and limits

Historical V1 and V2 results were PASS in their original bounded experiments. V3 was PASS_WITH_REPAIR_INCONCLUSIVE. Those are historical results, not a new live-model acceptance test. See `evidence/historical-summary.json` and `VALIDATION.md` for this publication's checks.

Repair effectiveness has not been comprehensively demonstrated, especially for stochastic description mutations. V1's router is deterministic and the repair is prescribed. One corpus and controlled mutations cannot establish reliability on arbitrary tools or production agents. V3 live inference was not rerun for publication. Transitive dependencies are not fully locked. No upstream implementation is bundled.

Original research files have no newly granted redistribution license; see RIGHTS.md. This is an experimental research asset, with questions and bug reports welcome in [Issues](https://github.com/laolao-dev/self-healing-agent-dependencies/issues).

中文：工具依赖变更与行为回归的小型实验；不代表成熟自动修复产品。
