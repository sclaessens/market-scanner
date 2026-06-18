# ME-RUN15 operator commands

## Validate branch

```bash
cd /Users/sclaessens/Documents/market-scanner

git fetch origin
git checkout me-run15-implement-real-cached-source-batch-dry-run-command-visibility-2
git status --short --branch | tee /dev/tty | pbcopy
python -m pytest tests/market_engine/run -q | tee /dev/tty | pbcopy
git diff --check | tee /dev/tty | pbcopy
```

## First real cached-source batch dry-run

```bash
market-engine-cached-source-batch-dry-run --source-snapshot-root data/market_engine/source_snapshots --discover-cached-tickers | tee /dev/tty | pbcopy
```

## Optional local artifact run

```bash
market-engine-cached-source-batch-dry-run --source-snapshot-root data/market_engine/source_snapshots --discover-cached-tickers --write-local-artifacts --artifact-output-root artifacts/market_engine | tee /dev/tty | pbcopy
```

## Artifact tree review

```bash
find artifacts/market_engine -maxdepth 4 -type f | sort | tee /dev/tty | pbcopy
```
