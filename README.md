# Build & Track ML Pipelines with DVC

raw → processed → features → model → evaluation → MLflow/DagsHub tracking


## How to run?

conda create -n test python=3.11 -y

conda activate test

pip install -r requirements.txt


## DVC Commands

git init

dvc init

dvc repro

dvc dag

dvc metrics show

-- to run dvc experiements without polluting the git



dvc repro: 
- Purpose: Reproduce the pipeline from scratch (or from the last changed stage).
- When to use:
- If you’ve changed code or data and want to rebuild the pipeline outputs.
- It runs through all dependent stages in order.

dvc exp run:
- Purpose: Reproduce the pipeline from scratch (or from the last changed stage).
- When to use:
- If you’ve changed code or data and want to rebuild the pipeline outputs.
- It runs through all dependent stages in order.

dvc exp run
dvc exp show

Override experiements:

dvc exp run -S logreg.max_iter=200 -S gb.n_estimators=300

dvc exp apply = pipeline promotion (choosing the winner and making it official in code + Git).