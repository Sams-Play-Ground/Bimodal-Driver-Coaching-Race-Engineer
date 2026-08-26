# Bimodal AI Driver Training Framework — Private Repository

**This repository is proprietary. Do not publish to GitHub or any public
host.** It is intended to live on the local machine and the 128GB removable
storage medium described in the project proposal (§3.4, §3.6).

## What's in this repository

| Module | Purpose | Proposal Reference |
|---|---|---|
| `src/models` | CNN (EfficientNet-B0) and LSTM model architectures | §3.5.3, §3.5.4 |
| `src/training` | Training loops for CNN, LSTM, and Fusion head (Kaggle) | §3.6 |
| `src/evaluation` | F1/macro-F1, FER, detuned-session t-test | §3.7 |
| `data/raw` | Raw video and telemetry from simulator sessions | §3.4 |
| `data/labelled` | Output of the public repo's annotation pipeline | §3.5.1 |
| `data/processed` | Preprocessed frames + telemetry windows ready for training | §3.5.2 |
| `checkpoints/` | Saved model weights per epoch | §3.6 |
| `notebooks/` | Kaggle notebook exports for training and evaluation runs | §3.6 |
| `results/` | Confusion matrices, FER logs, detuned-session test outputs | §3.7 |

## Workflow

1. Raw video + telemetry are recorded via the **public repo's**
   `src/collection` module and saved to the removable storage medium.
2. The **public repo's** `src/labelling` pipeline produces
   `labelled_telemetry.csv` files, saved into `data/labelled/` here.
3. Data is transferred from the removable storage medium to a private
   Kaggle Dataset for training (see proposal §3.6, Week 4 of Appendix A
   Gantt chart).
4. Training scripts in `src/training/` run inside Kaggle Notebooks
   (NVIDIA P100 GPU), with checkpoints saved back to Google Drive and
   subsequently downloaded into `checkpoints/`.
5. Evaluation scripts in `src/evaluation/` run against the held-out
   Nurburgring test sessions.
6. Once training and evaluation are complete, finalized model weights
   from `checkpoints/` are exported and copied into the **public repo's**
   `src/app` bundle for local inference in the desktop application.

## Setup

```bash
pip install -r requirements.txt
```

This repository is designed to be used both locally (for preprocessing
handoff) and uploaded as a private Kaggle Dataset / Notebook environment
for the actual training runs.

## Implementation status

`src/models`, `src/training`, `src/evaluation`, and `src/data` hold real,
working code ported from the project's Kaggle notebooks — not stubs. For
full traceability, `notebooks/` also contains the original, unmodified
`.ipynb` files plus flat `.py` exports of every cell, so any refactoring
choice made in `src/` can be checked against the exact code that actually
produced the reported results.

Raw/labelled/processed data and trained checkpoints are not included here —
only the code that produces and consumes them. Repopulate `data/` and
`checkpoints/` per the Workflow section above.
