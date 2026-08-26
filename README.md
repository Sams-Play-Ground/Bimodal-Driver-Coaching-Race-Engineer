# Bimodal AI Driver Training Framework — Public Repository

Companion code for *A Bimodal Deep Learning Framework for Driver Training by
Integrating Computer Vision and Telemetry via Explainable AI* (Samwel N.
Njehia, USIU-Africa, BSc Data Science and Analytics, Spring 2026).

## What's in this repository

This repository contains the data collection, labelling, preprocessing,
fusion architecture, feedback generation, and desktop application code for
the project.

| Module | Purpose | Report Reference |
|---|---|---|
| `src/collection` | Telemetry + video session recording | §3.4 |
| `src/labelling` | Threshold-based automated annotation | §3.5.1 |
| `src/preprocessing` | Temporal alignment, normalization, windowing | §3.5.2 |
| `src/fusion` | Late Fusion head architecture, Persona classifier | §3.5.5, §3.3 |
| `src/feedback` | SHAP attribution, template library, feedback generation | §3.5.5 |
| `src/app` | Tkinter desktop application (upload-and-process UI) | §1.7.2, §3.3 |

## What's intentionally excluded

The following are kept in a **private repository** and are not published
here:

- Latest version of model training scripts (CNN, LSTM, fusion head training loops)
- Latest version of trained model weights / checkpoints, with improved Macro-F1 scores
- Full evaluation scripts and result outputs (confusion matrices, FER logs)
- Majority of the raw driving and framework testing data (labelled datasets)
- Latest stables app version that still in development

This split keeps the Latest trained models private while
making the surrounding engineering pipeline open for review, The dataset is made public and here is a link to it in kaggle.

CNN Dataset: `https://www.kaggle.com/datasets/samwelnjehia/cnn-module-data/data`

LSTM Dataset: `https://www.kaggle.com/datasets/samwelnjehia/lstm-module-telemetry-data/data`

Framework App: `https://usiu-my.sharepoint.com/:u:/g/personal/snjehia_usiu_ac_ke/IQDJibOPtEo_SZEx9sMJcaiEAUGy9pbl7NfTpQ_p3tT6YM8?e=YnCuCb`

## Implementation status

Most of this repo now holds real, working code ported directly from the
project's Kaggle notebooks (`src/labelling`, `src/preprocessing`,
`src/fusion`, `src/feedback`, `src/collection/telemetry_recorder.py`).

`src/collection/video_capture.py`, `src/collection/session_manager.py`, and
all of `src/app/` are still interface stubs — the video capture and desktop
UI layers were designed but not implemented before submission. They document
the intended signatures for future work.

## Setup

```bash
pip install -r requirements.txt
```

Edit the storage path in `config/settings.py` to match your local removable
storage medium mount point before running any collection or preprocessing
scripts.

## Running the desktop app

```bash
python -m src.driver_coaching_app.main
```

The app expects trained model files to be bundled locally (see
`PipelineController` in `src/app/controllers/pipeline_controller.py`) — these
are produced by the training pipeline in the private repository and copied
into the application bundle at packaging time. Additionally there is a coppy of 
the trained models checkpoints along with optimizations for use in the app folder.

## Status

This is an active undergraduate research project. Module signatures are
defined and stubbed; implementation is in progress per the 13-week project
timeline in Appendix A of the proposal.
