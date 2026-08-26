# BI Modal Driving Feedback Race Engineer

A desktop app (PyQt6) that wraps your CNN (vision) + LSTM (telemetry) +
late-fusion coaching pipeline in a UI that follows your wireframe:

`Main Page → Car Spec / Process data → Diagnostics Hub (red/orange/green)
→ Deploy AI framework (processing) → Driving Coach report → Watch
annotated video (annotating) → Video player (pause/play, +5s/-5s)`

plus the hamburger menu: `Main Page / Feedback History / FER Track
record / Add new vehicle`.

## 1. Project layout

```
driver_coaching_app/
  main.py                 <- entry point
  core/                    ML + storage logic (no UI code)
    data_manager.py
    vehicle_manager.py
    cnn_engine.py          Vision (CNN) model + Grad-CAM
    lstm_engine.py         Telemetry (LSTM) model
    fusion_engine.py       Late-fusion MLP + event thresholds
    video_processor.py     Renders the annotated replay video
    pipeline.py            Orchestrates the above end to end
    report_generator.py    Builds the PDF coaching report
  ui/
    theme.py               Colors + stylesheet (matches the wireframe)
    widgets.py              Card, Sidebar, TopBar, status dots, buttons
    state.py                Shared session state passed between pages
    main_window.py          Assembles top bar + sidebar + page stack
    pages/                  One file per screen in the wireframe
  workers/                 QThread workers so long jobs don't freeze the UI
  storage/                 config / models / raw_uploads / outputs
  tests/                   pytest smoke tests for core/ (no UI needed)
  requirements.txt
  build.spec               PyInstaller spec used to produce the .exe
```

## 2. Run it locally first (recommended before building the .exe)

```bash
python -m venv .venv
.venv\Scripts\activate          # on Windows
pip install -r requirements.txt
python main.py
```

Notes:
- `torch` / `torchvision` are the two big downloads (~1-2 GB). If you
  just want to try the UI/flow without the deep learning weights, you
  can skip them — every engine (`cnn_engine.py`, `lstm_engine.py`,
  `fusion_engine.py`) detects torch's absence and falls back to a
  transparent heuristic so nothing crashes.
- Trained checkpoints already live in `storage/models/` and are
  bundled into the exe automatically (the spec includes the whole
  `storage/` folder):
  - `storage/models/CNN-module-best_model.pt`
  - `storage/models/LSTM-module-best_model.pt`
  - `storage/models/late_fusion_head.pt`
  plus their supporting config in `storage/config/`:
  `normalizer_params.json`, `lstm_optimal_thresholds.json`,
  `inference_config.json`. If any of the three checkpoints is ever
  swapped out or removed, that engine reports itself unavailable and
  the progress log says exactly which one and why — it never fails
  silently or fabricates a result.
- Telemetry CSVs should use the raw Live for Speed column names
  (`Speed_KMH`, `Brake`, `Steer`, `Engine_RPM`, etc. — see
  `LFS_TO_CANONICAL` in `core/lstm_engine.py` for the full map).
- Drop an optional `assets/waiting_loop.gif` if you want a real
  looping clip on the "Annotating video..." screen instead of the
  placeholder message.

## 3. Turn it into a Windows `.exe`

Do this step **on a Windows machine** (PyInstaller builds a `.exe`
only when run on Windows — it can't cross-compile from macOS/Linux).

1. Set up and activate a clean virtual environment, then install
   dependencies exactly as in step 2 above (`pip install -r
   requirements.txt`), so `pyinstaller` is installed too.

2. From inside the `driver_coaching_app/` folder, run:

   ```bash
   pyinstaller build.spec
   ```

3. PyInstaller writes a single-file executable to
   `dist/DrivingFeedbackRaceEngineer.exe` (this spec passes
   `a.binaries`/`a.zipfiles`/`a.datas` straight into `EXE(...)` with no
   `COLLECT()` step, which is PyInstaller's onefile pattern — there is
   no `dist/DrivingFeedbackRaceEngineer/` folder). Double-click that one
   file to launch the app — no Python install required on the target
   machine.

   - The first build can take several minutes (torch is large).
   - Onefile means the exe unpacks itself into a temp folder on every
     launch, so the first window appears a few seconds slower than a
     folder build would. That's expected, not a bug.
   - If you'd rather have a faster-starting folder build instead, swap
     the spec's `EXE(...)` call to pass `[]` instead of
     `a.binaries, a.zipfiles, a.datas`, add `exclude_binaries=True`, and
     append a `COLLECT(exe, a.binaries, a.zipfiles, a.datas, ...)` call
     after it — the standard PyInstaller onedir pattern.

4. To give it a custom icon, put an `.ico` file in `assets/` and set
   `icon="assets/your_icon.ico"` in `build.spec`'s `EXE(...)` block,
   then rebuild.

5. Test the `.exe` on a machine that does **not** have Python
   installed, to confirm nothing was missed by PyInstaller's dependency
   scan (rare, but sometimes a native DLL needs to be added to
   `binaries` in `build.spec`).

### Troubleshooting the build
- `ModuleNotFoundError` at runtime after building → add that package
  to the `collect_all(...)` loop near the top of `build.spec`.
- Exe is very large (1GB+) → expected with torch bundled; you can trim
  it by installing the CPU-only torch wheel
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`)
  before building, which is much smaller than the CUDA build.
- Antivirus flags the exe → common false-positive with PyInstaller
  binaries; code-signing the exe (via `codesign_identity` / a
  signtool step) resolves this for distribution.

## 4. Where each wireframe screen lives in the code

| Wireframe screen | File |
|---|---|
| Hamburger menu | `ui/widgets.py` (`Sidebar`) |
| Main Page (footage/telemetry/car spec/process) | `ui/pages/main_page.py` |
| Car spec (spec fields for the active vehicle) | `ui/pages/car_spec_page.py` |
| Add new vehicle (identity + full spec) | `ui/pages/add_vehicle_page.py` |
| FER Track record (list + switch active vehicle) | `ui/pages/track_record_page.py` |
| Red/orange/green hub | `ui/pages/hub_page.py` |
| "This feature is coming soon" (x2) | `ui/pages/coming_soon_page.py` |
| "Frame work is processing data..." | `ui/pages/processing_page.py` |
| "The driving coach will see you right away!" | `ui/pages/report_page.py` |
| "Annotating video, will take a while..." | `ui/pages/annotating_page.py` |
| Video replay + pause/play/+5/-5 | `ui/pages/video_player_page.py` |
| Feedback History | `ui/pages/feedback_history_page.py` |
