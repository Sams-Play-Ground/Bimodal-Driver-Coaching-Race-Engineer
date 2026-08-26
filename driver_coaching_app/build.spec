# PyInstaller spec for the BI Modal Driving Feedback Race Engineer app.
# Build with:  pyinstaller build.spec
import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas = []
binaries = []
hiddenimports = [
    # Qt's multimedia plugin backend often isn't picked up by static
    # analysis alone — needed for the "Annotating video..." waiting loop
    # (ui/pages/annotating_page.py) to play back an mp4 in the built exe.
    "PyQt6.QtMultimedia",
    "PyQt6.QtMultimediaWidgets",
]

# torch/torchvision/cv2 ship native binaries + data files that PyInstaller's
# static analysis can miss — collect_all pulls in everything needed.
for pkg in ["torch", "torchvision", "timm", "cv2", "fpdf"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# Bundle the local storage skeleton + any assets (e.g. a waiting_loop.gif)
datas += [
    ("storage", "storage"),
    ("assets", "assets"),
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="DrivingFeedbackRaceEngineer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
