# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for DynSonOpt
#
# A JRE is bundled inside the app folder — no Java installation needed.
# build.bat downloads and extracts it automatically before calling this spec.
#
# Build with:  build.bat
# Output:      dist\DynSonOpt\DynSonOpt.exe  (fully self-contained)

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

# ── paths ────────────────────────────────────────────────────────────────────

ROOT = Path(r"D:\Python\Python_Scripts\DynSonOpt")
VENV = ROOT / ".venv" / "Lib" / "site-packages"

PY5_ROOT     = VENV / "py5"
PY5_JARS     = PY5_ROOT / "jars"
PY5_NATIVES  = PY5_ROOT / "natives" / "windows-amd64"
JPYPE_JAR    = VENV / "org.jpype.jar"
SNDFILE_DATA = VENV / "_soundfile_data"
JRE          = ROOT / "jre"   # populated by build.bat before this spec runs

# ── helper: recursively collect a directory into a dest prefix ───────────────

def collect_dir(src: Path, dest: str):
    out = []
    for f in src.rglob("*"):
        if f.is_file():
            rel = f.parent.relative_to(src.parent)
            out.append((str(f), str(Path(dest) / rel)))
    return out

# ── collect packages that need special handling ───────────────────────────────

# line_profiler is imported by py5's bridge; its rc/ sub-package contains a
# .toml resource file accessed via importlib.resources — collect_all grabs it
lp_datas, lp_binaries, lp_hidden = collect_all("line_profiler")

# ── datas and binaries ───────────────────────────────────────────────────────

datas = [
    # py5 JARs
    *collect_dir(PY5_JARS, "py5/jars"),

    # soundfile native DLL
    (str(SNDFILE_DATA / "libsndfile_x64.dll"), "_soundfile_data"),

    # line_profiler data files (includes rc/line_profiler.toml)
    *lp_datas,
]

binaries = [
    # py5 Windows native DLLs
    *[(str(f), "py5/natives/windows-amd64")
      for f in PY5_NATIVES.glob("*.dll")],

    # jpype bridge JAR
    (str(JPYPE_JAR), "."),

    # line_profiler compiled extension and any other binaries
    *lp_binaries,
]

# ── hidden imports ───────────────────────────────────────────────────────────

hidden_imports = [
    "jpype",
    "jpype.imports",
    "jpype._pyinstaller",
    "py5",
    "py5_tools",
    "py5_tools.jvm",
    "sounddevice",
    "soundfile",
    "scipy.signal",
    "scipy.signal._upfirdn",
    "scipy.signal._upfirdn_apply",
    "matplotlib",
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends.backend_agg",
    "numpy",
    # line_profiler sub-modules (py5 bridge imports these)
    "line_profiler",
    "line_profiler.rc",
    "line_profiler.explicit_profiler",
    "line_profiler.toml_config",
    *lp_hidden,
]

# ── analysis ─────────────────────────────────────────────────────────────────

a = Analysis(
    [str(ROOT / "SonamicCode" / "Sonamic.py")],
    pathex=[str(ROOT / "SonamicCode")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[
        str(VENV / "jpype" / "_pyinstaller"),
    ],
    runtime_hooks=[
        str(ROOT / "SonamicCode" / "hook_runtime_jre.py"),
    ],
    excludes=["tkinter", "test", "unittest"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DynSonOpt",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,   # keep console visible so JVM/py5 errors show up
    icon=None,
)

# Tree() pulls the entire JRE folder in as-is under the prefix "jre"
jre_tree = Tree(str(JRE), prefix="jre")

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    jre_tree,       # bundled JRE — no Java install needed on target machine
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DynSonOpt",
)
