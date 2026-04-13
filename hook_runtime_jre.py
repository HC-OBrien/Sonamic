# runtime hook — runs before any app code when launched as a frozen exe
#
# Points jpype at the JRE we bundled inside the app folder so the
# user never needs Java installed separately.

import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    # sys._MEIPASS is the _internal/ folder next to the .exe in onedir mode
    jre = Path(sys._MEIPASS) / "jre"
    if jre.is_dir():
        os.environ["JAVA_HOME"] = str(jre)
        # put the JRE's bin/ on PATH so java.exe is findable too
        os.environ["PATH"] = str(jre / "bin") + os.pathsep + os.environ.get("PATH", "")
