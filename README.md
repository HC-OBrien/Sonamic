# Sonamic

A real-time visualisation and sonification tool for dynamic optimisation algorithms.

Pick a problem, pick an algorithm, pick a sonification pathway — watch and listen as the optimiser chases a moving target.

**[Download the latest release (Windows)](https://github.com/HC-OBrien/Sonamic/releases/latest)** — extract the zip and run `DynSonOpt.exe`, no installs needed.

---

## What it does

Dynamic optimisation problems have a moving optimum — the goal shifts over time and the algorithm has to keep up. Sonamic maps what the algorithm is doing into both visuals and sound simultaneously, so you can see *and hear* the search process as it unfolds.

**Problems**
- `DynamicOneMax` — the optimum drifts slowly, each dimension shifting by a small random amount each iteration
- `RandomIntStr` — the optimum jumps to a completely random position at a fixed rate
- `AllToNone` — the optimum alternates between two extremes on a fixed schedule

**Algorithms**
- `[1+1] Evolutionary Algorithm` — a single candidate solution mutates each step; keeps the mutant if it is at least as good
- `Particle Swarm Optimisation` — a swarm of particles fly through the search space, attracted toward the best positions they and the swarm have found

**Sonification pathways**
- `Fitness Pathway` — beat frequency converges as the algorithm gets closer to the optimum; diverges when the target moves
- `Position Pathway` — the global best position is mapped spectrally across the search dimensions
- `Mutation Pathway` — accepted mutations trigger chord events; rejected ones are silent

You can also record a session — it exports an MP3 of the audio and a PNG analysis plot together.

---

## Running it

### Option A — standalone executable (no Python needed)

1. Download `Sonamic_v1.1_Windows.zip` from the [Releases page](https://github.com/HC-OBrien/Sonamic/releases/latest)
2. Extract the zip anywhere
3. Run `DynSonOpt.exe` — everything is bundled, nothing to install

### Option B — build it yourself (Windows)

1. Clone this repository
2. Open `SonamicCode\` and run `build.bat`
   - Downloads a JRE automatically (~55 MB, one-time)
   - Produces `dist\DynSonOpt\DynSonOpt.exe` in the project root

### Option C — run from source

Requires Python 3.11+ and Java (any JDK/JRE 11+) on your PATH.

```bash
pip install py5 numpy scipy sounddevice soundfile matplotlib
python Sonamic.py
```

Or use `SonamicCode\DynSonOpt.bat` if you have a virtual environment set up at `.venv\` in the project root.

---

## Controls

| Control | Action |
|---|---|
| Problem / Algorithm / Pathway dropdowns | Set up the run (locked while running) |
| Play / Pause | Start or pause the simulation |
| Stop | Stop and reset |
| REC button (bottom right) | Start / stop audio + analysis recording |
| Parameter fields (right panel) | Edit algorithm parameters live while paused |

---

## File overview

| File | What it is |
|---|---|
| `Sonamic.py` | Main app window and UI |
| `DynamicProblems.py` | Moving-optimum problem definitions |
| `DynamicOptimisers.py` | OPO and PSO algorithm implementations |
| `Audiovisualisations.py` | Ties algorithm + visualiser + audio together |
| `Visualisations.py` | py5 rendering for each algorithm |
| `Sonifications.py` | Sound banks and sonification pathway classes |
| `Synthesizer.py` | Audio engine (sounddevice, threading) |
| `ExtractAnalysis.py` | Recording, fitness tracking, PNG/MP3 export |
| `build.bat` | Builds the standalone executable |
| `DynSonOpt.spec` | PyInstaller configuration |
| `hook_runtime_jre.py` | Points the frozen app at the bundled JRE |

---

## Dependencies

- [py5](https://py5coding.org/) — Python wrapper for Processing (requires Java)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [sounddevice](https://python-sounddevice.readthedocs.io/)
- [soundfile](https://pysoundfile.readthedocs.io/)
- [matplotlib](https://matplotlib.org/)

The standalone build bundles everything including the JRE — `build.bat` handles it all.
