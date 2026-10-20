# lpf-toolkit

A command-line driver and Jupyter notebooks for
**[Ligand PathFinder](https://github.com/meilerlab/LigandPathFinder)** (LPF) —
Rosetta-based ligand and peptide (un)binding pathways.

LPF produces a guide path through a protein and then a dense trajectory along
it, with a Rosetta binding-energy estimate at every step. It is a powerful
method with a demanding set-up: a Rosetta residue type for the ligand, a
conformer library, a `start.pdb` laid out just so, two guide points, retargeted
RosettaScripts protocols, and a `run.json` per stage. This toolkit builds all of
that, runs both stages, and analyses the result.

> **Unofficial.** Written by users of LPF, not by its authors. The upstream code
> is run **unmodified** — everything here produces ordinary input files that
> `lpf/lpf_coarse_path.py` and `lpf/lpf_trajectory.py` read. Nothing is
> monkey-patched.

Method paper: Zlobin A, Eisenhuth P, Ferrario V, Meiler J. *Exploration of
ligand (un)binding trajectories with Rosetta.* bioRxiv 2026,
doi:[10.64898/2026.10.07.757317](https://www.biorxiv.org/content/10.64898/2026.10.07.757317v1)

---

## What it gives you

```bash
# a small molecule, start to finish
python lpf_run.py run --target complex.pdb --target-chains A \
    --ligand ligand.sdf --project runs/mine

# a peptide, pulled out of the complex it is bound in
python lpf_run.py run --mode peptide --target complex.pdb \
    --target-chains A --peptide-chain B --project runs/my_peptide
```

* **Input preparation that enforces LPF's unwritten requirements.** The ligand
  must be last in `start.pdb`, HETATM, one residue, with names matching
  `lig.params`; the trajectory spline needs ≥ 4 guide points; the protocol XMLs
  hardcode `chain="X"` and Rosetta never reads `input.ligand_chain`. All of that
  is built correctly and then verified.
* **A preflight that reports every problem at once** instead of failing on the
  first one, 40 minutes into a run.
* **Exit-route scanning.** Rather than reading coordinates off PyMOL, rays are
  cast from the bound pose and ranked by the width of their narrowest point.
* **Peptide support**, treating the peptide as one large flexible Rosetta ligand
  (see [Peptides](#peptides)).
* **A PyRosetta backend**, so you need no compiled Rosetta binary.
* **Replicates, clustering and convergence**, implementing the paper's own
  analysis and telling you when you have enough replicates.
* **Four Jupyter notebooks** — ligand/peptide × local/Colab — for the same
  pipeline with inline plots and 3D views.

---

## Install

### 1. Python

```bash
conda env create -f environment.yml && conda activate lpf-toolkit
# or
pip install -r requirements.txt
```

LPF itself imports only `numpy` and `scipy`, and shells out to `gzip` (always)
and `zip` (when archiving). The rest is for preparation, plotting and clustering.

### 2. Rosetta — pick one

Rosetta is **not** included and is **licensed separately**. It is free for
academic and non-commercial use.

| | how | notes |
|---|---|---|
| **PyRosetta** | `pip install pyrosetta-installer` then<br>`python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"` | ~1.5 GB download, ~2.8 GB installed. No compiler, no database path. No MPI. |
| **A binary** | [downloads.rosettacommons.org](https://downloads.rosettacommons.org/downloads/academic/) or build from [source](https://github.com/RosettaCommons/rosetta) | the reference implementation, and the only option that can use MPI |

`--rosetta-backend auto` (the default) prefers a binary and falls back to an
importable PyRosetta, so activating a PyRosetta environment is enough.

> **PyRosetta's installer ignores virtualenvs** — it uses whichever `pip` is
> first on `PATH`. Check where it landed with
> `python -c "import pyrosetta; print(pyrosetta.__file__)"`.

### 3. LigandPathFinder

Cloned on demand, or point at an existing checkout:

```bash
python lpf_run.py doctor                      # what can it find?
python lpf_run.py prep --clone ...            # clone it if missing
python lpf_run.py prep --lpf-repo /path/to/LigandPathFinder ...
```

### 4. Check, then prove it

```bash
python lpf_run.py doctor                      # packages, tools, Rosetta
python lpf_run.py demo --project runs/demo    # the upstream BChE+paraoxon demo
python selftest.py                            # the toolkit's own tests, no Rosetta
```

The demo should start near −8 REU and climb — the published value for that
complex at the catalytic site.

---

## How it works

LPF runs in two stages, and this toolkit wraps both.

```
         your complex + a bound pose
                    |
   prep  ───────────┴──────────────────────────────────────────
     clean the receptor · build lig.params + lig_conformers.pdb
     pick the tracked atom · assemble start.pdb · scan for an exit
     write the protocols and options         →  project.json
                    |
   coarse ──────────┴────  stage 1: recursive bisection between two
     points, Rosetta placing the ligand at every node
                                             →  coarsepath.txt
                    |
   traj ────────────┴────  stage 2: spline the path, one Rosetta run
     per window, each starting from the last
                                             →  interface_scores.txt
                    |
   analyze / cluster ┴───  profile, replicates, convergence
```

`prep` writes `project.json`; every later stage reads it. That is what lets each
stage be its own cluster job.

### The PyRosetta backend

LPF does not import Rosetta — it **execs** the `rosetta_scripts` binary and
parses `score.sc`. PyRosetta ships no such executable, so
`pyrosetta_rosetta_scripts.py` provides one: it accepts the slice of the
`rosetta_scripts` command line LPF builds, runs the XML through PyRosetta, and
writes decoys plus a score file in the expected format. LPF is untouched; it
just execs a launcher instead of a compiled binary.

One wrinkle that matters: on some PyRosetta builds (confirmed on 2024.19)
`InterfaceScoreCalculator` delivers nothing readable from the pose, even when
constructed directly in Python — it reports to the JD2 job, which does not exist
outside the `rosetta_scripts` application. The shim therefore **computes the
interface energy itself** when the mover stays silent: complex minus separated,
with the ligand chain translated away and the constraint terms zeroed. That
agrees with Rosetta's own `InterfaceAnalyzerMover` to three decimals, and with
the native mover on a newer build to 0.35 REU.

### Peptides

LPF is built around a single Rosetta ligand residue: it tracks one `NBR_ATOM`,
writes every constraint as `<atom> <resnum><chain>`, and draws conformers from
`lig_conformers.pdb`. `--mode peptide` therefore treats the whole peptide as
**one large flexible ligand**: HETATM, one residue, with all of its backbone and
side-chain freedom in the conformer library (the options file sets
`-packing:ignore_ligand_chi true`, so Rosetta will not sample those torsions on
the fly).

It extracts the chain, reports missing heavy atoms, rebuilds a truncated
terminal carboxyl, transfers bond orders from a sequence template, sets pH-7
protonation, builds the library as a *connected ladder* from the bound pose and
reports whether it is actually connected, re-picks `NBR_ATOM` to a central
backbone atom, and sizes the scoring grid from the molecule.

**Usable range: roughly 2–12 residues.** At 20 residues the cost is survivable
(~3–6 min/window) but the conformational coverage is not — ~80 conformers cannot
represent a 20-residue chain, the library comes out disconnected at any
threshold, and the peptide can only leave as a near-rigid body. Beyond a dozen
residues the question has become protein–protein unbinding, which needs real
backbone degrees of freedom rather than a rotamer library.

---

## Documentation

**[`TUTORIAL.md`](TUTORIAL.md)** (also as a 33-page [PDF](TUTORIAL.pdf)) is the
full guide: quick starts per system type, every stage explained, replicates and
clustering, a troubleshooting table, a worked session with real numbers,
measured costs, and a reference for all 121 options generated from the parser
itself.

```bash
python lpf_run.py --help              # the stages
python lpf_run.py prep --help         # options for one stage
```

### The stages

| | |
|---|---|
| `prep` | build every input file, then stop |
| `coarse` | stage 1 — the coarse path by bisection |
| `traj` | stage 2 — the dense trajectory (resumable) |
| `analyze` | summary, energy profile, CSV, archive |
| `run` | prep + coarse + traj + analyze |
| `cluster` | pool replicates, cluster paths, average per cluster, convergence |
| `rescore` | recompute interface energies without redoing the docking |
| `relocate` | repair a project after moving its directory |
| `doctor` | check the environment |
| `demo` | the upstream BChE+paraoxon example |

---

## What is in this repository

| file | |
|---|---|
| `lpf_run.py` | the command-line driver — the thing you run |
| `lpf_nb.py` | the shared library: PDB/params handling, chemistry, path finding, config, analysis |
| `pyrosetta_rosetta_scripts.py` | `rosetta_scripts`-compatible front end for PyRosetta |
| `notebooks/LPF_{Ligand,Peptide}_{Local,Colab}.ipynb` | the same pipeline as notebooks |
| `TUTORIAL.md`, `TUTORIAL.pdf` | the full command-line guide |
| `selftest.py` | the toolkit's own tests; needs no Rosetta |
| `build_notebooks.py`, `make_tutorial.py`, `tutorial_body.md` | regenerate the notebooks and the tutorial |
| `requirements.txt`, `environment.yml` | dependencies |

The Colab notebooks embed compressed copies of `lpf_nb.py` and the shim, so they
are standalone — upload the `.ipynb` and nothing else. After editing `lpf_nb.py`,
run `python build_notebooks.py` to refresh them.

---

## Verification status

`selftest.py` runs seven groups and needs no Rosetta: module syntax, notebook
validity, notebook execution with Rosetta stubbed, CLI option coverage, the
shim's own self-test, a PDB round-trip, and the tutorial's commands.

Verified against **real Rosetta**, full pipeline:

* **ligand** — the upstream BChE+paraoxon demo, reproducing the paper's ≈ −8 REU
  bound state, and a 43-window T4 lysozyme L99A/M102Q + JZ4 run;
* **peptide** — 1CKA SH3 domain + a 9-mer, prep through trajectory: Rosetta
  accepts the 146-atom single-residue ligand, 99 rotamers and a 56 Å grid, at
  ~17 s/window with `nstruct` 2 (so ~85 s at the default 10).

Verified to Rosetta-ready inputs, with LPF's own validators accepting them: a
P450/HEM system (cofactor in a separate chain, external PDB guide path,
`dock_zero`).

**Not exercised:** metalloenzymes with `header.pdb` constraint records,
membrane or symmetry headers, non-canonical amino acids, MPI, and the compiled
`binary` backend's Rosetta calls. Parameters are system-specific — read the
retuning checklist in the upstream `demo/tutorial.md` §5.

---

## Citing

Cite the **method paper** for the science. If this toolkit saved you time, a
mention in the acknowledgements or a link is welcome, but it is the paper that
deserves the citation. See `CITATION.cff`.

## Licences

This toolkit is MIT (see `LICENSE`). LigandPathFinder is MIT, covering its own
code only.

**Rosetta is licensed separately and is not distributed here** — that includes
PyRosetta, `molfile_to_params.py`, and the `rosetta_py` modules this toolkit can
download on your behalf. You need your own academic or non-commercial Rosetta
licence; commercial use requires a separate one.
