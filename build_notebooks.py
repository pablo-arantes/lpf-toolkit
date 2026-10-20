#!/usr/bin/env python3
"""
Generate the four Ligand PathFinder notebooks.

    LPF_Ligand_Local.ipynb     LPF_Ligand_Colab.ipynb
    LPF_Peptide_Local.ipynb    LPF_Peptide_Colab.ipynb

They share most of their cells, so they are generated from one source rather
than maintained as four near-copies. Re-run this after editing lpf_nb.py:

    python build_notebooks.py
"""

import json
import os
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "lpf_nb.py")

PAPER = ("Zlobin A, Eisenhuth P, Ferrario V, Meiler J. *Exploration of ligand "
         "(un)binding trajectories with Rosetta*. bioRxiv 2026. "
         "doi:[10.64898/2026.10.07.757317]"
         "(https://www.biorxiv.org/content/10.64898/2026.10.07.757317v1)")
REPO = "https://github.com/meilerlab/LigandPathFinder"


# --------------------------------------------------------------------------
# cell helpers
# --------------------------------------------------------------------------

def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def param(name, value, kind="string", options=None, comment=None, colab=True):
    """
    One parameter assignment: a Colab form field, or a plain annotated assignment.

    Colab parses everything after `#@param` as the field spec, so the explanatory
    comment goes on its own line above rather than trailing the spec.
    """
    if isinstance(value, bool):
        lit = "True" if value else "False"
    elif isinstance(value, str):
        lit = json.dumps(value)
    else:
        lit = repr(value)

    lines = []
    if colab:
        if comment:
            lines.append(f"# {comment}")
        spec = f"{name} = {lit}"
        if options is not None:
            spec += "  #@param [" + ", ".join(json.dumps(o) for o in options) + "]"
        elif isinstance(value, bool):
            spec += '  #@param {type:"boolean"}'
        elif kind in ("number", "integer"):
            spec += f'  #@param {{type:"{kind}"}}'
        else:
            spec += '  #@param {type:"string"}'
        lines.append(spec)
    else:
        line = f"{name} = {lit}"
        if comment:
            line += f"   # {comment}"
        elif options:
            line += f"   # one of: {', '.join(options)}"
        lines.append(line)
        if options and comment:
            lines.append(f"#    one of: {', '.join(options)}")
    return "\n".join(lines)


def params_block(entries, colab, title=None):
    lines = []
    if colab and title:
        lines.append(f'#@title {title} {{run: "auto"}}')
    elif title:
        lines.append(f"# ---- {title} ----")
    for e in entries:
        lines.append(param(colab=colab, **e))
    return "\n".join(lines)


# ==========================================================================
# 0. title
# ==========================================================================

def cell_title(kind, env):
    who = "ligand" if kind == "ligand" else "peptide"
    badge = ""
    if env == "colab":
        badge = ("> **Google Colab.** Rosetta comes from `pip install pyrosetta` "
                 "(step 2, the default here) through a small `rosetta_scripts`-"
                 "compatible front end, since LPF execs that binary rather than "
                 "importing Rosetta. A Drive-hosted Rosetta binary also works and "
                 "skips the per-session install.\n\n"
                 "> Runs outlast Colab sessions. Put `output.path` on Drive and leave "
                 "checkpointing on; re-running the trajectory cell resumes where it "
                 "stopped.\n")
    else:
        badge = ("> **Local run.** Expects a conda environment with `numpy` and `scipy` "
                 "(the two packages LPF itself imports), plus `gzip` and `zip` on PATH. "
                 "See the setup cell.\n")

    peptide_extra = ""
    if kind == "peptide":
        peptide_extra = """
## How a peptide is handled here

LPF is built around a single Rosetta ligand residue: it reads `NBR_ATOM` from
`lig.params`, tracks that one atom along the path, restrains every ligand atom
to a per-atom ghost residue, and draws conformers from `lig_conformers.pdb`.
Every constraint it writes is of the form `<atom> <resnum><chain>` — one residue.

So this notebook treats the **whole peptide as one large flexible ligand**:

| | |
|---|---|
| representation | the peptide becomes a single Rosetta residue type via `molfile_to_params.py` |
| in `start.pdb` | HETATM records, one residue, one chain |
| flexibility | entirely from the conformer library — the options file sets `-packing:ignore_ligand_chi true`, so Rosetta does not sample those torsions on the fly |
| tracked atom | re-picked to a central backbone atom (`molfile_to_params` tends to choose a terminal one) |
| protocols | the upstream XMLs, retuned for peptide size: scoring grid sized from the molecule, wider interface cutoffs, looser C-alpha restraints, bigger `Transform` box |

**This keeps the upstream code unmodified.** The alternative — keeping the
peptide as a real polypeptide chain — would need `util.py` forked
(`read_window` only sees HETATM; `create_RMSD_constraints` and
`swap_lig_coords` key atoms by name alone, which collides across residues) and
the RosettaLigand movers replaced. That is a different project.

**Practical ceiling: roughly 2-12 residues.** Beyond that `molfile_to_params`
struggles, the scoring grid gets expensive, and a conformer library stops being
able to represent the accessible space.

**The peptide parameter defaults in this notebook are not from the paper.** They
are the authors' small-molecule values scaled for a molecule an order of
magnitude larger, with the reasoning written next to each one. Treat them as a
starting point and retune using the checklist in the appendix.
"""

    return md(f"""
# Ligand PathFinder — {who} (un)binding trajectories {"· Google Colab" if env == "colab" else "· local"}

Builds physically reasonable **{who} binding/unbinding pathways** for a
protein–{who} complex, and a Rosetta binding-energy estimate at every step
along them.

Method: {PAPER}
Code: [`meilerlab/LigandPathFinder`]({REPO}) — run here **unmodified**; this
notebook prepares its inputs, launches it, and analyses what it writes.

{badge}
## The two stages

| Stage | Script | In | Out |
|---|---|---|---|
| 1 · coarse path | `lpf_coarse_path.py` | start pose + a rough exit direction (2 points) | `coarsepath.txt` — a guide path found by recursive bisection |
| 2 · trajectory | `lpf_trajectory.py` | start pose + the full guide path | per-window structures + **`interface_scores.txt`** |

Stage 1 is stochastic; the recommended practice is several replicates, then the
medoid path forward. You can skip it entirely and supply a path from CAVER or
AQUA-DUCT.

## What you need

1. **Rosetta.** Either a `rosetta_scripts` binary plus its database, or
   PyRosetta via `pip` — step 2 sets up whichever you pick, and the rest of the
   notebook is identical. Free for academic and non-commercial use, and
   separately licensed from LPF:
   [downloads.rosettacommons.org](https://downloads.rosettacommons.org/downloads/academic/),
   [github.com/RosettaCommons/rosetta](https://github.com/RosettaCommons/rosetta).
2. **A protein–{who} complex in a functionally relevant bound pose.** LPF does
   *not* find the binding pose, and nothing here will either — dock or predict
   it first (Rosetta, Vina, GLIDE, AF3/Boltz/chai-1), and look at it.
3. Patience. One trajectory window is one Rosetta run of `nstruct` decoys; a
   full path is tens of windows. Expect tens of minutes to many hours.

## What this notebook will not do for you

* find or validate a binding pose,
* decide whether the exit route it proposes is the biologically relevant one,
* turn a single noisy replicate into a converged energy profile — pool
  replicates for that (appendix).
{peptide_extra}""")


# ==========================================================================
# 1. environment
# ==========================================================================

def cell_setup(env):
    if env == "colab":
        return [
            md("""
---
## 0 · Environment

Installs the Python side and mounts Drive. Rosetta itself comes in the next step.
"""),
            code('''
#@title Install packages and mount Google Drive {display-mode: "form"}
MOUNT_DRIVE = True  #@param {type:"boolean"}

import os, sys, subprocess
print("installing numpy, scipy, matplotlib, rdkit, py3Dmol ...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "numpy", "scipy", "matplotlib", "rdkit", "py3Dmol"], check=True)
# gzip is called after every Rosetta call; zip only when output.archive is on
print("installing zip ...")
subprocess.run("apt-get -qq install -y zip > /dev/null", shell=True)

if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    print("\\nDrive mounted at /content/drive — keep the Rosetta binary and the run "
          "outputs here so a dropped session does not lose them.")

print("\\nCPUs visible to this runtime:", os.cpu_count())
print("Note: MPI (rosetta_scripts.n_proc > 1) needs an .mpi. Rosetta build AND "
      "openmpi installed. A standard Colab runtime is better served by n_proc=1.")
'''),
        ]
    return [
        md("""
---
## 0 · Environment

LPF itself imports only `numpy` and `scipy`, and shells out to `gzip` (always)
and `zip` (when `output.archive` is on). The extras below are for this
notebook's preparation and plotting cells.

```bash
# the upstream environment, plus what this notebook adds
conda env create -f /path/to/LigandPathFinder/environment.yml
conda activate ligpf
conda install -c conda-forge matplotlib rdkit py3dmol jupyterlab
```
"""),
        code('''
import os, sys
# lpf_nb.py lives at the repository root; this notebook may sit in notebooks/
for _d in (".", "..", "../.."):
    if os.path.isfile(os.path.join(_d, "lpf_nb.py")):
        sys.path.insert(0, os.path.abspath(_d))
        break
else:
    raise FileNotFoundError(
        "lpf_nb.py not found here or in a parent directory. Run this "
        "notebook from the repository, or copy lpf_nb.py and "
        "pyrosetta_rosetta_scripts.py next to it.")
import lpf_nb as L

print("python:", sys.version.split()[0], "\\n")
L.check_python_deps()
print()
L.check_cli_tools(need_mpi=False)
print("\\nCPUs:", os.cpu_count())
'''),
    ]


SHIM = os.path.join(HERE, "pyrosetta_rosetta_scripts.py")


def _b64_lines(path, indent="    "):
    import base64
    import zlib
    with open(path, "rb") as fh:
        blob = base64.b64encode(zlib.compress(fh.read(), 9)).decode()
    return "".join(f'{indent}"{blob[i:i + 96]}"\n' for i in range(0, len(blob), 96))


def cell_helper_colab():
    """Colab: write lpf_nb.py and the PyRosetta shim out of embedded copies."""
    with open(HELPER) as fh:
        src = fh.read()
    # keep the embedded copy byte-identical; write it with an explicit file write
    # rather than %%writefile so the cell can also import and report afterwards
    body = (
        '#@title Helper module `lpf_nb.py` — run once, then collapse '
        '{display-mode: "form"}\n'
        '# This writes the notebook\'s helper module to disk and imports it.\n'
        '# It is the same file that ships with the local notebooks; nothing here\n'
        '# patches LigandPathFinder.\n'
        '# Imports os/sys here as well as in the install cell above, so this\n'
        '# notebook still works if you skip the install cell on a warm runtime.\n'
        'import base64, importlib, os, sys, zlib\n'
        '\n'
        '_LPF_NB_B64 = (\n'
    )
    body += _b64_lines(HELPER)
    body += (
        ')\n'
        '\n'
        '# the rosetta_scripts-compatible front end for the PyRosetta backend\n'
        '_SHIM_B64 = (\n'
    )
    body += _b64_lines(SHIM)
    body += (
        ')\n'
        '\n'
        'for _name, _blob in (("lpf_nb.py", _LPF_NB_B64),\n'
        '                     ("pyrosetta_rosetta_scripts.py", _SHIM_B64)):\n'
        '    with open(_name, "wb") as _fh:\n'
        '        _fh.write(zlib.decompress(base64.b64decode(_blob)))\n'
        'sys.path.insert(0, ".")\n'
        'import lpf_nb as L\n'
        'importlib.reload(L)\n'
        'print(f"lpf_nb.py written ({len(open(\'lpf_nb.py\').readlines())} lines) "\n'
        '      f"and imported")\n'
        'print(f"pyrosetta_rosetta_scripts.py written "\n'
        '      f"({len(open(\'pyrosetta_rosetta_scripts.py\').readlines())} lines)")\n'
        'print(f"{len(L.__all__)} helper functions available as L.*")\n'
    )
    return code(body)


# ==========================================================================
# 2. LigandPathFinder checkout
# ==========================================================================

def cells_repo(env):
    default_dir = "/content/LigandPathFinder" if env == "colab" else "./LigandPathFinder"
    colab = env == "colab"
    return [
        md("""
---
## 1 · Get LigandPathFinder

Clones the upstream repository if it is not already present. The notebook runs
`lpf/lpf_coarse_path.py` and `lpf/lpf_trajectory.py` from it as subprocesses,
and copies the stock RosettaScripts protocols out of `rosettaXML_files/`.
"""),
        code(params_block([
            dict(name="LPF_REPO", value=default_dir,
                 comment="existing checkout, or where to clone it"),
            dict(name="FORCE_RECLONE", value=False),
        ], colab, "LigandPathFinder location") + '''

LPF_REPO = L.find_lpf_repo(LPF_REPO) or L.clone_lpf_repo(LPF_REPO, force=FORCE_RECLONE)
if FORCE_RECLONE:
    LPF_REPO = L.clone_lpf_repo(LPF_REPO, force=True)
XML_SRC = os.path.join(LPF_REPO, "rosettaXML_files")
print("repo       :", LPF_REPO)
print("protocols  :", XML_SRC)
print("demo inputs:", os.path.join(LPF_REPO, "demo"))
print("\\ndocs worth reading:")
print("  ", os.path.join(LPF_REPO, "docs/config_reference.md"), " every run.json key")
print("  ", os.path.join(LPF_REPO, "demo/tutorial.md"), "       the worked walkthrough")
'''),
    ]


# ==========================================================================
# 3. Rosetta
# ==========================================================================

def cells_rosetta(env):
    colab = env == "colab"
    default_backend = "pyrosetta" if colab else "auto"
    hint = ("/content/drive/MyDrive/rosetta/rosetta_scripts.static.linuxgccrelease"
            if colab else "")

    if colab:
        backend_md = """
### Two ways to get Rosetta here

| backend | how | cost |
|---|---|---|
| **`pyrosetta`** (default on Colab) | `pip install pyrosetta-installer`, then it downloads the wheel | ~5-15 min and ~2.5 GB of runtime disk, **every session** |
| **`binary`** | a Linux `rosetta_scripts` build you keep on Drive, plus `main/database/` | one-time set-up, then instant |

PyRosetta is much the easier start: nothing to download by hand, no `chmod`, no
database path, no architecture mismatch. The price is paying the install on
every fresh runtime, and no MPI.

If you already keep a binary on Drive, that stays the faster option. Nothing
else in the notebook changes between the two.
"""
    else:
        backend_md = """
### Two ways to point this at Rosetta

| backend | when |
|---|---|
| **`auto`** (default locally) | prefers a compiled `rosetta_scripts` build and falls back to an importable PyRosetta. Activating a PyRosetta environment is enough |
| **`binary`** | force a compiled build — the normal case on a workstation or cluster, and the only one that can use MPI |
| **`pyrosetta`** | force the shim. Handy for trying the pipeline before committing to a full Rosetta install |

With `binary`, `n_proc > 1` needs an `.mpi.` build and an MPI launcher on PATH;
LPF refuses to start otherwise, and also refuses if `n_proc` exceeds `nstruct`.
If you moved the binary out of its source tree, set `ROSETTA_DATABASE` and the
notebook writes `-in:path:database` into the options file.
"""

    return [
        md("""
---
## 2 · Rosetta

LPF calls `rosetta_scripts` as a subprocess once per path node / trajectory
window — it never imports Rosetta. So a plain `pip install pyrosetta` gives it
nothing to call: **PyRosetta ships no `rosetta_scripts` executable.**

`pyrosetta_rosetta_scripts.py` (written alongside this notebook) closes that
gap. It presents the slice of the `rosetta_scripts` command line LPF uses, runs
the RosettaScripts XML through PyRosetta, and writes decoys plus a `score.sc`
in the format LPF's `ScoreParser` reads. LPF itself stays untouched — it just
execs a launcher instead of a compiled binary.

Rosetta is licensed separately from LPF, PyRosetta included. You need your own
academic or non-commercial licence; installing PyRosetta is accepting it.
""" + backend_md),
        code(params_block([
            dict(name="ROSETTA_BACKEND", value=default_backend,
                 options=["auto", "pyrosetta", "binary"],
                 comment="'auto' prefers a compiled binary and falls back to an importable PyRosetta"),
            dict(name="INSTALL_PYROSETTA", value=colab,
                 comment="pip-install PyRosetta if it is needed and absent (~1.5 GB download)"),
            dict(name="PYROSETTA_PYTHON", value="",
                 comment="interpreter that has PyRosetta; empty = this kernel"),
            dict(name="SHIM_JOBS", value=1, kind="integer",
                 comment="PyRosetta backend: worker processes per Rosetta call (~1-2 GB RAM each)"),
            dict(name="ROSETTA_SCRIPTS", value=hint,
                 comment="binary backend: full path; empty = search common locations"),
            dict(name="ROSETTA_DATABASE", value="",
                 comment="binary backend: empty = find it next to the binary"),
        ], colab, "Rosetta backend") + '\nBACKEND_DIR = os.path.abspath("pyrosetta_backend")\n\n_info = L.resolve_rosetta_backend(\n    backend=ROSETTA_BACKEND,\n    rosetta_hint=ROSETTA_SCRIPTS or None,\n    python_exe=PYROSETTA_PYTHON or None,\n    shim_dir=BACKEND_DIR,\n    install_if_missing=INSTALL_PYROSETTA,\n    shim_jobs=SHIM_JOBS,\n    database=ROSETTA_DATABASE or None,\n)\nROSETTA_SCRIPTS = _info["rosetta"]\nROSETTA_DATABASE = _info["database"] or ""\nIS_MPI = _info["is_mpi"]\nBACKEND = _info["backend"]\nprint()\nprint("rosetta  :", ROSETTA_SCRIPTS)\nprint("backend  :", BACKEND)\nprint("mpi      :", IS_MPI, "(n_proc > 1 needs an .mpi. binary; the shim cannot)")\nprint("database :", ROSETTA_DATABASE or "(PyRosetta\'s own, or found by Rosetta itself)")\n'),
    ]


# ==========================================================================
# 4. project settings
# ==========================================================================

def cells_project(kind, env):
    colab = env == "colab"
    is_pep = kind == "peptide"
    default_root = ("/content/drive/MyDrive/lpf_runs" if colab else "./lpf_runs")
    entries = [
        dict(name="PROJECT_DIR", value=f"{default_root}/{'peptide' if is_pep else 'ligand'}_demo",
             comment="all inputs and runs live here"),
        dict(name="LIGAND_CHAIN", value="X",
             comment="chain the " + ("peptide" if is_pep else "ligand") + " gets in start.pdb; must be free"),
        dict(name="LIGAND_RESNAME", value="pep" if is_pep else "lig",
             comment="3-character PDB residue name"),
        dict(name="N_PROC", value=1, kind="integer",
             comment="MPI ranks; >1 needs an .mpi. build and n_proc <= nstruct"),
        dict(name="ARCHIVE_WINDOWS", value=True,
             comment="zip each finished node/window directory"),
    ]
    if not is_pep:
        entries.insert(1, dict(
            name="USE_SHIPPED_DEMO", value=True,
            comment="True = run the repo's BChE+paraoxon demo to validate your install"))
    drive_note = ""
    if colab:
        drive_note = '''
if PROJECT_DIR.startswith("/content/") and not PROJECT_DIR.startswith("/content/drive"):
    print("\\n[warn] PROJECT_DIR is on the ephemeral Colab disk. Everything here is lost "
          "when the session ends — prefer a path under /content/drive/MyDrive/.")
'''
    demo_block = ""
    if not is_pep:
        demo_block = '''
if USE_SHIPPED_DEMO:
    print("\\nSHIPPED DEMO MODE: human butyrylcholinesterase + paraoxon, from "
          f"{os.path.join(LPF_REPO, 'demo')}.")
    print("The prepared start.pdb, lig.params and lig_conformers.pdb are taken from the")
    print("repository, so the preparation cells below are skipped. Use this first to")
    print("prove your Rosetta install works end to end, then set USE_SHIPPED_DEMO=False")
    print("and come back through the preparation cells with your own system.")
'''
    return [
        md("""
---
## 3 · Project settings

One directory holds the prepared inputs (`files/`, `ligand/`, `start.pdb`,
`path`) and one subdirectory per run. Paths written into `run.json` are
absolute, so a run behaves the same however it is launched.
"""),
        code(params_block(entries, colab, "Project") + f'''

PROJECT_DIR = os.path.abspath(os.path.expanduser(PROJECT_DIR))
FILES_DIR   = os.path.join(PROJECT_DIR, "files")     # protocols + options + VRT1.params
LIGAND_DIR  = os.path.join(PROJECT_DIR, "ligand")    # lig.params + lig_conformers.pdb
for d in (PROJECT_DIR, FILES_DIR, LIGAND_DIR):
    os.makedirs(d, exist_ok=True)

if N_PROC > 1 and not IS_MPI:
    extra = ("The PyRosetta backend has no MPI: set N_PROC=1 and raise SHIM_JOBS in "
             "step 2 instead (L.set_shim_jobs)." if BACKEND == "pyrosetta-shim"
             else "Use the .mpi. build, or set N_PROC=1.")
    raise ValueError(f"N_PROC={{N_PROC}} needs an '.mpi.' Rosetta build; you have "
                     f"{{os.path.basename(ROSETTA_SCRIPTS)}}. {{extra}}")

print("project:", PROJECT_DIR)
print("files  :", FILES_DIR)
print("ligand :", LIGAND_DIR)
{demo_block}{drive_note}'''),
    ]


# ==========================================================================
# 5. target protein
# ==========================================================================

def cells_protein(kind, env):
    colab = env == "colab"
    is_pep = kind == "peptide"
    entries = [
        dict(name="PROTEIN_SOURCE", value="1CKA" if is_pep else "",
             comment="a local PDB path, or a 4-character PDB ID to download"),
        dict(name="KEEP_CHAINS", value="A", comment="comma-separated protein chains to keep"),
        dict(name="REMOVE_WATERS", value=True),
        dict(name="KEEP_HETERO", value="",
             comment="cofactor/metal resnames to keep, comma-separated (e.g. HEM,ZN)"),
        dict(name="STRIP_HYDROGENS", value=False,
             comment="True lets Rosetta rebuild hydrogens"),
    ]
    if is_pep:
        entries.insert(1, dict(
            name="PEPTIDE_CHAIN_IN_SOURCE", value="B",
            comment="chain of the bound peptide in PROTEIN_SOURCE"))
    guard = "" if is_pep else '''
if USE_SHIPPED_DEMO:
    START_PDB   = os.path.join(LPF_REPO, "demo/coarse_path/start.pdb")
    PROTEIN_PDB = None
    print("demo mode: using the repository's prepared complex")
    L.pdb_summary(START_PDB, START_PDB)
    print("\\nchain X is paraoxon, already placed in the catalytic gorge and minimised.")
else:
'''
    indent = "    " if not is_pep else ""
    body = f'''
src = PROTEIN_SOURCE.strip()
if len(src) == 4 and not os.path.exists(src):
    src = L.fetch_pdb(src, PROJECT_DIR)
if not os.path.isfile(src):
    raise FileNotFoundError(f"PROTEIN_SOURCE not found: {{PROTEIN_SOURCE!r}}")

print()
L.pdb_summary(src, f"{{os.path.basename(src)}} as downloaded/supplied:")
print()

keep = {{c.strip() for c in KEEP_CHAINS.split(",") if c.strip()}}
het  = {{h.strip().upper() for h in KEEP_HETERO.split(",") if h.strip()}}
PROTEIN_PDB = L.clean_protein(
    src, os.path.join(PROJECT_DIR, "protein.pdb"),
    keep_chains=keep or None, remove_waters=REMOVE_WATERS,
    keep_hetero=het, strip_hydrogens=STRIP_HYDROGENS,
)
if LIGAND_CHAIN in keep:
    raise ValueError(f"LIGAND_CHAIN={{LIGAND_CHAIN!r}} is also a kept protein chain. "
                     f"Pick a free chain letter.")
print()
L.pdb_summary(PROTEIN_PDB, "cleaned protein:")
SOURCE_COMPLEX = src
'''
    if not is_pep:
        body = guard + "\n".join((indent + l) if l.strip() else l
                                 for l in body.strip("\n").splitlines())
    note = ("The peptide is pulled out of the same complex in the next step, so give "
            "the bound complex here and keep only the receptor chains."
            if is_pep else
            "The ligand is added in a later step, so keep only the protein (plus any "
            "cofactor you need) here.")
    extra = ("\n\nThe default is a worked example: **1CKA**, the c-Crk N-terminal SH3 "
             "domain with the proline-rich peptide PPPALPPKK in chain B. It is a real "
             "structure with a real defect — its C-terminal carboxyl is truncated — "
             "which the next step detects and repairs." if is_pep else "")
    return [
        md(f"""
---
## 4 · Target protein

{note}

Anything non-canonical you keep (cofactor, metal, modified residue) needs its own
Rosetta `.params` file, listed in `extra_files` **and** in the options file's
`-extra_res_fa`. The preflight check later verifies both.

LPF starts from a pose already relaxed into the Rosetta force field. If yours is
not, minimise it first (see `demo/tutorial.md` 1.3) — an unrelaxed start pose
shows up as a huge window-0 energy.{extra}
"""),
        code(params_block(entries, colab, "Target protein") + "\n" + body),
    ]


# ==========================================================================
# 6a. ligand input
# ==========================================================================

def cells_ligand_input(env):
    colab = env == "colab"
    entries = [
        dict(name="LIGAND_INPUT_MODE", value="sdf",
             options=["sdf", "existing_params", "smiles"],
             comment="where the ligand comes from"),
        dict(name="LIGAND_SDF", value="",
             comment="mode 'sdf': SD/MOL file whose FIRST entry is the bound pose"),
        dict(name="LIGAND_SMILES", value="",
             comment="mode 'smiles': NO bound pose — for testing only"),
        dict(name="EXISTING_LIGAND_DIR", value="",
             comment="mode 'existing_params': directory with lig.params + lig_conformers.pdb"),
        dict(name="EXISTING_LIGAND_POSE", value="",
             comment="mode 'existing_params': PDB of the ligand in its bound pose (optional)"),
        dict(name="N_CONFORMERS", value=200, kind="integer",
             comment="ETKDG conformers to embed"),
        dict(name="CONFORMER_PRUNE_RMSD", value=0.5, kind="number",
             comment="minimum RMSD between kept conformers"),
        dict(name="MOLFILE_TO_PARAMS", value="",
             comment="empty = find it in your Rosetta tree, else download it"),
    ]
    return [
        md("""
---
## 5 · Ligand → `lig.params` + `lig_conformers.pdb`

LPF needs those two filenames exactly. Three ways in:

| mode | use when |
|---|---|
| `sdf` | you have the ligand as SD/MOL **in its bound pose** (exported from your docking result). Recommended. |
| `existing_params` | you already made `lig.params` + `lig_conformers.pdb` with `molfile_to_params.py` (the upstream tutorial's route). |
| `smiles` | no bound pose at all — fine for checking the machinery, useless as science. |

**Aromaticity matters.** `molfile_to_params` expects aromatic bonds to be typed
explicitly (MDL type 4), not as alternating single/double. The SD file written
here does that, which is the manual edit `demo/tutorial.md` 1.2.1 describes.

**The first SD entry becomes the ligand in `start.pdb`.** `molfile_to_params`
writes entry 1 to `lig.pdb` and the rest to `lig_conformers.pdb`, and `lig.pdb`
is what gets merged with the protein — so entry 1 must be the bound pose. The
helpers keep it there.
"""),
        code(params_block(entries, colab, "Ligand input") + '''

import shutil
LIG_PARAMS = os.path.join(LIGAND_DIR, "lig.params")
LIG_CONFS  = os.path.join(LIGAND_DIR, "lig_conformers.pdb")
LIG_POSE_PDB = None
ATOM_MAP = {}

if USE_SHIPPED_DEMO:
    for f in ("lig.params", "lig_conformers.pdb"):
        shutil.copy(os.path.join(LPF_REPO, "demo/coarse_path/ligand", f), LIGAND_DIR)
    print("demo mode: copied the repository's paraoxon ligand files")

elif LIGAND_INPUT_MODE == "existing_params":
    src = os.path.abspath(os.path.expanduser(EXISTING_LIGAND_DIR))
    for f in ("lig.params", "lig_conformers.pdb"):
        p = os.path.join(src, f)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{f} not found in {src}")
        shutil.copy(p, LIGAND_DIR)
    LIG_POSE_PDB = (os.path.abspath(os.path.expanduser(EXISTING_LIGAND_POSE))
                    if EXISTING_LIGAND_POSE else None)
    print("using your existing ligand files")

else:
    mftp = L.ensure_molfile_to_params(
        os.path.join(PROJECT_DIR, "_molfile_to_params"),
        MOLFILE_TO_PARAMS, rosetta_bin=ROSETTA_SCRIPTS)

    if LIGAND_INPUT_MODE == "sdf":
        if not LIGAND_SDF:
            raise ValueError("set LIGAND_SDF (and make its first entry the bound pose)")
        mol = L.mol_from_sdf(LIGAND_SDF)
    else:
        if not LIGAND_SMILES:
            raise ValueError("set LIGAND_SMILES")
        mol = L.mol_from_smiles_3d(LIGAND_SMILES, n_confs=1)

    print()
    lib = L.generate_conformers(mol, n_confs=N_CONFORMERS,
                                prune_rmsd=CONFORMER_PRUNE_RMSD)
    sdf = L.write_conformer_sdf(lib, os.path.join(PROJECT_DIR, "ligand_confs.sdf"))
    print()
    res = L.run_molfile_to_params(sdf, LIGAND_DIR, script=mftp,
                                  resname=LIGAND_RESNAME, verbose=False)
    LIG_POSE_PDB = res["lig_pdb"]

print()
print("lig.params        :", LIG_PARAMS)
print("lig_conformers.pdb:", LIG_CONFS, f"({L.count_conformers(LIG_CONFS)} rotamers)")
LIGAND_RESNAME = L.params_resname(LIG_PARAMS)
print("residue name      :", LIGAND_RESNAME, "(read back from lig.params)")
LIGAND_SPAN = L.ligand_span(LIG_CONFS)
print(f"span over rotamers: {LIGAND_SPAN:.1f} A")
'''),
    ]


# ==========================================================================
# 6b. peptide input
# ==========================================================================

def cells_peptide_input(env):
    colab = env == "colab"
    entries = [
        dict(name="PEPTIDE_INPUT_MODE", value="from_complex",
             options=["from_complex", "pdb", "sdf", "existing_params"],
             comment="where the peptide comes from"),
        dict(name="PEPTIDE_PDB", value="",
             comment="mode 'pdb': peptide-only PDB, in the receptor's frame"),
        dict(name="PEPTIDE_SDF", value="",
             comment="mode 'sdf': SD file, first entry = bound pose"),
        dict(name="EXISTING_LIGAND_DIR", value="",
             comment="mode 'existing_params': dir with lig.params + lig_conformers.pdb"),
        dict(name="EXISTING_LIGAND_POSE", value="",
             comment="mode 'existing_params': PDB of the peptide in its bound pose, "
                     "atom names matching lig.params"),
        dict(name="PEPTIDE_SEQUENCE", value="",
             comment="override the sequence read from the PDB (one-letter)"),
        dict(name="REPAIR_C_TERMINUS", value=True,
             comment="build a missing terminal O / OXT"),
        dict(name="PHYSIOLOGICAL_PROTONATION", value=True,
             comment="COO-, Lys-NH3+, Arg+ instead of the neutral template"),
        dict(name="MOLFILE_TO_PARAMS", value="",
             comment="empty = find it in your Rosetta tree, else download it"),
    ]
    conf_entries = [
        dict(name="N_LOCAL", value=120, kind="integer",
             comment="small perturbations of a few torsions"),
        dict(name="N_MEDIUM", value=120, kind="integer",
             comment="larger perturbations of more torsions"),
        dict(name="N_GLOBAL", value=60, kind="integer",
             comment="free ETKDG embeddings"),
        dict(name="SIGMA_LOCAL_DEG", value=12.0, kind="number"),
        dict(name="SIGMA_MEDIUM_DEG", value=45.0, kind="number"),
        dict(name="CONFORMER_PRUNE_RMSD", value=0.6, kind="number",
             comment="minimum RMSD between kept conformers"),
        dict(name="MINIMIZE_CONFORMERS", value=True,
             comment="short MMFF relaxation to relieve clashes"),
        dict(name="RANDOM_SEED", value=2026, kind="integer"),
    ]
    return [
        md("""
---
## 5 · Peptide → one Rosetta ligand residue

Four ways in; `from_complex` pulls the peptide straight out of the complex you
loaded above, which is the normal case.

A peptide taken out of a crystal structure is usually *not* chemically complete.
That does not matter while it is "chain B residues 1–9"; it matters completely
once it has to become a single residue type with its own partial charges. So
this step:

1. extracts the chain and reports every missing heavy atom per residue,
2. rebuilds a truncated terminal carboxyl (planar `O`/`OXT`, 1.25 Å, 120°) —
   but refuses to invent missing side chains, which you must fix upstream,
3. transfers bond orders from a sequence template,
4. sets pH-7 protonation (`-COO⁻`, Lys `-NH₃⁺`, Arg guanidinium, charged
   N-terminus) — the neutral template would carry the wrong charges across
   every interface the peptide crosses, which is exactly what the energy
   profile measures,
5. adds hydrogens at the bound geometry.

Histidine is left as the template made it: the tautomer is a judgement call and
Rosetta cannot be told which one to use for a ligand residue.
"""),
        code(params_block(entries, colab, "Peptide input") + '''

import shutil
LIG_PARAMS = os.path.join(LIGAND_DIR, "lig.params")
LIG_CONFS  = os.path.join(LIGAND_DIR, "lig_conformers.pdb")
LIG_POSE_PDB, PEPTIDE_CLEAN_PDB, PEP_MOL = None, None, None

if PEPTIDE_INPUT_MODE == "existing_params":
    src = os.path.abspath(os.path.expanduser(EXISTING_LIGAND_DIR))
    for f in ("lig.params", "lig_conformers.pdb"):
        shutil.copy(os.path.join(src, f), LIGAND_DIR)
    LIG_POSE_PDB = (os.path.abspath(os.path.expanduser(EXISTING_LIGAND_POSE))
                    if EXISTING_LIGAND_POSE else None)
    print("using your existing ligand files; skipping peptide preparation")

else:
    if PEPTIDE_INPUT_MODE == "from_complex":
        PEPTIDE_CLEAN_PDB = L.extract_chain(
            SOURCE_COMPLEX, PEPTIDE_CHAIN_IN_SOURCE,
            os.path.join(PROJECT_DIR, "peptide_raw.pdb"))
    elif PEPTIDE_INPUT_MODE == "pdb":
        PEPTIDE_CLEAN_PDB = os.path.abspath(os.path.expanduser(PEPTIDE_PDB))
        if not os.path.isfile(PEPTIDE_CLEAN_PDB):
            raise FileNotFoundError(PEPTIDE_CLEAN_PDB)

    if PEPTIDE_INPUT_MODE in ("from_complex", "pdb"):
        print()
        rep = L.peptide_completeness_report(PEPTIDE_CLEAN_PDB)
        if (rep["needs_oxt"] or rep["needs_o"]) and REPAIR_C_TERMINUS:
            print()
            PEPTIDE_CLEAN_PDB = L.complete_c_terminus(
                PEPTIDE_CLEAN_PDB, os.path.join(PROJECT_DIR, "peptide.pdb"))
            rep = L.peptide_completeness_report(PEPTIDE_CLEAN_PDB, verbose=False)
        if not rep["ok"]:
            raise ValueError(
                "The peptide is not chemically complete (see above). Rebuild the "
                "missing atoms before continuing — PyMOL, pdbfixer, or a Rosetta "
                "FastRelax with the side chain repacked. Converting an incomplete "
                "peptide would silently give it the wrong chemistry.")
        print()
        PEP_MOL = L.peptide_mol_from_pdb(
            PEPTIDE_CLEAN_PDB,
            sequence=PEPTIDE_SEQUENCE or None,
            physiological_protonation=PHYSIOLOGICAL_PROTONATION)
    else:
        PEP_MOL = L.mol_from_sdf(PEPTIDE_SDF)
'''),
        md("""
### The conformer library is the peptide's only flexibility

`-packing:ignore_ligand_chi true` means Rosetta will not sample the peptide's
torsions during docking; the library below *is* its conformational freedom.

LPF filters the library to conformers within `conformer_rmsd_threshold` of the
**current** pose at every window. So the library has to be a *connected ladder*:
a purely local ensemble would never let the peptide unfold on the way out, and a
purely random one leaves gaps the filter cannot cross. Three populations are
mixed for that reason — local perturbations, medium perturbations, and free
embeddings.

The report that follows tells you whether the ladder actually connects. If it
says `NO`, raise `N_MEDIUM`, raise `SIGMA_MEDIUM_DEG`, or lower
`CONFORMER_PRUNE_RMSD` — a disconnected library makes the far conformations
unreachable however many entries the file has.
"""),
        code(params_block(conf_entries, colab, "Conformer library") + '''

if PEP_MOL is not None and PEPTIDE_INPUT_MODE != "existing_params":
    PEP_LIB = L.generate_peptide_conformers(
        PEP_MOL, n_local=N_LOCAL, n_medium=N_MEDIUM, n_global=N_GLOBAL,
        sigma_local_deg=SIGMA_LOCAL_DEG, sigma_medium_deg=SIGMA_MEDIUM_DEG,
        prune_rmsd=CONFORMER_PRUNE_RMSD, minimize=MINIMIZE_CONFORMERS,
        seed=RANDOM_SEED)
    print()
    NBR_SD_INDEX = L.suggest_nbr_atom_index(PEP_LIB)
    print()
    _D, _rep = L.conformer_library_report(PEP_LIB, nbr_idx=NBR_SD_INDEX - 1, plot=True)
'''),
        code('''
if PEP_MOL is not None and PEPTIDE_INPUT_MODE != "existing_params":
    mftp = L.ensure_molfile_to_params(
        os.path.join(PROJECT_DIR, "_molfile_to_params"),
        MOLFILE_TO_PARAMS, rosetta_bin=ROSETTA_SCRIPTS)
    sdf = L.write_conformer_sdf(PEP_LIB, os.path.join(PROJECT_DIR, "peptide_confs.sdf"))
    print()
    # nbr_atom_index sets NBR_ATOM directly, so molfile_to_params also computes a
    # matching NBR_RADIUS. Rosetta overlays every rotamer on NBR_ATOM and on the
    # orientation of its bonded atoms, so a central, well-connected atom matters.
    res = L.run_molfile_to_params(sdf, LIGAND_DIR, script=mftp,
                                  resname=LIGAND_RESNAME,
                                  nbr_atom_index=NBR_SD_INDEX)
    LIG_POSE_PDB = res["lig_pdb"]

print()
LIGAND_RESNAME = L.params_resname(LIG_PARAMS)
LIGAND_SPAN = L.ligand_span(LIG_CONFS)
print("residue name      :", LIGAND_RESNAME)
print("lig_conformers.pdb:", L.count_conformers(LIG_CONFS), "rotamers")
print(f"span over rotamers: {LIGAND_SPAN:.1f} A")

# molfile_to_params renames every atom (C1, N1, ... in atom-tree order), so for a
# peptide collapsed into one residue you lose track of which atom was whose CA.
# lig.pdb holds the bound pose with the new names, so the two can be matched on
# coordinates and the identities recovered.
ATOM_MAP = {}
if LIG_POSE_PDB and PEPTIDE_CLEAN_PDB:
    print()
    ATOM_MAP = L.map_params_atoms_to_source(LIG_POSE_PDB, PEPTIDE_CLEAN_PDB)
'''),
    ]


# ==========================================================================
# 7. NBR atom
# ==========================================================================

def cells_nbr(kind, env):
    colab = env == "colab"
    is_pep = kind == "peptide"
    why = ("""
For a peptide this is the single most consequential choice in the set-up.
`molfile_to_params` tends to pick a terminal atom, and tracking a dangling
terminus would make the guide path describe the motion of one tip rather than
the peptide leaving. It is already set from the central-atom suggestion above;
this cell lets you override it, including by naming an atom of the original
peptide.
""" if is_pep else """
For a compact small molecule whatever `molfile_to_params` chose is usually fine.
Override it if the chosen atom is a protruding tip, or if there is a chemically
meaningful atom you want the path to describe (the demo tracks paraoxon's
phosphate oxygen `O1`).
""")
    entries = [dict(name="OVERRIDE_NBR_ATOM", value="",
                    comment="params atom name, e.g. C17; empty = keep the current one")]
    if is_pep:
        entries.append(dict(name="OVERRIDE_NBR_BY_SOURCE", value="",
                            comment='or name it in the original peptide, e.g. "CA 5"'))
    src_block = '''
if OVERRIDE_NBR_BY_SOURCE.strip():
    want = OVERRIDE_NBR_BY_SOURCE.split()
    if len(want) != 2:
        raise ValueError('OVERRIDE_NBR_BY_SOURCE must be "<atom> <resseq>", e.g. "CA 5"')
    atom_name, resseq = want[0], int(want[1])
    hits = [p for p, (rn, rs, orig) in ATOM_MAP.items()
            if orig == atom_name and rs == resseq]
    if not hits:
        raise ValueError(f"no params atom maps to {atom_name} of residue {resseq}. "
                         f"Available for that residue: "
                         f"{sorted(o for _, (rn, rs, o) in ATOM_MAP.items() if rs == resseq)}")
    OVERRIDE_NBR_ATOM = hits[0]
    print(f"{OVERRIDE_NBR_BY_SOURCE!r} -> params atom {OVERRIDE_NBR_ATOM}\\n")
''' if is_pep else ""
    return [
        md(f"""
---
## 6 · Which atom does the path track?

LPF reads `NBR_ATOM` from `lig.params` and follows **that one atom** along the
guide path; Rosetta's `StartFrom(use_nbr=1)` places it on each target point, and
every conformer in the library is overlaid on it.
{why}
Changing it also changes `NBR_RADIUS`, which must enclose the whole residue as
seen from the new atom — too small and Rosetta's neighbour detection silently
truncates interactions. The helper recomputes it over **all** conformers and
adds a margin.
"""),
        code(params_block(entries, colab, "Tracked atom") + "\n" + src_block + '''
print("current NBR_ATOM  :", L.params_nbr_atom(LIG_PARAMS))
print("current NBR_RADIUS:", L.params_nbr_radius(LIG_PARAMS))
print()
ranked, scores = L.suggest_nbr_atom(conformers_pdb=LIG_CONFS,
                                    params_path=LIG_PARAMS, top=8)

if OVERRIDE_NBR_ATOM.strip():
    print()
    L.set_nbr_atom(LIG_PARAMS, OVERRIDE_NBR_ATOM.strip(),
                   conformers_pdb=LIG_CONFS, margin=0.5)

NBR_ATOM = L.params_nbr_atom(LIG_PARAMS)
print()
print("tracking:", (L.describe_atom(NBR_ATOM, ATOM_MAP) if ATOM_MAP else NBR_ATOM))
'''),
    ]


# ==========================================================================
# 8. start.pdb
# ==========================================================================

def cells_start(kind, env):
    colab = env == "colab"
    is_pep = kind == "peptide"
    entries = [
        dict(name="LIGAND_RESNUM", value=1, kind="integer",
             comment="residue number of the ligand in its own chain"),
        dict(name="HEADER_LINES", value="",
             comment="lines prepended to every task.pdb (metal constraints, membrane spans); normally empty"),
        dict(name="MINIMIZE_START", value=False,
             comment="relax the complex into the Rosetta force field first (tutorial 1.3)"),
        dict(name="MINIMIZE_NSTRUCT", value=5, kind="integer",
             comment="attempts; the best by constraint-free total_score is kept"),
        dict(name="MINIMIZE_CA_SD", value=0.5, kind="number",
             comment="C-alpha restraint sd; 0 = the tutorial's unrestrained minimize"),
    ]
    demo_guard = "" if is_pep else '''
if USE_SHIPPED_DEMO:
    print("demo mode: using the repository's start.pdb unchanged")
else:
'''
    body = '''
pose = LIG_POSE_PDB
if pose is None:
    raise ValueError(
        "No ligand pose file. Either use an input mode that produces lig.pdb, or "
        "set EXISTING_LIGAND_POSE to a PDB of the ligand in its bound pose whose "
        "atom names match lig.params.")
START_PDB = L.assemble_start_pdb(
    PROTEIN_PDB, pose, os.path.join(PROJECT_DIR, "start.pdb"),
    params_path=LIG_PARAMS, ligand_chain=LIGAND_CHAIN,
    ligand_resnum=LIGAND_RESNUM)
'''
    if not is_pep:
        body = demo_guard + "\n".join(("    " + l) if l.strip() else l
                                      for l in body.strip("\n").splitlines())
    return [
        md("""
---
## 7 · Assemble `start.pdb`

The layout LPF requires, and why:

* **the ligand comes last** — `PDBHandler.prepare_start` streams protein atoms
  and non-ligand HETATMs into the protein block only until it first meets the
  ligand chain, so anything after the ligand is dropped;
* **the ligand is HETATM, one chain, one residue** — `read_window` ignores ATOM
  records, and every constraint is written as `<atom> <resnum><chain>`;
* **its atom names match `lig.params`** — Rosetta matches ligand atoms to the
  residue type by name. File *order* need not match (the upstream demo's own
  `start.pdb` is a permutation of its params order).

The ligand block is taken from the `lig.pdb` that `molfile_to_params` wrote: its
coordinates are the bound pose and its names are already Rosetta's, which
removes the manual PyMOL renaming in `demo/tutorial.md` 1.2.4.

`header.pdb` is prepended verbatim to every generated `task.pdb` — used for
metal-binding constraints, membrane spans or symmetry records. Empty otherwise.

### Relaxing the start pose (tutorial section 1.3)

LPF assumes the start pose is already relaxed into the Rosetta force field. The
tutorial does this with the standalone `minimize` application and recommends
several attempts, keeping the best. `MINIMIZE_START` does the equivalent through
the `rosetta_scripts` binary you already configured, so it works on either
backend.

This is **not** the same as window 0 of the trajectory: that pass
(`zero_minimize.xml`) only settles the interface region under the `LigandArea`
movemap. An unrelaxed complex shows up as a large window-0 energy.

The tutorial's `minimize` command is unrestrained; `MINIMIZE_CA_SD` adds
C-alpha coordinate restraints so the fold cannot drift, which a free all-atom
minimisation of a whole protein can cause. Set it to 0 to reproduce the
tutorial exactly. Either way, **look at the result** — the tutorial calls manual
verification highly recommended, and this is the step most likely to need
system-specific attention (metalloenzymes, non-standard protonation, structural
waters).
"""),
        code(params_block(entries, colab, "start.pdb") + "\n" + body + '''

HEADER_PDB = L.write_header_pdb(
    os.path.join(PROJECT_DIR, "header.pdb"),
    [l for l in HEADER_LINES.split("\\n") if l.strip()])
print()
ok = L.check_start_pdb_vs_params(START_PDB, LIG_PARAMS, LIGAND_CHAIN)
if not ok:
    raise ValueError("start.pdb does not match lig.params — see above. LPF would "
                     "either crash or silently score the wrong thing.")
print()
L.pdb_summary(START_PDB, "start.pdb:")

if MINIMIZE_START:
    print()
    print("=" * 64)
    print("relaxing the start pose into the Rosetta force field (tutorial 1.3)")
    print("=" * 64)
    # the minimisation reads the options file, which section 9 writes; put an
    # options file and VRT1.params in place now so this can run here
    L.write_protocol_xmls(FILES_DIR, XML_SRC, chain=LIGAND_CHAIN,
                          tuning=L.LIGAND_XML_TUNING, files=("coarsepath.xml",),
                          verbose=False)
    L.write_options_file(os.path.join(FILES_DIR, "options"),
                         database=ROSETTA_DATABASE or None)
    print()
    START_PDB = L.minimize_start_pose(
        START_PDB, os.path.join(PROJECT_DIR, "start_minimized.pdb"),
        rosetta=ROSETTA_SCRIPTS, files_dir=FILES_DIR, ligand_dir=LIGAND_DIR,
        ligand_chain=LIGAND_CHAIN, nstruct=MINIMIZE_NSTRUCT,
        ca_restraint_sd=MINIMIZE_CA_SD, n_proc=N_PROC)
print()
NBR_ATOM = L.params_nbr_atom(LIG_PARAMS)
START_NBR = [a.xyz for a in L.read_pdb_atoms(START_PDB)
             if a.chain == LIGAND_CHAIN and a.name == NBR_ATOM][0]
print(f"tracked atom {NBR_ATOM} starts at "
      f"[{START_NBR[0]:.3f} {START_NBR[1]:.3f} {START_NBR[2]:.3f}]")
print()
L.conformer_subsampling_preview(START_PDB, LIG_CONFS, LIGAND_CHAIN, LIG_PARAMS)
'''),
    ]


# ==========================================================================
# 9. guide path endpoints
# ==========================================================================

def cells_path(kind, env):
    colab = env == "colab"
    is_pep = kind == "peptide"
    what = "peptide" if is_pep else "ligand"
    entries = [
        dict(name="EXIT_MODE", value="scan",
             options=["scan", "manual", "use_file"],
             comment="how the exit direction is chosen"),
        dict(name="RAY_INDEX", value=0, kind="integer",
             comment="mode 'scan': which ranked direction to take"),
        dict(name="HINT_VECTOR", value="",
             comment="mode 'scan': restrict to a cone, e.g. '0,1,0'"),
        dict(name="HINT_CONE_DEG", value=60.0, kind="number"),
        dict(name="EXIT_POINT", value="",
             comment="mode 'manual': 'x,y,z' of the exit point"),
        dict(name="GUIDE_PATH_FILE", value="",
             comment="mode 'use_file': a CAVER/AQUA-DUCT path, site-to-outside (xyz text or PDB)"),
        dict(name="REVERSE_PATH", value=False,
             comment="mode 'use_file': flip a tunnel exported outside-in"),
        dict(name="MIN_EXIT_DISTANCE", value=0.0, kind="number",
             comment="0 = auto from the molecule's span"),
        dict(name="EXIT_CLEARANCE", value=8.0, kind="number",
             comment="how far from the protein the endpoint must sit"),
        dict(name="N_SCAN_DIRECTIONS", value=512, kind="integer"),
    ]
    demo_hint = ("" if is_pep else '''
if USE_SHIPPED_DEMO and EXIT_MODE == "scan":
    print("Demo note: the repository's own two points for BChE+paraoxon are")
    print("    5.755 -5.775 -3.752    ->    9.156 -25.469 -1.308")
    print("i.e. straight down the catalytic gorge. The scan below is independent of")
    print("them; set EXIT_MODE='use_file' with")
    print(f"    {os.path.join(LPF_REPO, 'demo/coarse_path/path')}")
    print("to reproduce the published demo exactly.\\n")
''')
    return [
        md(f"""
---
## 8 · Where does it leave? (the two guide points)

The coarse stage takes exactly two points: the tracked atom's position, and a
point roughly outside the protein. It only reads the first and last point of the
file — everything between is generated by bisection — and it jitters the
endpoint anyway (`endpoint_jitter`), so the direction only has to be roughly
right.

Rather than reading coordinates off PyMOL, `scan` casts rays from the bound pose
and ranks them by how wide the exit is: each ray is sampled every 0.5 Å, the
clearance at each sample is the distance to the nearest protein heavy atom, and
the ray's **bottleneck** is the narrowest clearance before it reaches open
solvent or gets pinched shut. Compare that bottleneck with your molecule's size.

Use `manual` when you know the route, and `use_file` to skip stage 1 entirely
with a path from CAVER or AQUA-DUCT (that path needs ≥ 4 points for the
trajectory stage's spline, and must run **from the binding site outwards** —
LPF overwrites its first point with the ligand's real position, so a tunnel
exported outside-in is silently mangled rather than rejected; the cell below
checks for that).

> Clearance is measured from the **tracked atom**, not the whole molecule. For
> an elongated {"peptide" if is_pep else "ligand"} the far end is still buried
> when the tracked atom first reaches solvent, so the endpoint distance is set
> from the molecule's span by default.
"""),
        code(params_block(entries, colab, "Exit direction") + '''

import numpy as np
PROTEIN_COORDS = L.protein_heavy_coords(START_PDB, exclude_chain=LIGAND_CHAIN)
min_dist = MIN_EXIT_DISTANCE or max(15.0, LIGAND_SPAN / 2 + 10.0)
if not MIN_EXIT_DISTANCE:
    print(f"MIN_EXIT_DISTANCE auto: {min_dist:.0f} A "
          f"(span {LIGAND_SPAN:.1f} A / 2 + 10 A, floor 15 A)\\n")
''' + demo_hint + '''
RAYS = None
if EXIT_MODE == "use_file":
    GUIDE_PATH = os.path.abspath(os.path.expanduser(GUIDE_PATH_FILE))
    if not os.path.isfile(GUIDE_PATH):
        raise FileNotFoundError(GUIDE_PATH)
    pts = L.read_guide_points(GUIDE_PATH)
    print(f"using your path: {len(pts)} points")
    print()
    # A tunnel finder does not guarantee an orientation, and LPF overwrites point 0
    # with the ligand's real position — so a path exported outside-in is silently
    # mangled rather than rejected.
    pts, was_reversed = L.check_path_direction(pts, START_NBR,
                                               auto_reverse=REVERSE_PATH)
    if was_reversed and not REVERSE_PATH:
        raise ValueError("the supplied path runs the wrong way (see above). Set "
                         "REVERSE_PATH=True to flip it, or export it site-to-outside.")
    if was_reversed:
        GUIDE_PATH = L.write_path_points(os.path.join(PROJECT_DIR, "path_reversed"), pts)
    print()
    L.path_clearance_report(pts, PROTEIN_COORDS, ligand_radius=LIGAND_SPAN / 2)

elif EXIT_MODE == "manual":
    end = np.array([float(v) for v in EXIT_POINT.split(",")])
    GUIDE_PATH = L.write_two_point_path(os.path.join(PROJECT_DIR, "path"),
                                        START_NBR, end)
else:
    hint = ([float(v) for v in HINT_VECTOR.split(",")] if HINT_VECTOR.strip() else None)
    RAYS = L.exit_direction_scan(
        PROTEIN_COORDS, START_NBR, n_dirs=N_SCAN_DIRECTIONS,
        hint=hint, hint_cone_deg=HINT_CONE_DEG if hint else None, top=10)
    ray = RAYS[int(RAY_INDEX)]
    print(f"\\nchosen ray #{RAY_INDEX}: {ray}")
    if ray.bottleneck < LIGAND_SPAN / 4:
        print(f"[warn] bottleneck {ray.bottleneck:.1f} A is tight for a molecule "
              f"spanning {LIGAND_SPAN:.1f} A — the protein will have to open up. "
              f"That is legitimate (and what memory_mode is for) but it makes the "
              f"run harder; try another RAY_INDEX if one is wider.")
    print()
    end = L.exit_point_along(PROTEIN_COORDS, START_NBR, ray.direction,
                             clearance=EXIT_CLEARANCE, min_dist=min_dist,
                             ligand_span=LIGAND_SPAN)
    GUIDE_PATH = L.write_two_point_path(os.path.join(PROJECT_DIR, "path"),
                                        START_NBR, end)
'''),
        code(f'''
# 3D check: protein surface, the {what}, and the proposed exit vector
try:
    import py3Dmol
    v = py3Dmol.view(width=820, height=520)
    v.addModel(open(START_PDB).read(), "pdb")
    v.setStyle({{"cartoon": {{"color": "lightgrey"}}}})
    v.setStyle({{"chain": LIGAND_CHAIN}}, {{"stick": {{"radius": 0.2, "colorscheme": "cyanCarbon"}}}})
    v.addSphere({{"center": {{"x": float(START_NBR[0]), "y": float(START_NBR[1]),
                            "z": float(START_NBR[2])}},
                 "radius": 0.8, "color": "blue"}})
    if EXIT_MODE != "use_file":
        v.addArrow({{"start": {{"x": float(START_NBR[0]), "y": float(START_NBR[1]),
                              "z": float(START_NBR[2])}},
                    "end": {{"x": float(end[0]), "y": float(end[1]), "z": float(end[2])}},
                    "radius": 0.25, "color": "red"}})
        print("blue sphere = tracked atom (path start), red arrow = exit direction")
    v.addSurface("VDW", {{"opacity": 0.55, "color": "white"}},
                 {{"chain": LIGAND_CHAIN, "invert": 1}})
    v.zoomTo({{"chain": LIGAND_CHAIN}})
    v.show()
except ImportError:
    print("py3Dmol not installed — skipping the 3D view (pip install py3Dmol)")
'''),
    ]


# ==========================================================================
# 10. protocols + options
# ==========================================================================

def cells_protocols(kind, env):
    colab = env == "colab"
    is_pep = kind == "peptide"
    box = 8.0 if is_pep else 5.0
    tuning = "L.PEPTIDE_XML_TUNING" if is_pep else "L.LIGAND_XML_TUNING"
    entries = [
        dict(name="GRID_WIDTH", value=0.0, kind="number",
             comment="0 = size it from the molecule's span"),
        dict(name="TRANSFORM_BOX_SIZE", value=box, kind="number",
             comment="Transform translation box; also feeds the grid size"),
        dict(name="EXTRA_PARAMS_FILES", value="",
             comment="cofactor .params paths, comma-separated"),
        dict(name="AUTO_SETUP_METALS", value=False,
             comment="adds -in:auto_setup_metals"),
        dict(name="APPLY_TUNING", value=is_pep,
             comment=("peptide-scaled protocol tuning" if is_pep
                      else "leave False to keep the upstream protocols verbatim")),
    ]
    why = ("""
### What the peptide tuning changes, and why

| change | from → to | reason |
|---|---|---|
| `SCORINGGRIDS width` | 20 → sized from the molecule | the grid is centred on the ligand and nothing outside it is scored; 20 Å does not contain a peptide |
| `LigandArea cutoff` | 6/6/7 → 8/8/10 Å | a peptide contacts far more of the protein, and only residues inside these cutoffs may repack or move |
| `Calpha_restraints` (main) | 0.3 → 0.5 | this is a standard deviation, so a larger value lets the backbone actually open; left at 0.1 in `zero_minimize.xml`, which only settles the start pose |
| `DetectProteinLigandInterface` cuts | 6/6/6/6 → 7/9/11/13 Å | ditto, for the repacking shell |
| `Transform box_size` | 5 → 8 Å | a bigger body needs a bigger translation box |
| `Transform cycles/repeats` | 500/1 → 500/2 | more sampling per window |
| `HighResDocker cycles` | 9 → 15 | ditto |
| `initial_angle_perturb` (coarse) | 90 → 20° | spinning an elongated peptide freely inside a channel just makes clashes |
| `add_nbr_radius` (coarse) | true → false | the coarse protocol adds `NBR_RADIUS` to its cutoff; that is ~2 Å for a small ligand and 20+ Å for a peptide, which would drag the whole protein into every repack |

The coarse protocol's cycle counts are left alone — stage 1 is meant to be cheap.
`main_rigid.xml` and `main_extended.xml` are not written, because blanket
retuning would erase the only thing that distinguishes them from
`main_standard.xml`.
""" if is_pep else """
The upstream protocols are copied verbatim by default (`APPLY_TUNING=False`) —
they are tuned for small molecules already. Two things are still rewritten when
needed:

* **the chain**, everywhere it is hardcoded (`LigandArea`, `SCORINGGRIDS`,
  `Transform`, `StartFrom`, `InterfaceScoreCalculator`). Rosetta never reads
  `input.ligand_chain` from `run.json`, so a mismatch would silently score the
  wrong chain — LPF warns, and the preflight here treats it as an error.
* **the scoring grid width**, if your ligand is larger than the stock 20 Å grid
  comfortably holds.
""")
    return [
        md(f"""
---
## 9 · RosettaScripts protocols and the options file

Copies the stock protocols out of `rosettaXML_files/` and retargets them, rather
than shipping forks that would drift from the repository.
{why}
The options file is written fresh so that `-in:path:database` can be included
when your binary sits outside its source tree — without it every window fails
identically.
"""),
        code(params_block(entries, colab, "Protocols") + f'''

extra = [os.path.abspath(os.path.expanduser(p.strip()))
         for p in EXTRA_PARAMS_FILES.split(",") if p.strip()]
grid_w = GRID_WIDTH or L.auto_grid_width(LIGAND_SPAN, box_size=TRANSFORM_BOX_SIZE)

tuning = {tuning} if APPLY_TUNING else L.LIGAND_XML_TUNING
if APPLY_TUNING:
    tuning = dict(tuning)
    tuning["rules"] = list(tuning["rules"]) + [
        {{"tag": "Transform", "name": "transform",
         "attrs": {{"box_size": str(int(TRANSFORM_BOX_SIZE))}}, "files": L._MAIN_LIKE}},
        {{"tag": "Transform", "name": "transform_2",
         "attrs": {{"box_size": str(int(TRANSFORM_BOX_SIZE))}}, "files": L._MAIN_LIKE}},
    ]
print()
PROTOCOLS = L.write_protocol_xmls(FILES_DIR, XML_SRC, chain=LIGAND_CHAIN,
                                  tuning=tuning, grid_width=grid_w)
print()
OPTIONS = L.write_options_file(
    os.path.join(FILES_DIR, "options"),
    extra_res=["lig.params", "VRT1.params"] + [os.path.basename(p) for p in extra],
    database=ROSETTA_DATABASE or None,
    auto_setup_metals=AUTO_SETUP_METALS)
EXTRA_FILES = extra
'''),
    ]


# ==========================================================================
# 11. stage 1 — coarse path
# ==========================================================================

def cells_coarse(kind, env):
    colab = env == "colab"
    is_pep = kind == "peptide"
    pset = "L.PEPTIDE_COARSE_PARAMS" if is_pep else "L.LIGAND_COARSE_PARAMS"
    step = 3.0 if is_pep else 2.0
    peptide_table = ("""
### Peptide coarse defaults — what differs from the paper's values and why

| key | ligand | peptide | reasoning |
|---|---|---|---|
| `step` | 2.0 | 3.0 Å | each node is a full Rosetta run and a peptide's node is much dearer; 2 Å buys no extra routing detail |
| `memory_mode` | false | **true** | with it off, every node restarts from the original protein, so the channel never opens and a peptide-sized body may find no route at all. The cost is that induced distortion accumulates along a stochastic run — which is itself a reason to run replicates |
| `conformer_subsampling` | false | **true** (4.0 Å) | the full library is large and mostly irrelevant to the current pose; the threshold is generous so the ladder stays connected |
| `rmsd_constraint_sd` | 1.5 | 2.0 Å | the effective weight is `sd·√N`, i.e. already normalised for atom count, so this really is a looser shape restraint: the peptide has to change conformation to get out |
| `plane_constraint_point_extent` | 7 | 12 Å | circumradius of the virtual triangle defining the guide plane; it must exceed the molecule or the three dihedrals holding it in-plane are badly conditioned |
| `angle_constraint_bound` | 2.25 | 2.00 rad | an elongated body cannot keep its tracked atom as straight |
""" if is_pep else """
These are the upstream demo values verbatim. The authors explicitly recommend
leaving the coarse parameters as they are, so change them only with a reason.
""")
    entries = [
        dict(name="DOCK_ZERO_COARSE", value=False,
             comment="dock the ligand onto the path's first point before stage 1"),
        dict(name="APPLY_LARGE_LIGAND_RULE_COARSE", value=True,
             comment="the paper's >50-atom rule (conformer_rmsd_threshold 5.0)"),
        dict(name="COARSE_STEP", value=step, kind="number",
             comment="bisection stops when neighbouring nodes are closer than this"),
        dict(name="COARSE_NSTRUCT", value=10, kind="integer",
             comment="Rosetta decoys per node"),
        dict(name="N_REPLICATES", value=1, kind="integer",
             comment="independent coarse runs; the medoid path goes forward"),
        dict(name="ENDPOINT_JITTER_RADIUS", value=5.0, kind="number",
             comment="each replicate lands at a slightly different endpoint"),
        dict(name="SEC_PER_DECOY_GUESS", value=16.0, kind="number",
             comment="cost estimate only; the paper's benchmark average — recalibrate from your first node"),
        dict(name="CLEAN_PREVIOUS", value=True,
             comment="the coarse stage cannot resume, so re-runs need a clean directory"),
    ]
    return [
        md(f"""
---
## 10 · Stage 1 — coarse path

Recursive bisection between the two guide points: Rosetta places the
{"peptide" if is_pep else "ligand"} at the midpoint under the plane / in-plane /
angle constraints, the midpoint moves to wherever the tracked atom actually
ended up, and the halves are subdivided until neighbouring points are closer
than `step`. Output: **`coarsepath.txt`**.

It is stochastic and not guaranteed to converge — it depends on the endpoint you
chose. Several replicates with `endpoint_jitter` on, then the medoid, is the
recommended practice.

Skip this entire section if you set `EXIT_MODE='use_file'` above.
{peptide_table}"""),
        code(params_block(entries, colab, "Stage 1 parameters") + f'''

coarse_params = dict({pset})
if APPLY_LARGE_LIGAND_RULE_COARSE:
    coarse_params = L.apply_large_ligand_rule(
        coarse_params, len(L.params_atom_names(LIG_PARAMS)))
    print()
coarse_params["step"] = COARSE_STEP
coarse_params["dock_zero"] = DOCK_ZERO_COARSE
coarse_params["endpoint_jitter"] = N_REPLICATES > 1 or coarse_params["endpoint_jitter"]
coarse_params["endpoint_jitter_radius"] = ENDPOINT_JITTER_RADIUS

print("coarse parameters:")
for k, v in sorted(coarse_params.items()):
    print(f"   {{k:34s}} {{v}}")

_gp = L.read_guide_points(GUIDE_PATH)
if not L.check_path_anchor(_gp, START_NBR, dock_zero=coarse_params.get("dock_zero")):
    raise ValueError("the guide path and the start pose disagree (see above). "
                     "Set DOCK_ZERO_COARSE=True, or fix the path / start pose.")
print()
guide_len = float(np.linalg.norm(_gp[-1] - _gp[0])) \\
    if EXIT_MODE != "use_file" else 0.0
n_nodes = max(1, int(guide_len / COARSE_STEP)) if guide_len else 15
print(f"\\nguide segment {{guide_len:.1f}} A -> roughly {{n_nodes}} nodes per replicate")
L.estimate_cost(n_nodes, COARSE_NSTRUCT, SEC_PER_DECOY_GUESS, N_PROC, label="node")
if N_REPLICATES > 1:
    print(f"x {{N_REPLICATES}} replicates")
'''),
        code('''
# One coarse run per replicate, each in its own directory.
COARSE_RUNS, COARSE_PATHS = [], []
if EXIT_MODE == "use_file":
    print("EXIT_MODE='use_file' — skipping stage 1; your path goes straight to stage 2.")
else:
    for rep in range(int(N_REPLICATES)):
        run_dir = os.path.join(PROJECT_DIR, f"coarse_{rep:02d}")
        L.prepare_run_dir(run_dir, clean=CLEAN_PREVIOUS)
        cfg = L.build_run_config(
            "coarse", ROSETTA_SCRIPTS, FILES_DIR, LIGAND_DIR, GUIDE_PATH, run_dir,
            start_pdb=START_PDB, header_pdb=HEADER_PDB, ligand_chain=LIGAND_CHAIN,
            params=coarse_params, nstruct=COARSE_NSTRUCT, n_proc=N_PROC,
            extra_files=EXTRA_FILES, rosetta_database=ROSETTA_DATABASE or None,
            archive=ARCHIVE_WINDOWS)
        cfg_path = L.write_run_config(os.path.join(run_dir, "run.json"), cfg)

        errs, warns = L.preflight_config(cfg_path, "coarse")
        if errs:
            raise RuntimeError(f"preflight found {len(errs)} problem(s) — fix them above")

        print(f"\\n{'=' * 64}\\nreplicate {rep + 1}/{N_REPLICATES}\\n{'=' * 64}")
        L.run_lpf_stage("coarse", LPF_REPO, cfg_path, run_dir,
                        log_path=os.path.join(run_dir, "coarse.log"))
        COARSE_RUNS.append(run_dir)
        COARSE_PATHS.append(os.path.join(run_dir, "coarsepath.txt"))
'''),
        code('''
# Inspect the coarse result(s) and choose the path for stage 2.
if EXIT_MODE == "use_file":
    print("using your supplied path:", GUIDE_PATH)
    TRAJ_PATH = GUIDE_PATH
else:
    for rd in COARSE_RUNS:
        print("=" * 64)
        L.summarize_run(rd, stage="coarse")
        print()
    TRAJ_PATH, k, _D = (L.coarse_path_medoid(COARSE_PATHS) if len(COARSE_PATHS) > 1
                        else (COARSE_PATHS[0], 0, None))
    print("\\npath going forward to stage 2:", TRAJ_PATH)

pts = L.read_guide_points(TRAJ_PATH)
print()
L.path_clearance_report(pts, PROTEIN_COORDS, ligand_radius=LIGAND_SPAN / 2)
if len(pts) < 4:
    raise ValueError(
        f"the path has only {len(pts)} points. The trajectory stage fits a cubic "
        f"smoothing spline and needs at least 4 — lower COARSE_STEP, or supply a "
        f"denser path.")

try:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6.4, 5))
    ax = fig.add_subplot(111, projection="3d")
    for i, p in enumerate(COARSE_PATHS):
        q = np.loadtxt(p)
        ax.plot(*q.T, "-o", ms=2.5, lw=1.0, alpha=0.9 if p == TRAJ_PATH else 0.35,
                label=f"replicate {i}" + (" (medoid)" if p == TRAJ_PATH else ""))
    ax.scatter(*np.atleast_2d(START_NBR).T, c="k", s=60, label="bound pose")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("coarse path(s)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
except Exception as exc:
    print("(3D plot skipped:", exc, ")")
'''),
    ]


# ==========================================================================
# 12. stage 2 — trajectory
# ==========================================================================

def cells_trajectory(kind, env):
    colab = env == "colab"
    is_pep = kind == "peptide"
    pset = "L.PEPTIDE_TRAJ_PARAMS" if is_pep else "L.LIGAND_TRAJ_PARAMS"
    step = 1.0 if is_pep else 0.5
    smooth = 1.0 if is_pep else 0.5
    ckpt = 5 if colab else 10
    peptide_table = ("""
### Peptide trajectory defaults — what differs and why

| key | ligand | peptide | reasoning |
|---|---|---|---|
| `step` | 0.5 | 1.0 Å | halves the window count, and each peptide window is already several times dearer. Drop it back towards 0.5 if the conformation jumps between windows |
| `smooth` | 0.5 | 1.0 | a long body cannot take sharp corners; the script prints the max turning angle and what 2× / 5× smoothing would give |
| `conformer_rmsd_threshold` | 1.5 | 3.0 Å | a peptide's conformers differ far more; 1.5 Å would leave it near-rigid |
| `rmsd_constraint_sd` | 0.75 | 1.25 Å | looser shape restraint (the weight is already `sd·√N`) |
| `rmsd_constraint_hydrogens` | true | **false** | hydrogens roughly double the atom count and add stiffness without shape information |
| `ipdc_d0` / `ipdc_sd` | 3.0 / 0.5 | 4.0 / 0.75 Å | the tracked atom of a long molecule legitimately sits further off the guide point while the rest threads through |
| `end_padding` | 5 | 10 Å | more room to disengage; the in-plane restraint relaxes over the last `end_padding/step` windows |
| `angle_constraint_bound` | 2.00 | 1.80 rad | ditto, direction restraints |
| `next_angle_constraint_bound` | 1.57 | 1.40 rad | ditto |
| `outlier_rmsd` | 2.0 | 3.0 Å | 2.0 Å would reject most decoys of a large, genuinely mobile body |
""" if is_pep else """
These are the upstream demo values verbatim.
""")
    resume = ("""
> **Colab sessions end.** `checkpoint_interval` is set low and the run resumes
> automatically from `checkpoint.pkl` — just re-run the cell after reconnecting
> (and keep `PROJECT_DIR` on Drive). `FRESH_START=True` overrides and begins again.
""" if colab else """
> The run auto-resumes from `checkpoint.pkl` if it was interrupted — re-run the
> cell. `FRESH_START=True` ignores the checkpoint and starts over.
""")
    entries = [
        dict(name="PARAM_PRESET", value="auto", options=["auto", "benchmark"],
             comment="'benchmark' = the settings from the paper's 136-system benchmark"),
        dict(name="APPLY_LARGE_LIGAND_RULE", value=True,
             comment="the paper's >50-atom rule: conformer_rmsd_threshold and outlier_rmsd both 5.0"),
        dict(name="TRAJ_STEP", value=step, kind="number", comment="window spacing in A"),
        dict(name="TRAJ_SMOOTH", value=smooth, kind="number",
             comment="spline smoothing; larger = smoother, less faithful"),
        dict(name="TRAJ_NSTRUCT", value=10, kind="integer", comment="decoys per window"),
        dict(name="RUN_STATS", value=False,
             comment="second independent scoring pass per window = error bars"),
        dict(name="NSTRUCT_STATS", value=10, kind="integer"),
        dict(name="DOCK_ZERO", value=False,
             comment="dock onto the path's first point before starting (ligand present but unplaced)"),
        dict(name="CHECKPOINT_INTERVAL", value=ckpt, kind="integer",
             comment="0 disables; the run resumes from checkpoint.pkl"),
        dict(name="USE_TMP", value=False,
             comment="run each window on a RAM disc, copy results back"),
        dict(name="FRESH_START", value=False,
             comment="ignore an existing checkpoint"),
        dict(name="SEC_PER_DECOY_GUESS", value=16.0, kind="number",
             comment="cost estimate only; the paper's benchmark average — recalibrate from your first window"),
    ]
    return [
        md(f"""
---
## 11 · Stage 2 — trajectory

The guide path is fitted with a smoothing spline, resampled at `step` spacing,
and one Rosetta run is done per window, each starting from the previous window's
result. Output: **`interface_scores.txt`** — the `interface_delta_X` binding
energy per window, which is the result you analyse.

{resume}{peptide_table}"""),
        code(params_block(entries, colab, "Stage 2 parameters") + f'''

traj_params = dict(L.BENCHMARK_TRAJ_PARAMS if PARAM_PRESET == "benchmark"
                   else {pset})
if PARAM_PRESET == "benchmark":
    print("using the paper's benchmark settings (SI, Geometry of Tunnels): "
          "smooth 5.0,\\nrmsd sd 0.66, ipdc d0 2.0, angle 2.5/0.5 — these suit an "
          "external CAVER path.\\n")
# The paper loosens two keys for ligands over 50 atoms: conformer_rmsd_threshold
# and outlier_rmsd both to 5.0 (SI, benchmark settings). Every peptide trips this.
if APPLY_LARGE_LIGAND_RULE:
    traj_params = L.apply_large_ligand_rule(
        traj_params, len(L.params_atom_names(LIG_PARAMS)))
    print()
traj_params["step"] = TRAJ_STEP
traj_params["smooth"] = TRAJ_SMOOTH
traj_params["run_stats"] = RUN_STATS
traj_params["dock_zero"] = DOCK_ZERO

print("trajectory parameters:")
for k, v in sorted(traj_params.items()):
    print(f"   {{k:34s}} {{v}}")

pts = L.read_guide_points(TRAJ_PATH)
# Unless dock_zero is on, LPF moves the path's first point onto the ligand. If the
# path starts somewhere else entirely that leaves a huge first segment and a
# meaningless spline, so check the two agree before paying for a run.
if not L.check_path_anchor(pts, START_NBR, dock_zero=traj_params.get("dock_zero")):
    raise ValueError("the guide path and the start pose disagree (see above). "
                     "Set DOCK_ZERO=True, or fix the path / start pose.")
print()
path_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
n_windows = max(1, int(path_len / TRAJ_STEP))
print(f"\\nguide path {{path_len:.1f}} A -> roughly {{n_windows}} windows at "
      f"step={{TRAJ_STEP}} A")
L.estimate_cost(n_windows, TRAJ_NSTRUCT, SEC_PER_DECOY_GUESS, N_PROC)
if RUN_STATS:
    print("plus the statistics pass:")
    L.estimate_cost(n_windows, NSTRUCT_STATS, SEC_PER_DECOY_GUESS, N_PROC)
'''),
        code('''
TRAJ_RUN = os.path.join(PROJECT_DIR, "trajectory")
L.prepare_run_dir(TRAJ_RUN, clean=False)   # keep the checkpoint so a stopped run resumes

cfg = L.build_run_config(
    "trajectory", ROSETTA_SCRIPTS, FILES_DIR, LIGAND_DIR, TRAJ_PATH, TRAJ_RUN,
    start_pdb=START_PDB, header_pdb=HEADER_PDB, ligand_chain=LIGAND_CHAIN,
    params=traj_params, nstruct=TRAJ_NSTRUCT, nstruct_stats=NSTRUCT_STATS,
    n_proc=N_PROC, extra_files=EXTRA_FILES,
    rosetta_database=ROSETTA_DATABASE or None, archive=ARCHIVE_WINDOWS,
    checkpoint_interval=CHECKPOINT_INTERVAL, use_tmp=USE_TMP)
cfg_path = L.write_run_config(os.path.join(TRAJ_RUN, "run.json"), cfg)

errs, warns = L.preflight_config(cfg_path, "trajectory")
if errs:
    raise RuntimeError(f"preflight found {len(errs)} problem(s) — fix them above")
'''),
        code('''
# The long one. Progress is streamed; each window prints its constraint energies
# and how far the tracked atom ended up from the guide plane (d_to_plane) and the
# guide point (d_in_plane). Small values mean the constraints are holding.
L.run_lpf_stage("trajectory", LPF_REPO, cfg_path, TRAJ_RUN,
                fresh=FRESH_START,
                log_path=os.path.join(TRAJ_RUN, "trajectory.log"))
'''),
    ]


# ==========================================================================
# 13. analysis + export
# ==========================================================================

def cells_analysis(kind, env):
    colab = env == "colab"
    is_pep = kind == "peptide"
    what = "peptide" if is_pep else "ligand"
    download = ('''
try:
    from google.colab import files
    files.download(zip_path)
except Exception as exc:
    print("(automatic download unavailable:", exc, ")")
    print("The archive is on Drive/disk at the path above.")
''' if colab else '''
print("\\nOpen the trajectory in PyMOL with:")
print(f"  pymol {os.path.join(TRAJ_RUN, 'full_trajectory.pdb')}")
print("  # then:  set all_states, off ;  mset 1 x<nframes> ;  mplay")
''')
    return [
        md(f"""
---
## 12 · Results

| file | what it is |
|---|---|
| **`interface_scores.txt`** | `interface_delta_X` per scored window — **the main result** |
| `nbr_trajectory.txt` | tracked atom's position per window |
| `full_trajectory.pdb` | protein + {what} along the path, one frame per window |
| `lig_trajectory.pdb` | the {what} alone |
| `total_scores.txt` | whole-complex energy (mostly for debugging) |
| `stats.pkl` | per-window score distributions, if `run_stats` was on |
| `guide_path`, `spline`, `normals`, `input_path` | path construction intermediates |

Constraint energies are subtracted before any decoy is ranked or reported, so
these numbers are not contaminated by the restraints that produced them.

**One replicate is noisy.** Read the shape, not the digits; pool replicates
before quoting numbers (appendix).
"""),
        code('''
print("=" * 64)
info = L.summarize_run(TRAJ_RUN, stage="trajectory")
'''),
        code('''
import matplotlib.pyplot as plt
ax = L.plot_energy_profile(TRAJ_RUN, x="distance", show_stats=True,
                           smooth_window=0)
plt.show()

d = L.load_run_scores(TRAJ_RUN)
y = d["interface_scores"]
if y is not None and len(y):
    bound, free = float(y[0]), float(y[-1])
    print(f"window 0 (bound)   {bound:+.2f} REU")
    print(f"last window (free) {free:+.2f} REU")
    print(f"deepest minimum    {y.min():+.2f} REU at window {int(y.argmin())}")
    print(f"energy released along this path: {free - y.min():+.2f} REU")
    print()
    print("Local minima along the way are where the " + "''' + what + '''" +
          " pauses against")
    print("gateway residues. To find out which residues, look at the window PDBs:")
    print(f"  {TRAJ_RUN}/window_XXXX/best.pdb   (zipped if archive was on)")
'''),
        code('''
# Overlay: the protein once, every pose of the ''' + what + ''' coloured blue (bound)
# to red (released). mode="frames" animates instead.
try:
    L.view_trajectory(TRAJ_RUN, ligand_chain=LIGAND_CHAIN, mode="overlay", stride=1)
except ImportError:
    print("py3Dmol not installed (pip install py3Dmol)")
except Exception as exc:
    print("3D view unavailable:", exc)
'''),
        md("### Pooling replicates: the paper's clustering\n\nOne replicate is noisy; the paper's results come from an ensemble. Its recipe\n(SI, *Clustering of sampled paths*) is implemented here:\n\n1. the distance between two trajectories is the **frame-by-frame coordinate RMSD\n   of the ligand**, summed over frames with **linearly decreasing weights** — so\n   differences while the ligand is still inside the tunnel count for more than\n   differences out in bulk solvent;\n2. where one path is shorter, each missing frame is charged a fixed **0.5 Å**;\n3. the **negative** of that matrix goes to **affinity propagation**\n   (scikit-learn, default parameters) as a precomputed affinity;\n4. the interface profile is **averaged within each cluster**;\n5. clusters of **10 or fewer** paths are discarded;\n6. the most populated cluster is drawn bold, the rest thin and semi-transparent.\n\nThis needs several finished trajectory runs — the paper pooled about 30. Produce\nthem by re-running section 11 with a different `RUN_NAME`, or on a cluster with\nthe array job in the appendix. With only a handful, lower `MIN_CLUSTER_SIZE` or\nnothing will survive step 5."),
        code(params_block([
            dict(name="REPLICATE_GLOB", value="",
                 comment="finished trajectory runs, or a glob; empty = repl-*/ and trajectory*/ in the project"),
            dict(name="MIN_CLUSTER_SIZE", value=10, kind="integer",
                 comment="discard clusters with this many members or fewer (10 is the paper's value)"),
            dict(name="MISSING_PENALTY", value=0.5, kind="number",
                 comment="distance charged per missing frame (0.5 is the paper's value)"),
        ], colab, "Cluster pooled replicates") + '\nimport glob as _glob\nruns = sorted(_glob.glob(os.path.expanduser(REPLICATE_GLOB))) if REPLICATE_GLOB \\\n    else (sorted(_glob.glob(os.path.join(PROJECT_DIR, "repl-*")))\n          + sorted(_glob.glob(os.path.join(PROJECT_DIR, "trajectory*"))))\nruns = [r for r in runs if os.path.isdir(r)]\n\nif len(runs) < 2:\n    print(f"found {len(runs)} run(s) - clustering needs at least two.")\n    print("Re-run the trajectory section with a different RUN_NAME, or use the")\n    print("cluster array job in the appendix, then come back here.")\nelse:\n    clusters = L.cluster_trajectories(\n        runs, ligand_chain=LIGAND_CHAIN, missing_penalty=MISSING_PENALTY,\n        min_cluster_size=MIN_CLUSTER_SIZE, plot=True)\n    print()\n    for c in clusters["kept"]:\n        m = clusters["means"][c]\n        print(f"  cluster {c}: n={m[\'n\']:3d}  exemplar {m[\'exemplar\']}  "\n              f"bound {m[\'mean\'][0]:+.2f} REU  final {m[\'mean\'][-1]:+.2f} REU")\n    plt.show()\n'),
        md("### If `interface_scores.txt` came out all zeros\n\nThat is not a result — it means Rosetta's `InterfaceScoreCalculator` contributed\nno `interface_delta_<chain>` and the column was written as a placeholder. The\nposes themselves are fine, so the profile can be recomputed without redoing any\ndocking: each window's selected pose is rescored once with upstream's\nscoring-only `zero_score.xml`. One Rosetta call per window instead of `nstruct`\ndocking runs — minutes, not hours. Archived windows are read out of their zips,\nand the original file is left untouched so you can compare."),
        code(params_block([
            dict(name="RUN_RESCORE", value=False,
                 comment="recompute the interface energies from the saved poses"),
            dict(name="RESCORE_RUN", value="",
                 comment="run directory to rescore; empty = the one just run"),
            dict(name="INSTALL_RESCORED", value=True,
                 comment="replace interface_scores.txt so the plots and clustering use it"),
        ], colab, "Rescore a zeroed run") + '\nif "RESCORE_RUN" not in dir():\n    RESCORE_RUN = ""\ntarget = RESCORE_RUN or TRAJ_RUN\n_scores = L.load_run_scores(target)["interface_scores"]\nif _scores is not None and len(_scores) and not np.allclose(_scores, 0.0):\n    print("This run already has real interface energies — nothing to rescore.")\n    print(f"  bound {_scores[0]:+.2f} REU, final {_scores[-1]:+.2f} REU")\nelif RUN_RESCORE:\n    # zero_score.xml is upstream\'s scoring-only protocol: one\n    # InterfaceScoreCalculator and nothing else.\n    L.write_protocol_xmls(FILES_DIR, XML_SRC, chain=LIGAND_CHAIN,\n                          files=("zero_score.xml",), verbose=False)\n    print()\n    rescored = L.rescore_windows(\n        target, rosetta=ROSETTA_SCRIPTS, files_dir=FILES_DIR,\n        ligand_dir=LIGAND_DIR, ligand_chain=LIGAND_CHAIN,\n        extra_files=EXTRA_FILES, install=INSTALL_RESCORED)\nelse:\n    print("interface_scores.txt is all zeros, which is not a result.")\n    print("Set RUN_RESCORE=True to recompute it from the saved poses — one")\n    print("scoring call per window, no re-docking.")\n'),
        md("### How many replicates do you need?\n\nMeasured rather than guessed. For each ensemble size *n*, bootstrap resamples of\n*n* runs are drawn and the mean profile computed for each; the spread of those\nmeans is how much your answer would still move if you had run a different *n*.\nTo resolve a feature of size *D* REU you want that spread comfortably below *D*.\n\nIt falls as 1/√n, drawn as a reference. A curve sitting **above** the reference\nmeans the replicates disagree structurally — different exit routes — and the\nanswer is to cluster them, not to average harder.\n\nFor scale: the paper's tutorial runs **30** replicates in its worked example, and\nits clustering discards clusters of 10 or fewer paths, so an ensemble intended\nfor clustering has to be a good deal larger than 10. The SI does not state the\nper-system counts behind the published figures."),
        code(params_block([
            dict(name="CONVERGENCE_GLOB", value="",
                 comment="replicate run dirs, or a glob; empty = repl-*/ in the project"),
        ], colab, "Replicate convergence") + '\nimport glob as _g\nruns = sorted(_g.glob(os.path.expanduser(CONVERGENCE_GLOB))) if CONVERGENCE_GLOB \\\n    else sorted(_g.glob(os.path.join(PROJECT_DIR, "repl-*")))\nruns = [r for r in runs if os.path.isdir(r)]\nif len(runs) < 3:\n    print(f"found {len(runs)} replicate(s); this needs at least 3.")\n    print("Produce them by re-running the trajectory section with a different")\n    print("RUN_NAME (repl-00, repl-01, ...), or with the sbatch array in the")\n    print("appendix, then come back here.")\nelse:\n    conv = L.replicate_convergence(runs, step=traj_params["step"], plot=True)\n    plt.show()\n'),
        code('''
# Package the analysis files. include_windows=True also takes every window
# directory, which is where the decoy archives are — usually hundreds of MB.
zip_path = L.zip_results(TRAJ_RUN, os.path.join(PROJECT_DIR, "trajectory_results.zip"),
                         include_windows=False)
''' + download),
    ]


# ==========================================================================
# 14. appendix
# ==========================================================================

def cell_appendix(kind, env):
    is_pep = kind == "peptide"
    what = "peptide" if is_pep else "ligand"
    pep_tuning = ("""
### Peptide-specific things that go wrong

**The conformer library is disconnected.** The report in step 5 said `NO`. The
peptide cannot walk from its bound conformation to the extended ones, so it
leaves as a rigid body or not at all. Raise `N_MEDIUM` and `SIGMA_MEDIUM_DEG`,
lower `CONFORMER_PRUNE_RMSD`, or raise `conformer_rmsd_threshold`.

**Only 1–2 rotamers survive subsampling.** `conformer_subsampling_preview` in
step 7 tells you this before you spend anything. Raise
`conformer_rmsd_threshold`, or build a denser library.

**`molfile_to_params` refuses the peptide.** You are past its practical range.
Shorten the peptide, or treat the problem as a protein–protein unbinding
question with a different tool.

**The tracked atom is a terminus.** Check the `tracking:` line in step 6. A
dangling terminus makes the guide path describe the motion of one tip.

**The scoring grid is smaller than the peptide.** Step 9 sizes it from the
molecule's span; if you set `GRID_WIDTH` by hand, keep it above
`span + 2*box_size`.

**Window 0 has a huge energy.** The start pose is not relaxed into the Rosetta
force field. Minimise the complex first (`demo/tutorial.md` 1.3).
""" if is_pep else "")
    return md(f"""
---
## Appendix

### Retuning, from `demo/tutorial.md` section 5

**The {what} is too stiff — it moves as a rigid body.**
1. Check the conformer library actually holds distinct conformations.
2. Raise `conformer_rmsd_threshold`.
3. Raise `rmsd_constraint_sd`.
4. Set `rmsd_constraint_hydrogens` false.
5. Lower `angle_constraint_bound`.

**The {what} changes conformation too sharply between windows.** The reverse of
the above, and lower `step`.

**The protein is not flexible enough.**
1. Check `memory_mode` is true.
2. Raise `Calpha_restraints` in `main_standard.xml` (it is a standard deviation,
   so larger is looser).
3. Raise the `LigandArea` cutoffs.

**The protein takes weird conformations.** The reverse of the above.

**A window stalls or the {what} cuts a corner.** The trajectory script prints
the guide path's maximum turning angle at start-up and warns above 25°. Raise
`smooth`, or run more coarse replicates and take a straighter medoid.
{pep_tuning}
### If you are using the PyRosetta backend

**Every window fails identically.** Run the launcher by hand in a window
directory to see the real error:
`cd trajectory/window_0001 && ./pyrosetta_backend/rosetta_scripts.pyrosetta @options -parser:protocol task.xml -in:file:s task.pdb -nstruct 1 -out:path:pdb ./results`.
Set `LPF_SHIM_VERBOSE=1` to keep PyRosetta's own tracer output instead of muting it.

**It is slower than you expected.** The shim runs `nstruct` decoys in one
process by default. Raise `SHIM_JOBS` (step 2) to spread them over worker
processes — each one initialises its own PyRosetta, so budget 1-2 GB of RAM
apiece and do not exceed `nstruct`. `rosetta_scripts.n_proc` must stay at 1;
that path is MPI and the shim does not implement it.

**A protocol works with the binary but not the shim.** The shim builds movers
through PyRosetta's RosettaScripts parser, whose entry points differ between
releases; it tries the known ones in order and prints which it used. If none
work with your build, report what it printed — and fall back to a real binary,
which is the reference implementation.

### Replicates on a cluster

A single replicate is noisy. Pool `interface_scores.txt` across independent
trajectory runs for a converged profile.

`lpf_run.py` (shipped next to this notebook) is the command-line version of
everything above, and is easier to drive from a queue: `prep` writes
`project.json`, and each later stage reads it, so they can be separate jobs.

```bash
#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH -t 720
lpf_run.py traj --project /path/to/project \
    --run-name repl-$SLURM_ARRAY_TASK_ID --traj-nstruct 10
```

```bash
lpf_run.py prep ... --project /path/to/project   # once
lpf_run.py coarse --project /path/to/project --replicates 10
sbatch -J lpf -a 0-29 run.bash                   # 30 trajectory replicates
lpf_run.py cluster --project /path/to/project --convergence
```

On one machine, `lpf_run.py traj --traj-replicates 30` runs them serially
instead. **How many you need** is measurable rather than a guess — the
convergence cell above bootstraps your runs and reports how much the pooled mean
still moves as replicates are added. For scale, the tutorial's worked example
uses 30, and its clustering discards clusters of 10 or fewer paths; the SI does
not give the per-system counts behind the published figures.

Or drive the upstream script directly, with `run.json` one level up:

```bash
RUNDIR=repl-$SLURM_ARRAY_TASK_ID
mkdir -p $RUNDIR && cp run.json $RUNDIR && cd $RUNDIR
python /path/to/LigandPathFinder/lpf/lpf_trajectory.py run.json
```

Then pool them:

```python
import numpy as np, glob
runs = sorted(glob.glob("repl-*/interface_scores.txt"))
profiles = [np.loadtxt(f) for f in runs]
n = min(map(len, profiles))
A = np.stack([p[:n] for p in profiles])
mean, sem = A.mean(0), A.std(0) / np.sqrt(len(A))
```

### Reference and licences

* Method: {PAPER}
* Code: [`meilerlab/LigandPathFinder`]({REPO}) — MIT, covering the LPF code only.
* Every `run.json` key: `docs/config_reference.md` in the repository. Keys each
  stage ignores: `docs/unused_config_settings.md`.
* Reference runs and the paper's dataset:
  [10.5281/zenodo.23183178](https://doi.org/10.5281/zenodo.23183178).
* **Rosetta is licensed separately** and is not included. Free for academic and
  non-commercial use; commercial use needs its own licence.
  `molfile_to_params.py` and the `rosetta_py` modules this notebook can download
  are Rosetta files, covered by that same licence.

### What this notebook adds on top of the upstream code

Nothing here patches LPF; it all ends up as input files the upstream scripts
read. The non-obvious parts:

* `start.pdb` assembly that enforces LPF's unwritten requirements (ligand last,
  HETATM, one residue, names matching `lig.params`) and checks them;
* protocol XMLs retargeted from `rosettaXML_files/` to your chain and your
  molecule's size, instead of forked copies;
* an options file that can carry `-in:path:database`, for a binary moved out of
  its source tree;
* `pyrosetta_rosetta_scripts.py`, a `rosetta_scripts`-compatible front end so a
  pip-installed PyRosetta can serve as the backend — LPF execs the binary and
  parses `score.sc`, so the shim reproduces that command line and that file
  format rather than patching LPF;
* a preflight that reports every configuration problem at once rather than
  failing on the first one mid-run — including the two that otherwise waste a
  whole run: a guide path with fewer than 4 points (the trajectory spline needs
  4) and a protocol/`ligand_chain` mismatch (Rosetta never reads
  `input.ligand_chain`);
* exit-direction scanning with bottleneck widths, so the guide points are chosen
  against the geometry rather than by eye;
* `NBR_ATOM` re-selection with a recomputed `NBR_RADIUS`;
* conformer-library diagnostics in LPF's own RMSD metric — how many rotamers
  survive subsampling, and whether the library is connected.
""")


# ==========================================================================
# assembly
# ==========================================================================

def build(kind, env):
    cells = [cell_title(kind, env)]
    cells += cell_setup(env)
    if env == "colab":
        cells.append(cell_helper_colab())
    cells += cells_repo(env)
    cells += cells_rosetta(env)
    cells += cells_project(kind, env)
    cells += cells_protein(kind, env)
    cells += (cells_peptide_input(env) if kind == "peptide" else cells_ligand_input(env))
    cells += cells_nbr(kind, env)
    cells += cells_start(kind, env)
    cells += cells_path(kind, env)
    cells += cells_protocols(kind, env)
    cells += cells_coarse(kind, env)
    cells += cells_trajectory(kind, env)
    cells += cells_analysis(kind, env)
    cells.append(cell_appendix(kind, env))

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    })
    if env == "colab":
        nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    return nb


def main():
    out = []
    for kind in ("ligand", "peptide"):
        for env in ("local", "colab"):
            nb = build(kind, env)
            name = f"LPF_{kind.capitalize()}_{env.capitalize()}.ipynb"
            out_dir = os.path.join(HERE, "notebooks")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, name)
            nbf.validate(nb)
            with open(path, "w") as fh:
                nbf.write(nb, fh)
            n_code = sum(1 for c in nb.cells if c.cell_type == "code")
            n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
            size = os.path.getsize(path) / 1024
            print(f"{name:34s} {len(nb.cells):3d} cells "
                  f"({n_code} code, {n_md} md)  {size:7.1f} KiB")
            out.append(path)
    return out


if __name__ == "__main__":
    main()
