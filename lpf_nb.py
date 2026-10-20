"""
lpf_nb — notebook helpers for Ligand PathFinder (LPF).

Companion module for the four LPF notebooks (ligand/peptide x local/Colab).
It prepares inputs for, launches, and analyses runs of the *unmodified* upstream
code at https://github.com/meilerlab/LigandPathFinder:

    lpf/lpf_coarse_path.py   stage 1 -> coarsepath.txt
    lpf/lpf_trajectory.py    stage 2 -> interface_scores.txt

Reference: Zlobin, Eisenhuth, Ferrario & Meiler, "Exploration of ligand
(un)binding trajectories with Rosetta", bioRxiv 2026, doi 10.64898/2026.10.07.757317

Nothing here monkey-patches LPF. Everything this module writes (start.pdb,
lig.params, lig_conformers.pdb, path, run.json, files/*.xml, files/options) is a
plain input file that the upstream scripts read.

Hard requirements of the upstream code that this module exists to satisfy:
  * the ligand must appear as HETATM records, in one chain, as ONE residue
  * ligand files must be named exactly  lig.params  and  lig_conformers.pdb
  * the tracked atom is lig.params' NBR_ATOM
  * ligand atom order in start.pdb must match the ATOM order in lig.params
  * the ligand must come after the protein in start.pdb
  * the shipped RosettaScripts XMLs hardcode chain="X" in several movers

Dependencies: numpy, scipy (required); matplotlib, rdkit, py3Dmol (optional,
only for the cells that use them).
"""

from __future__ import annotations

import glob
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    # environment
    "check_python_deps", "check_cli_tools", "find_lpf_repo", "clone_lpf_repo",
    "find_rosetta_scripts", "guess_rosetta_database", "validate_rosetta",
    "find_molfile_to_params", "ensure_molfile_to_params",
    # pyrosetta backend
    "pyrosetta_available", "install_pyrosetta", "find_pyrosetta_shim",
    "setup_pyrosetta_shim", "detect_rosetta_backend", "set_shim_jobs",
    "resolve_rosetta_backend",
    # pdb io
    "Atom", "read_pdb_atoms", "write_pdb_atoms", "format_atom_line",
    "pdb_summary", "clean_protein", "extract_chain", "chain_sequence", "fetch_pdb",
    "assemble_start_pdb", "write_header_pdb", "normalize_multimodel_pdb",
    "minimize_start_pose", "write_minimize_xml", "read_score_file",
    # params / ligand files
    "params_atom_names", "params_nbr_atom", "params_resname", "params_nbr_radius",
    "params_bonds", "read_conformers", "count_conformers", "ligand_span",
    "check_start_pdb_vs_params", "suggest_nbr_atom", "set_nbr_atom",
    "map_params_atoms_to_source", "describe_atom",
    "conformer_subsampling_preview",
    # chemistry
    "mol_from_sdf", "mol_from_smiles_3d", "peptide_mol_from_pdb",
    "peptide_completeness_report", "complete_c_terminus",
    "set_physiological_protonation",
    "generate_conformers", "generate_peptide_conformers",
    "conformer_library_report", "write_conformer_sdf",
    "run_molfile_to_params", "suggest_nbr_atom_index",
    # guide path
    "protein_heavy_coords", "ExitRay", "exit_direction_scan", "exit_point_along",
    "write_two_point_path", "write_path_points", "path_clearance_report",
    "read_guide_points",
    "check_path_direction", "check_path_anchor",
    "coarse_path_medoid",
    # protocols
    "auto_grid_width", "write_protocol_xmls", "write_options_file",
    "LIGAND_XML_TUNING", "PEPTIDE_XML_TUNING",
    # config
    "LIGAND_COARSE_PARAMS", "LIGAND_TRAJ_PARAMS",
    "PEPTIDE_COARSE_PARAMS", "PEPTIDE_TRAJ_PARAMS", "BENCHMARK_TRAJ_PARAMS",
    "apply_large_ligand_rule", "LARGE_LIGAND_ATOMS", "SEC_PER_DECOY_BENCHMARK",
    "build_run_config", "write_run_config", "preflight_config",
    # run
    "prepare_run_dir", "run_lpf_stage", "estimate_cost", "relocate_project",
    # analysis
    "load_run_scores", "cumulative_path_length", "plot_energy_profile",
    "summarize_run", "load_stats", "view_trajectory", "zip_results",
    "read_ligand_frames", "path_distance_matrix", "cluster_trajectories",
    "replicate_convergence",
    "rescore_windows",
]

# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

_OK, _WARN, _BAD = "OK", "WARNING", "ERROR"


def _say(level, msg):
    mark = {_OK: "[ok]  ", _WARN: "[warn]", _BAD: "[FAIL]"}[level]
    print(f"{mark} {msg}")


def _abs(p):
    return os.path.abspath(os.path.expanduser(str(p)))


def _require(cond, msg):
    if not cond:
        raise ValueError(msg)


def _rel(path, start):
    """Relative path if it stays inside `start`'s tree, else absolute."""
    path, start = _abs(path), _abs(start)
    try:
        r = os.path.relpath(path, start)
    except ValueError:
        return path
    return r


# --------------------------------------------------------------------------
# environment checks
# --------------------------------------------------------------------------

def check_python_deps(optional=("matplotlib", "rdkit", "py3Dmol")):
    """Report the Python packages the notebooks use. Raises if numpy/scipy missing."""
    report = {}
    for mod in ("numpy", "scipy"):
        try:
            m = __import__(mod)
            report[mod] = getattr(m, "__version__", "?")
            _say(_OK, f"{mod} {report[mod]}")
        except ImportError:
            report[mod] = None
            _say(_BAD, f"{mod} is REQUIRED by LigandPathFinder and is missing")
    for mod in optional:
        try:
            m = __import__(mod)
            report[mod] = getattr(m, "__version__", "?")
            _say(_OK, f"{mod} {report[mod]} (optional)")
        except ImportError:
            report[mod] = None
            _say(_WARN, f"{mod} not installed (optional; some cells will be skipped)")
    if not report.get("numpy") or not report.get("scipy"):
        raise ImportError("numpy and scipy are required: pip install numpy scipy")
    return report


def check_cli_tools(need_mpi=False):
    """gzip is called after every Rosetta call; zip only when output.archive is on."""
    report = {}
    for tool, why, critical in (
        ("gzip", "compresses decoys after every Rosetta call (not optional)", True),
        ("zip", "needed only when output.archive is true", False),
        ("mpirun", "needed only when rosetta_scripts.n_proc > 1", need_mpi),
    ):
        path = shutil.which(tool)
        report[tool] = path
        if path:
            _say(_OK, f"{tool} -> {path}")
        else:
            _say(_BAD if critical else _WARN, f"{tool} not on PATH ({why})")
    if not report["gzip"]:
        raise FileNotFoundError("gzip is required by LigandPathFinder (util.run_tool)")
    return report


_LPF_MARKERS = ("lpf/lpf_coarse_path.py", "lpf/lpf_trajectory.py", "rosettaXML_files")


def _is_lpf_repo(path):
    return all(os.path.exists(os.path.join(path, m)) for m in _LPF_MARKERS)


def find_lpf_repo(*hints):
    """Locate a LigandPathFinder checkout. Returns the repo root or None."""
    cands = [h for h in hints if h]
    cands += [
        os.getcwd(),
        os.path.join(os.getcwd(), "LigandPathFinder"),
        os.path.join(os.path.dirname(os.getcwd()), "LigandPathFinder"),
        os.path.expanduser("~/LigandPathFinder"),
        "/content/LigandPathFinder",
    ]
    for c in cands:
        c = _abs(c)
        if _is_lpf_repo(c):
            return c
    return None


def clone_lpf_repo(dest, url="https://github.com/meilerlab/LigandPathFinder.git",
                   depth=1, force=False):
    """Clone (or reuse) the upstream repository. Returns the repo root."""
    dest = _abs(dest)
    if _is_lpf_repo(dest) and not force:
        _say(_OK, f"LigandPathFinder already present: {dest}")
        return dest
    if os.path.exists(dest) and force:
        shutil.rmtree(dest)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    cmd = ["git", "clone"] + (["--depth", str(depth)] if depth else []) + [url, dest]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    if not _is_lpf_repo(dest):
        raise RuntimeError(f"clone finished but {dest} does not look like LigandPathFinder")
    _say(_OK, f"cloned to {dest}")
    return dest


_ROSETTA_BIN_GLOBS = (
    "rosetta_scripts.*",
    "*/bin/rosetta_scripts.*",
    "*/source/bin/rosetta_scripts.*",
    "*/main/source/bin/rosetta_scripts.*",
)


def find_rosetta_scripts(*hints, search_roots=()):
    """Find a rosetta_scripts binary. Returns a list of candidates, best first."""
    found = []
    for h in hints:
        if h and os.path.isfile(_abs(h)):
            found.append(_abs(h))
    roots = list(search_roots) + [
        os.getcwd(), os.path.expanduser("~"), "/opt", "/usr/local",
        "/content/drive/MyDrive", "/content",
    ]
    for root in roots:
        root = _abs(root)
        if not os.path.isdir(root):
            continue
        for pat in _ROSETTA_BIN_GLOBS:
            for hit in glob.glob(os.path.join(root, pat)):
                if os.path.isfile(hit) and hit not in found:
                    found.append(hit)
    # prefer static > mpi-less > shorter path; never prefer a .py or .debug build
    def rank(p):
        b = os.path.basename(p)
        return (
            0 if ".static." in b else 1,
            0 if "release" in b else 1,
            1 if ".debug" in b else 0,
            len(p),
        )
    found = [f for f in found if not f.endswith(".py")]
    return sorted(dict.fromkeys(found), key=rank)


def guess_rosetta_database(rosetta_bin):
    """
    Locate the Rosetta database for a binary.

    A binary lifted out of its source tree cannot find the database on its own;
    the notebooks then pass -in:path:database explicitly in the options file.
    """
    p = _abs(rosetta_bin)
    for _ in range(6):
        p = os.path.dirname(p)
        if not p or p == "/":
            break
        for cand in (os.path.join(p, "database"),
                     os.path.join(p, "main", "database"),
                     os.path.join(p, "rosetta", "main", "database")):
            if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "chemical",
                                                                   "residue_type_sets",
                                                                   "fa_standard",
                                                                   "residue_types.txt")):
                return cand
    return None


def validate_rosetta(rosetta_bin, database=None, timeout=300, make_executable=True):
    """
    Check that a rosetta_scripts binary exists, runs here, and can find its database.

    Returns a dict: {binary, is_mpi, executable, launched, database, stdout_head}.
    Prints a readable report; does not raise unless the file is missing.
    """
    rb = _abs(rosetta_bin)
    out = {"binary": rb, "is_mpi": False, "executable": False, "launched": False,
           "database": None, "stdout_head": "", "backend": "missing"}
    if not os.path.isfile(rb):
        _say(_BAD, f"no such file: {rb}")
        return out

    out["backend"] = detect_rosetta_backend(rb)
    is_shim = out["backend"] == "pyrosetta-shim"
    out["is_mpi"] = (not is_shim) and ".mpi." in os.path.basename(rb)
    if not os.access(rb, os.X_OK):
        if make_executable:
            os.chmod(rb, os.stat(rb).st_mode | 0o111)
            _say(_WARN, "binary was not executable; chmod +x applied")
        else:
            _say(_BAD, "binary is not executable (chmod +x it)")
            return out
    out["executable"] = os.access(rb, os.X_OK)

    if is_shim:
        # PyRosetta carries its own database and finds it on init, so there is
        # nothing to point -in:path:database at.
        out["database"] = None
        _say(_OK, "PyRosetta backend: the database ships inside the pyrosetta package")
        db = None
    else:
        db = _abs(database) if database else guess_rosetta_database(rb)
        out["database"] = db
        if db:
            _say(_OK, f"Rosetta database: {db}")
        else:
            _say(_WARN, "Rosetta database not found next to the binary — set it "
                        "explicitly (the notebooks will write -in:path:database into "
                        "the options file)")

    argv = [rb, "-help"]
    if db:
        argv += ["-in:path:database", db]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        blob = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        _say(_WARN, f"`{os.path.basename(rb)} -help` did not finish in {timeout}s; "
                    "assuming it works but check manually")
        out["launched"] = True
        return out
    except OSError as exc:
        _say(_BAD, f"could not execute the binary: {exc}")
        return out

    out["stdout_head"] = blob[:4000]
    markers = ("rosetta", "Rosetta", "core.init", "-parser:protocol", "usage")
    out["launched"] = any(m in blob for m in markers)
    if out["launched"]:
        kind = ("PyRosetta-backed shim" if is_shim else
                ("MPI build" if out["is_mpi"] else "binary"))
        _say(_OK, f"runs here: {os.path.basename(rb)} ({kind})")
        if is_shim:
            jobs = os.environ.get("LPF_SHIM_JOBS", "1")
            print(f"       keep rosetta_scripts.n_proc = 1; decoy parallelism comes "
                  f"from LPF_SHIM_JOBS (currently {jobs})")
    else:
        _say(_BAD, "the binary did not produce recognisable Rosetta output. First lines:")
        print(textwrap.indent(blob[:1500] or "(no output)", "       "))
        print("       Common causes: wrong architecture (a Linux build on macOS, or an "
              "arm64 build on x86), or a dynamic build whose shared libraries are missing.")
    return out


_MFTP_GLOBS = (
    "main/source/scripts/python/public/molfile_to_params.py",
    "source/scripts/python/public/molfile_to_params.py",
    "scripts/python/public/molfile_to_params.py",
    "molfile_to_params.py",
)


def find_molfile_to_params(*hints, rosetta_bin=None):
    """Find Rosetta's molfile_to_params.py (ships with the Rosetta source tree)."""
    cands = [_abs(h) for h in hints if h]
    roots = []
    if rosetta_bin:
        p = _abs(rosetta_bin)
        for _ in range(6):
            p = os.path.dirname(p)
            if not p or p == "/":
                break
            roots.append(p)
    roots += [os.getcwd(), os.path.expanduser("~/rosetta"), "/opt/rosetta",
              "/content/drive/MyDrive", "/content"]
    for root in roots:
        for pat in _MFTP_GLOBS:
            cands.extend(glob.glob(os.path.join(root, pat)))
    for c in dict.fromkeys(cands):
        if os.path.isfile(c):
            return c
    return None


# --------------------------------------------------------------------------
# PyRosetta backend
#
# LPF runs the `rosetta_scripts` *executable* as a subprocess and parses
# score.sc; it never imports Rosetta. So `pip install pyrosetta` alone gives it
# nothing to call — PyRosetta ships no such executable.
#
# pyrosetta_rosetta_scripts.py bridges that: it presents the slice of the
# rosetta_scripts command line LPF uses, runs the XML through PyRosetta, and
# writes decoys plus a score file in the expected format. These helpers install
# PyRosetta, write the launcher LPF will exec, and tell the two backends apart
# so the rest of the notebook can stay the same.
# --------------------------------------------------------------------------

_SHIM_MARKER = "pyrosetta_rosetta_scripts"
_SHIM_BASENAME = "pyrosetta_rosetta_scripts.py"


def find_pyrosetta_shim(*hints):
    """Locate pyrosetta_rosetta_scripts.py (it ships next to this module)."""
    cands = [h for h in hints if h]
    here = os.path.dirname(os.path.abspath(__file__))
    cands += [os.path.join(here, _SHIM_BASENAME),
              os.path.join(os.getcwd(), _SHIM_BASENAME),
              _SHIM_BASENAME]
    for c in cands:
        c = _abs(c)
        if os.path.isfile(c):
            return c
    return None


def pyrosetta_available(python_exe=None):
    """Is PyRosetta importable by `python_exe` (default: this interpreter)?"""
    exe = python_exe or sys.executable
    proc = subprocess.run(
        [exe, "-c", "import pyrosetta, os; print(pyrosetta.__file__)"],
        capture_output=True, text=True)
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    return False, (proc.stderr or proc.stdout).strip()[-400:]


def install_pyrosetta(python_exe=None, silent=False):
    """
    Install PyRosetta with pip, via RosettaCommons' installer package.

    `pip install pyrosetta-installer` fetches a ~4 KB helper; calling its
    install_pyrosetta() then downloads the real ~1.5 GB wheel from
    rosettacommons.org. Budget 5-15 minutes and ~2.5 GB of disk.

    PyRosetta is Rosetta: free for academic and non-commercial use under
    Rosetta's own licence, not LigandPathFinder's. Installing it here is you
    accepting that licence.

    Caveat worth knowing: pyrosetta_installer shells out to whichever `pip` is
    first on PATH, so it does NOT respect `python_exe` being a virtualenv — it
    can install into a different environment than the one you asked for. This
    function therefore re-checks importability from `python_exe` afterwards and
    reports where the package actually landed, rather than trusting the
    installer's own success message.
    """
    exe = python_exe or sys.executable
    have, where = pyrosetta_available(exe)
    if have:
        _say(_OK, f"PyRosetta already present: {where}")
        return True

    print("installing pyrosetta-installer ...")
    proc = subprocess.run([exe, "-m", "pip", "install", "-q", "pyrosetta-installer"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        _say(_BAD, "could not install pyrosetta-installer:")
        print(textwrap.indent((proc.stderr or proc.stdout)[-1500:], "       "))
        return False

    print("downloading PyRosetta (~1.5 GB wheel; this is the slow part) ...")
    proc = subprocess.run(
        [exe, "-c", "import pyrosetta_installer; "
                    f"pyrosetta_installer.install_pyrosetta(silent={bool(silent)})"],
        capture_output=True, text=True)
    tail = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        _say(_BAD, "PyRosetta install failed:")
        print(textwrap.indent(tail[-2000:], "       "))
        return False

    have, where = pyrosetta_available(exe)
    if have:
        _say(_OK, f"PyRosetta installed: {where}")
        if exe != sys.executable and not where.startswith(os.path.dirname(
                os.path.dirname(exe))):
            _say(_WARN, f"it landed outside {exe}'s environment — pyrosetta_installer "
                        f"uses the pip on PATH, not the interpreter you named. It is "
                        f"importable from {exe}, so this run will work.")
    else:
        _say(_BAD, f"the installer reported success but PyRosetta is not importable "
                   f"from {exe}:\n{where}\n"
                   f"It most likely went to whichever pip is first on PATH. Find it "
                   f"with `python -c \'import pyrosetta; print(pyrosetta.__file__)\'` "
                   f"in your other environments and set PYROSETTA_PYTHON to that "
                   f"interpreter.")
    return have


def setup_pyrosetta_shim(dest_dir, python_exe=None, shim_path=None,
                         wrapper_name="rosetta_scripts.pyrosetta", check=True):
    """
    Write the executable launcher LPF will call, backed by PyRosetta.

    Returns its path, ready for the "rosetta" key of run.json.

    The launcher is a one-line shell script that execs a pinned interpreter, so
    it works when PyRosetta lives in a virtualenv that is not on PATH. The name
    deliberately avoids '.mpi.': LPF reads that substring as an MPI build and
    would then permit n_proc > 1, which the shim does not implement (use the
    LPF_SHIM_JOBS environment variable instead).
    """
    exe = python_exe or sys.executable
    shim = find_pyrosetta_shim(shim_path)
    _require(shim, f"{_SHIM_BASENAME} not found. It ships alongside lpf_nb.py; pass "
                   f"shim_path=... if you keep it elsewhere.")

    if check:
        have, where = pyrosetta_available(exe)
        if not have:
            raise RuntimeError(
                f"PyRosetta is not importable by {exe}:\n{where}\n"
                f"Run install_pyrosetta() first, or pass the python_exe that has it.")

    dest_dir = _abs(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    local_shim = os.path.join(dest_dir, _SHIM_BASENAME)
    if _abs(local_shim) != _abs(shim):
        shutil.copy(shim, local_shim)
    wrapper = os.path.join(dest_dir, wrapper_name)

    proc = subprocess.run([exe, local_shim, "--write-wrapper", wrapper,
                           "--python", exe], capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.isfile(wrapper):
        raise RuntimeError("could not write the launcher:\n"
                           + (proc.stderr or proc.stdout)[-1500:])

    proc = subprocess.run([exe, local_shim, "--self-test"],
                          capture_output=True, text=True)
    if "self-test: PASS" not in proc.stdout:
        _say(_WARN, "the shim's own self-test did not pass:")
        print(textwrap.indent(proc.stdout[-1500:], "       "))
    else:
        _say(_OK, "shim self-test passed (command-line parsing and score-file format)")

    _say(_OK, f"PyRosetta launcher: {wrapper}")
    print(f"       interpreter: {exe}")
    print(f"       set rosetta_scripts.n_proc = 1; for parallel decoys set the "
          f"LPF_SHIM_JOBS environment variable instead")
    return wrapper


def resolve_rosetta_backend(backend="auto", rosetta_hint=None, python_exe=None,
                            shim_dir=None, install_if_missing=False, shim_jobs=1,
                            database=None, verbose=True):
    """
    Settle on a Rosetta backend and return something LPF can exec.

    `backend`:
      "auto"       prefer a compiled rosetta_scripts binary; fall back to an
                   importable PyRosetta; only then give up. This is what you
                   want if you have just activated a PyRosetta environment.
      "binary"     require a compiled binary.
      "pyrosetta"  require PyRosetta, installing it when install_if_missing.

    Returns {"rosetta", "database", "is_mpi", "backend", "chose"}.
    """
    exe = python_exe or sys.executable
    _require(backend in ("auto", "binary", "pyrosetta"),
             f"backend must be auto/binary/pyrosetta, got {backend!r}")

    cands = find_rosetta_scripts(rosetta_hint) if backend != "pyrosetta" else []
    have_pr, where = (pyrosetta_available(exe) if backend != "binary"
                      else (False, ""))

    chose = None
    if backend == "binary" or (backend == "auto" and cands):
        if not cands:
            extra = ""
            if pyrosetta_available(exe)[0]:
                extra = ("\n  PyRosetta IS importable here — either drop "
                         "--rosetta-backend binary (auto will use it) or pass "
                         "--rosetta-backend pyrosetta.")
            raise FileNotFoundError(
                "No rosetta_scripts binary found. Give its full path, or use the "
                "PyRosetta backend." + extra +
                "\n  binaries: https://downloads.rosettacommons.org/downloads/academic/"
                "\n  source:   https://github.com/RosettaCommons/rosetta")
        if len(cands) > 1 and not rosetta_hint and verbose:
            print("several rosetta_scripts candidates, using the first:")
            for c in cands[:5]:
                print("   ", c)
        binary, chose = cands[0], "binary"
        db = database
    else:
        if not have_pr:
            if not install_if_missing:
                raise RuntimeError(
                    f"PyRosetta is not importable by {exe}.\n  {where}\n"
                    f"  Install it (pip install pyrosetta-installer, then "
                    f"pyrosetta_installer.install_pyrosetta()), point this at the "
                    f"interpreter that has it, or supply a rosetta_scripts binary.")
            if not install_pyrosetta(exe):
                raise RuntimeError("PyRosetta install failed — see above")
        elif verbose:
            print(f"PyRosetta: {where}")
        _require(shim_dir, "the PyRosetta backend needs shim_dir (where to write "
                           "the launcher)")
        if verbose and backend == "auto":
            print("no rosetta_scripts binary found, but PyRosetta is available — "
                  "using it through the shim")
            print()
        binary, chose = setup_pyrosetta_shim(shim_dir, python_exe=exe), "pyrosetta"
        db = None
        if int(shim_jobs) > 1:
            print()
            set_shim_jobs(shim_jobs)

    if verbose:
        print()
    info = validate_rosetta(binary, db)
    if not info["launched"]:
        raise RuntimeError("Rosetta did not run here — see the report above")
    return {"rosetta": binary, "database": info["database"], "chose": chose,
            "is_mpi": info["is_mpi"], "backend": info["backend"]}


def detect_rosetta_backend(path):
    """
    'pyrosetta-shim' or 'binary'.

    The two differ in what the rest of the set-up should do: the shim needs no
    -in:path:database (PyRosetta carries its own database) and cannot do MPI.
    """
    p = _abs(path)
    if not os.path.isfile(p):
        return "missing"
    try:
        with open(p, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return "binary"
    if _SHIM_MARKER.encode() in head:
        return "pyrosetta-shim"
    return "binary"


def set_shim_jobs(n):
    """
    Set LPF_SHIM_JOBS for subsequent runs.

    LPF builds the Rosetta command line itself, so the shim takes this through
    the environment. LPF's subprocess inherits the notebook's environment, so
    setting it here reaches the shim.

    Each worker initialises its own PyRosetta — budget 1-2 GB of RAM per job,
    and do not exceed nstruct.
    """
    n = max(1, int(n))
    os.environ["LPF_SHIM_JOBS"] = str(n)
    print(f"LPF_SHIM_JOBS={n}"
          + ("  (serial)" if n == 1 else f"  (~{n * 1.5:.0f} GB RAM for workers)"))
    return n


# --------------------------------------------------------------------------
# PDB reading / writing
#
# Column layout matters: LPF's util.PDBHandler.read_window slices
# line[12:16] name, line[21] chain, line[22:26] resnum, line[30:54] xyz.
# Anything written here must land in exactly those columns.
# --------------------------------------------------------------------------

_TWO_LETTER_ELEMENTS = {
    "CL", "BR", "ZN", "FE", "MG", "MN", "CU", "CA", "NA", "SE", "SI", "LI",
    "BE", "AL", "NI", "CO", "CD", "HG", "PT", "PD", "AU", "AG", "SN", "PB",
    "AS", "SB", "TE", "MO", "CR", "TI", "ZR", "RU", "RH", "IR", "OS", "W",
}


@dataclass
class Atom:
    record: str = "ATOM"
    serial: int = 1
    name: str = "C"
    altloc: str = " "
    resname: str = "UNK"
    chain: str = "A"
    resseq: int = 1
    icode: str = " "
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    occ: float = 1.0
    bfac: float = 0.0
    element: str = ""
    charge: str = "  "
    name_col13: bool = False   # True -> atom name starts in column 13

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def key(self):
        """Identity used to match atoms across files."""
        return (self.chain, self.resseq, self.icode, self.name)


#: Standard amino-acid / nucleotide atom names whose first two characters collide
#: with a real element symbol (CA calcium, CD cadmium, NI nickel, ...). Without
#: this guard a protein C-alpha read from a file that omits the element column
#: becomes calcium, and every clash and clearance calculation shifts.
_PROTEIN_ATOM_NAMES = {
    "CA", "CB", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3", "CG", "CG1", "CG2",
    "CH2", "CZ", "CZ2", "CZ3", "ND", "ND1", "ND2", "NE", "NE1", "NE2", "NH1",
    "NH2", "NZ", "OD", "OD1", "OD2", "OE", "OE1", "OE2", "OG", "OG1", "OH",
    "OXT", "SD", "SG", "HA", "HB", "HD", "HE", "HG", "HH", "HZ", "NA1",
}


def _infer_element(name, raw_element="", col13=False):
    """
    Element symbol for an atom name.

    Uses the explicit element column when the file has one. Otherwise it falls
    back on the PDB convention that a one-character element is right-shifted
    into column 14, so a name starting in column 13 may be a two-character
    element — plus an explicit guard for protein atom names like CA and CD,
    which would otherwise be read as calcium and cadmium.
    """
    e = (raw_element or "").strip().upper()
    if e:
        return e
    n = name.strip().upper()
    n = re.sub(r"^\d+", "", n)          # 1HB -> HB
    if not n:
        return ""
    if n in _PROTEIN_ATOM_NAMES:
        return n[0]
    if len(n) >= 2 and n[:2] in _TWO_LETTER_ELEMENTS:
        # a one-character element would have been shifted right, so only trust a
        # two-character reading when the name really does start in column 13
        if col13 or not re.fullmatch(r"[A-Z][A-Z]\d*", n):
            return n[:2]
        return n[:2] if col13 else n[0]
    return n[0]


def _name_field(name, element, starts_col13):
    """
    Render a 4-character atom-name field.

    PDB convention: a 1-character element symbol is right-shifted by one column
    (so ' CA ' is a carbon alpha and 'CA  ' is calcium). 4-character names and
    2-character elements start in column 13.
    """
    nm = name.strip()
    if len(nm) >= 4 or starts_col13:
        return nm[:4].ljust(4)
    if len(element) >= 2:
        return nm.ljust(4)
    return (" " + nm).ljust(4)


def format_atom_line(a: Atom) -> str:
    elem = _infer_element(a.name, a.element, a.name_col13)
    namef = _name_field(a.name, elem, a.name_col13)
    serial = a.serial
    ser = f"{serial:5d}" if serial < 100000 else f"{serial:05X}"[-5:]
    res = f"{a.resseq:4d}" if -999 <= a.resseq <= 9999 else f"{a.resseq % 10000:4d}"
    return (
        f"{a.record:<6s}{ser}"
        f" {namef}{a.altloc or ' '}{a.resname[:3]:>3s}"
        f" {(a.chain or ' ')[:1]}{res}{a.icode or ' '}"
        f"   {a.x:8.3f}{a.y:8.3f}{a.z:8.3f}"
        f"{a.occ:6.2f}{a.bfac:6.2f}"
        f"          {elem:>2s}{a.charge:<2s}"
    ).rstrip() + "\n"


def read_pdb_atoms(path, records=("ATOM", "HETATM"), models="first"):
    """
    Parse ATOM/HETATM records into a list of Atom.

    models='first' stops at the first ENDMDL (multi-model files such as
    full_trajectory.pdb); models='all' keeps everything.
    """
    path = _abs(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDB not found: {path}")
    atoms = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            tag = line[:6].strip()
            if tag in ("ENDMDL",) and models == "first" and atoms:
                break
            if tag not in records:
                continue
            if len(line) < 54:
                continue
            try:
                serial = int(line[6:11])
            except ValueError:
                serial = len(atoms) + 1
            try:
                resseq = int(line[22:26])
            except ValueError:
                resseq = 0
            raw_name = line[12:16]
            atoms.append(Atom(
                record=tag,
                serial=serial,
                name=raw_name.strip(),
                altloc=line[16] if len(line) > 16 else " ",
                resname=line[17:20].strip(),
                chain=line[21] if len(line) > 21 else " ",
                resseq=resseq,
                icode=line[26] if len(line) > 26 else " ",
                x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                occ=float(line[54:60]) if len(line) >= 60 and line[54:60].strip() else 1.0,
                bfac=float(line[60:66]) if len(line) >= 66 and line[60:66].strip() else 0.0,
                element=line[76:78].strip() if len(line) >= 78 else "",
                charge=line[78:80] if len(line) >= 80 else "  ",
                name_col13=(raw_name[:1] != " "),
            ))
    return atoms


def write_pdb_atoms(path, atoms, header_lines=(), renumber=True,
                    ter_after_chain_change=True, end=True):
    """Write atoms back out, optionally renumbering serials from 1."""
    lines = list(header_lines)
    prev = None
    serial = 0
    for a in atoms:
        if ter_after_chain_change and prev is not None and a.chain != prev.chain:
            serial += 1
            lines.append("TER\n")
        serial += 1
        b = Atom(**{**a.__dict__})
        if renumber:
            b.serial = serial
        lines.append(format_atom_line(b))
        prev = a
    if atoms:
        lines.append("TER\n")
    if end:
        lines.append("END\n")
    path = _abs(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.writelines(lines)
    return path


_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL", "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN",
}
_WATER = {"HOH", "WAT", "DOD", "TIP", "TIP3", "SOL", "H2O"}
_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "C", "HID": "H", "HIE": "H",
    "HIP": "H", "CYX": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}


def pdb_summary(path_or_atoms, label=""):
    """Print a per-chain inventory: residues, atoms, waters, hetero residues."""
    atoms = (read_pdb_atoms(path_or_atoms)
             if isinstance(path_or_atoms, (str, os.PathLike)) else list(path_or_atoms))
    chains = {}
    for a in atoms:
        c = chains.setdefault(a.chain, {"res": set(), "atoms": 0, "het": {},
                                        "water": 0, "h": 0})
        c["atoms"] += 1
        c["res"].add((a.resseq, a.icode, a.resname))
        if a.resname in _WATER:
            c["water"] += 1
        elif a.resname not in _AA3 and a.record == "HETATM":
            c["het"][a.resname] = c["het"].get(a.resname, 0) + 1
        if _infer_element(a.name, a.element, a.name_col13) == "H":
            c["h"] += 1
    if label:
        print(label)
    print(f"  {'chain':6s}{'residues':>10s}{'atoms':>8s}{'hydrogens':>11s}"
          f"{'waters':>8s}  hetero")
    for c in sorted(chains):
        d = chains[c]
        het = ", ".join(f"{k}x{v//max(1,1)}" for k, v in sorted(d["het"].items())[:6])
        het = (het + " ...") if len(d["het"]) > 6 else het
        print(f"  {c!r:6s}{len(d['res']):>10d}{d['atoms']:>8d}{d['h']:>11d}"
              f"{d['water']:>8d}  {het}")
    return chains


def fetch_pdb(pdb_id, dest_dir=".", assembly=False):
    """Download a structure from RCSB. Returns the local path."""
    import urllib.request
    pdb_id = str(pdb_id).strip().lower()
    _require(re.fullmatch(r"[0-9a-z]{4}", pdb_id), f"not a PDB ID: {pdb_id!r}")
    name = f"{pdb_id}.pdb1" if assembly else f"{pdb_id}.pdb"
    url = f"https://files.rcsb.org/download/{name}"
    dest = os.path.join(_abs(dest_dir), f"{pdb_id}.pdb")
    os.makedirs(_abs(dest_dir), exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    _say(_OK, f"downloaded {url} -> {dest}")
    return dest


def clean_protein(in_pdb, out_pdb, keep_chains=None, drop_chains=(),
                  remove_waters=True, keep_hetero=(), remove_other_hetero=True,
                  strip_hydrogens=False, keep_altloc="A", drop_elements=()):
    """
    Produce a protein-only starting file.

    keep_hetero: hetero residue names to KEEP (cofactors, metals, structural
    waters). Each kept non-canonical residue needs its own Rosetta .params file
    listed in extra_files and in the options file's -extra_res_fa.
    """
    atoms = read_pdb_atoms(in_pdb)
    kept, dropped = [], {}

    def drop(a, why):
        dropped[why] = dropped.get(why, 0) + 1

    keep_hetero = {h.upper() for h in keep_hetero}
    drop_elements = {e.upper() for e in drop_elements}
    for a in atoms:
        rn = a.resname.upper()
        if keep_chains is not None and a.chain not in keep_chains:
            drop(a, f"chain not in {sorted(keep_chains)}"); continue
        if a.chain in drop_chains:
            drop(a, "dropped chain"); continue
        if rn in _WATER:
            if remove_waters and rn not in keep_hetero:
                drop(a, "water"); continue
        elif rn not in _AA3:
            if rn in keep_hetero:
                pass
            elif remove_other_hetero:
                drop(a, f"hetero {rn}"); continue
        el = _infer_element(a.name, a.element, a.name_col13)
        if strip_hydrogens and el in ("H", "D"):
            drop(a, "hydrogen"); continue
        if el in drop_elements:
            drop(a, f"element {el}"); continue
        if a.altloc.strip() and keep_altloc and a.altloc.strip() != keep_altloc:
            drop(a, f"altloc {a.altloc.strip()}"); continue
        b = Atom(**a.__dict__)
        b.altloc = " "
        kept.append(b)

    _require(kept, "clean_protein removed every atom — check keep_chains/keep_hetero")
    write_pdb_atoms(out_pdb, kept)
    print(f"clean_protein: {len(atoms)} -> {len(kept)} atoms  ({_abs(out_pdb)})")
    for why, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"    removed {n:6d}  {why}")
    nonstd = sorted({a.resname for a in kept if a.resname.upper() not in _AA3})
    if nonstd:
        _say(_WARN, "non-canonical residues kept: " + ", ".join(nonstd)
                    + " — each needs a .params file in extra_files and in the "
                      "options file (-extra_res_fa)")
    return _abs(out_pdb)


def extract_chain(in_pdb, chain, out_pdb, resnames=None, resseq_range=None,
                  strip_hydrogens=False):
    """Pull one chain (optionally a residue subset) into its own PDB."""
    atoms = read_pdb_atoms(in_pdb)
    sel = []
    for a in atoms:
        if a.chain != chain:
            continue
        if resnames and a.resname.upper() not in {r.upper() for r in resnames}:
            continue
        if resseq_range and not (resseq_range[0] <= a.resseq <= resseq_range[1]):
            continue
        if strip_hydrogens and _infer_element(a.name, a.element, a.name_col13) in ("H", "D"):
            continue
        if a.resname.upper() in _WATER:
            continue
        sel.append(a)
    _require(sel, f"no atoms for chain {chain!r} in {in_pdb} "
                  f"(chains present: {sorted({a.chain for a in atoms})})")
    write_pdb_atoms(out_pdb, sel)
    nres = len({(a.resseq, a.icode) for a in sel})
    print(f"extract_chain {chain!r}: {nres} residue(s), {len(sel)} atoms -> {_abs(out_pdb)}")
    return _abs(out_pdb)


def chain_sequence(in_pdb, chain):
    """One-letter sequence of a chain, in file order. Unknown residues -> 'X'."""
    atoms = read_pdb_atoms(in_pdb)
    seen, seq, resids = set(), [], []
    for a in atoms:
        if a.chain != chain:
            continue
        rid = (a.resseq, a.icode)
        if rid in seen or a.resname.upper() in _WATER:
            continue
        seen.add(rid)
        seq.append(_AA1.get(a.resname.upper(), "X"))
        resids.append((a.resseq, a.icode, a.resname))
    return "".join(seq), resids


def write_header_pdb(path, lines=()):
    """
    header.pdb is prepended verbatim to every generated task.pdb.

    Empty for an ordinary system; used for metal-binding constraint records,
    membrane spans or symmetry definitions.
    """
    path = _abs(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        for ln in lines:
            fh.write(ln if ln.endswith("\n") else ln + "\n")
    _say(_OK, f"header.pdb written ({len(list(lines))} line(s)): {path}")
    return path


# --------------------------------------------------------------------------
# Rosetta ligand .params files
#
# LPF reads NBR_ATOM straight out of lig.params and tracks that atom along the
# guide path, and Rosetta's StartFrom(use_nbr=1) places *that* atom on the
# target point. Changing which atom it is therefore changes what the path
# means — which is the main thing to get right for a peptide.
# --------------------------------------------------------------------------

def _params_lines(params_path):
    with open(_abs(params_path), "r", errors="replace") as fh:
        return fh.readlines()


def params_atom_names(params_path):
    """Atom names in lig.params ATOM order."""
    names = [ln.split()[1] for ln in _params_lines(params_path) if ln.startswith("ATOM")]
    _require(names, f"no ATOM records in {params_path}")
    return names


def params_nbr_atom(params_path):
    for ln in _params_lines(params_path):
        if ln.startswith("NBR_ATOM"):
            return ln.split()[1]
    raise ValueError(f"no NBR_ATOM record in {params_path}")


def params_nbr_radius(params_path):
    for ln in _params_lines(params_path):
        if ln.startswith("NBR_RADIUS"):
            return float(ln.split()[1])
    return None


def params_resname(params_path):
    """The 3-character PDB residue name Rosetta will use for the ligand."""
    for ln in _params_lines(params_path):
        if ln.startswith("IO_STRING"):
            return ln.split()[1][:3]
    for ln in _params_lines(params_path):
        if ln.startswith("NAME"):
            return ln.split()[1][:3]
    raise ValueError(f"no IO_STRING/NAME record in {params_path}")


def read_conformers(conformers_pdb):
    """
    Parse lig_conformers.pdb into [{name: xyz}, ...], one dict per conformer.

    The file is TER-separated blocks, which is what LPF's ConformerSampler
    expects; every block must carry the same atom names.
    """
    blocks, cur = [], {}
    with open(_abs(conformers_pdb), "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("TER"):
                if cur:
                    blocks.append(cur)
                    cur = {}
            elif line[:6].strip() in ("ATOM", "HETATM") and len(line) >= 54:
                cur[line[12:16].strip()] = (
                    float(line[30:38]), float(line[38:46]), float(line[46:54]))
    if cur:
        blocks.append(cur)
    return blocks


def count_conformers(conformers_pdb):
    return len(read_conformers(conformers_pdb))


def ligand_span(pdb_or_conformers, chain=None):
    """
    Longest interatomic distance of the ligand, in A.

    Used to size SCORINGGRIDS width and the plane-constraint triangle: the
    scoring grid must comfortably contain the whole molecule, which the stock
    width=20 does not for anything peptide-sized.
    """
    p = _abs(pdb_or_conformers)
    if chain is not None:
        pts = np.array([a.xyz for a in read_pdb_atoms(p) if a.chain == chain])
    else:
        blocks = read_conformers(p)
        pts = np.array([xyz for b in blocks for xyz in b.values()])
    _require(len(pts) > 1, f"fewer than two atoms found in {p}")
    centred = pts - pts.mean(axis=0)
    # bounding-sphere-ish: max pairwise distance via the convex extremes
    d = 0.0
    for axis in np.eye(3):
        proj = centred @ axis
        d = max(d, proj.max() - proj.min())
    far = np.linalg.norm(centred, axis=1)
    return float(max(d, 2 * far.max()))


def suggest_nbr_atom(ligand_pdb=None, conformers_pdb=None, chain=None,
                     params_path=None, top=8):
    """
    Rank candidate NBR_ATOMs by how central they are.

    Score = max distance from the candidate to any heavy atom, over every
    conformer. The minimiser is the most central atom; tracking it makes the
    guide path describe the molecule's overall translation rather than the
    motion of one protruding tip. molfile_to_params' own pick is often a
    terminal atom, which is fine for a small ligand and poor for a peptide.
    """
    if conformers_pdb and os.path.isfile(_abs(conformers_pdb)):
        blocks = read_conformers(conformers_pdb)
    else:
        _require(ligand_pdb, "give conformers_pdb or ligand_pdb")
        atoms = [a for a in read_pdb_atoms(ligand_pdb)
                 if chain is None or a.chain == chain]
        blocks = [{a.name: (a.x, a.y, a.z) for a in atoms}]
    _require(blocks, "no conformers parsed")
    names = list(blocks[0])
    heavy = [n for n in names if _infer_element(n) != "H"]
    _require(heavy, "no heavy atoms found")

    worst = {n: 0.0 for n in heavy}
    for b in blocks:
        hv = np.array([b[n] for n in heavy])
        for i, n in enumerate(heavy):
            worst[n] = max(worst[n], float(np.linalg.norm(hv - hv[i], axis=1).max()))
    ranked = sorted(worst.items(), key=lambda kv: kv[1])

    current = params_nbr_atom(params_path) if params_path else None
    print(f"{'atom':>8s}  max distance to any heavy atom (A), over "
          f"{len(blocks)} conformer(s)")
    for n, d in ranked[:top]:
        tag = "   <- current NBR_ATOM" if n == current else ""
        print(f"{n:>8s}  {d:8.2f}{tag}")
    if current and current not in dict(ranked[:top]):
        print(f"{current:>8s}  {worst.get(current, float('nan')):8.2f}   "
              f"<- current NBR_ATOM")
    return [n for n, _ in ranked], dict(ranked)


def set_nbr_atom(params_path, atom_name, out_path=None, conformers_pdb=None,
                 margin=0.5, recompute_radius=True):
    """
    Rewrite NBR_ATOM (and NBR_RADIUS) in a .params file.

    NBR_RADIUS must enclose every atom of the residue as seen from NBR_ATOM, or
    Rosetta's neighbour detection silently truncates interactions. It is
    recomputed as the maximum NBR_ATOM-to-atom distance over *all* conformers,
    plus `margin`. Erring large is safe; erring small is not.
    """
    params_path = _abs(params_path)
    out_path = _abs(out_path or params_path)
    names = params_atom_names(params_path)
    _require(atom_name in names,
             f"{atom_name!r} is not an atom of {os.path.basename(params_path)}; "
             f"available: {names}")

    radius = None
    if recompute_radius:
        _require(conformers_pdb, "recompute_radius needs conformers_pdb")
        blocks = read_conformers(conformers_pdb)
        _require(blocks, f"no conformers in {conformers_pdb}")
        worst = 0.0
        for b in blocks:
            missing = [n for n in names if n not in b]
            _require(not missing,
                     f"conformer file is missing atoms {missing[:5]} present in params")
            o = np.array(b[atom_name])
            pts = np.array([b[n] for n in names])
            worst = max(worst, float(np.linalg.norm(pts - o, axis=1).max()))
        radius = worst + float(margin)

    old_nbr = params_nbr_atom(params_path)
    old_rad = params_nbr_radius(params_path)
    out = []
    for ln in _params_lines(params_path):
        if ln.startswith("NBR_ATOM"):
            out.append(f"NBR_ATOM {atom_name}\n")
        elif ln.startswith("NBR_RADIUS") and radius is not None:
            out.append(f"NBR_RADIUS {radius:.6f}\n")
        else:
            out.append(ln)
    with open(out_path, "w") as fh:
        fh.writelines(out)
    print(f"NBR_ATOM   {old_nbr} -> {atom_name}")
    if radius is not None:
        print(f"NBR_RADIUS {old_rad} -> {radius:.6f}   "
              f"(max over {len(blocks)} conformers + {margin} A margin)")
    _say(_OK, f"wrote {out_path}")
    return out_path


def check_start_pdb_vs_params(start_pdb, params_path, ligand_chain):
    """
    Verify the ligand in start.pdb matches lig.params.

    Rosetta matches ligand atoms to the residue type by NAME, so the atom-name
    set must be identical; file order only has to be self-consistent (LPF
    re-reads it every window), so an order difference is reported, not fatal.
    The upstream demo's own start.pdb is a permutation of its params order.
    """
    atoms = [a for a in read_pdb_atoms(start_pdb) if a.chain == ligand_chain]
    ok = True
    if not atoms:
        _say(_BAD, f"no atoms at all in chain {ligand_chain!r} of {start_pdb}")
        return False
    het = [a for a in atoms if a.record == "HETATM"]
    if len(het) != len(atoms):
        _say(_BAD, f"chain {ligand_chain!r} has {len(atoms) - len(het)} ATOM records; "
                   "LPF's read_window only sees HETATM")
        ok = False
    resids = {(a.resseq, a.icode) for a in atoms}
    if len(resids) != 1:
        _say(_BAD, f"chain {ligand_chain!r} spans {len(resids)} residues {sorted(resids)}; "
                   "LPF treats the ligand as exactly one residue")
        ok = False

    p_names = params_atom_names(params_path)
    s_names = [a.name for a in atoms]
    missing = [n for n in p_names if n not in s_names]
    extra = [n for n in s_names if n not in p_names]
    dupes = sorted({n for n in s_names if s_names.count(n) > 1})
    if missing:
        _say(_BAD, f"atoms in params but not in start.pdb: {missing}")
        ok = False
    if extra:
        _say(_BAD, f"atoms in start.pdb but not in params: {extra}")
        ok = False
    if dupes:
        _say(_BAD, f"duplicate atom names in the ligand: {dupes}")
        ok = False

    rn_params = params_resname(params_path)
    rn_start = atoms[0].resname
    if rn_start != rn_params:
        _say(_BAD, f"residue name mismatch: start.pdb has {rn_start!r}, "
                   f"params IO_STRING says {rn_params!r}")
        ok = False

    nbr = params_nbr_atom(params_path)
    if nbr not in s_names:
        _say(_BAD, f"NBR_ATOM {nbr!r} is absent from the ligand in start.pdb")
        ok = False

    # the ligand must come after the protein: prepare_start() streams ATOM
    # records and non-ligand HETATMs into the protein block only until it first
    # sees the ligand chain.
    all_atoms = read_pdb_atoms(start_pdb)
    first_lig = next(i for i, a in enumerate(all_atoms) if a.chain == ligand_chain)
    after = [a for a in all_atoms[first_lig:] if a.chain != ligand_chain]
    if after:
        chains = sorted({a.chain for a in after})
        _say(_BAD, f"{len(after)} non-ligand atoms (chains {chains}) appear AFTER the "
                   "ligand; the ligand must be last in start.pdb")
        ok = False

    if ok:
        if s_names != p_names:
            _say(_OK, f"ligand matches params: {len(s_names)} atoms, 1 residue "
                      f"{rn_start} {ligand_chain}{atoms[0].resseq}, NBR_ATOM {nbr} "
                      "(atom order differs from params, which is fine)")
        else:
            _say(_OK, f"ligand matches params exactly: {len(s_names)} atoms, "
                      f"residue {rn_start} {ligand_chain}{atoms[0].resseq}, NBR_ATOM {nbr}")
    return ok


def assemble_start_pdb(protein_pdb, ligand_pdb, out_pdb, params_path,
                       ligand_chain="X", ligand_resnum=1, ligand_chain_in_source=None,
                       b_factor=20.0):
    """
    Build the start.pdb LPF expects: protein first, ligand last, as a single
    HETATM residue in `ligand_chain`, named and ordered per lig.params.

    `ligand_pdb` is normally the lig.pdb that molfile_to_params writes, whose
    coordinates are the first conformer of the input SDF — i.e. the bound pose,
    already carrying Rosetta's atom names. That removes the manual PyMOL
    renaming step the upstream tutorial describes (demo/tutorial.md 1.2.4).
    """
    p_names = params_atom_names(params_path)
    resname = params_resname(params_path)

    prot = [a for a in read_pdb_atoms(protein_pdb)
            if ligand_chain_in_source is None or a.chain != ligand_chain_in_source]
    _require(prot, f"no protein atoms in {protein_pdb}")
    if any(a.chain == ligand_chain for a in prot):
        raise ValueError(
            f"the protein file already uses chain {ligand_chain!r}; pick a free chain "
            f"for the ligand (protein chains: {sorted({a.chain for a in prot})})")

    lig_atoms = [a for a in read_pdb_atoms(ligand_pdb)]
    if ligand_chain_in_source is not None:
        lig_atoms = [a for a in lig_atoms if a.chain == ligand_chain_in_source]
    _require(lig_atoms, f"no ligand atoms in {ligand_pdb}")
    by_name = {}
    for a in lig_atoms:
        if a.name in by_name:
            raise ValueError(f"duplicate atom name {a.name!r} in {ligand_pdb}")
        by_name[a.name] = a
    missing = [n for n in p_names if n not in by_name]
    if missing:
        raise ValueError(
            f"{os.path.basename(ligand_pdb)} is missing {len(missing)} atom(s) that "
            f"lig.params declares: {missing[:10]}{' ...' if len(missing) > 10 else ''}\n"
            f"Use the lig.pdb written by molfile_to_params (its names match params).")
    unknown = [n for n in by_name if n not in p_names]
    if unknown:
        raise ValueError(
            f"{os.path.basename(ligand_pdb)} has atom(s) absent from lig.params: "
            f"{unknown[:10]}{' ...' if len(unknown) > 10 else ''}")

    # protein block: canonical residues and kept cofactors, in file order
    out = []
    for a in prot:
        b = Atom(**a.__dict__)
        out.append(b)
    for n in p_names:                       # ligand block, in params order
        src = by_name[n]
        out.append(Atom(
            record="HETATM", name=n, resname=resname, chain=ligand_chain,
            resseq=int(ligand_resnum), x=src.x, y=src.y, z=src.z,
            occ=1.0, bfac=b_factor,
            element=_infer_element(n, src.element),
        ))
    write_pdb_atoms(out_pdb, out)
    print(f"assemble_start_pdb: {len(prot)} protein atoms + {len(p_names)} ligand atoms "
          f"-> {_abs(out_pdb)}")
    check_start_pdb_vs_params(out_pdb, params_path, ligand_chain)
    return _abs(out_pdb)


# --------------------------------------------------------------------------
# start-pose minimisation (tutorial section 1.3)
#
# LPF expects a start pose already relaxed into the Rosetta force field. The
# tutorial does this with the standalone `minimize` application:
#
#   minimize.static.linuxgccrelease -s complex.pdb \
#       -extra_res_fa lig.params -restore_pre_talaris_2013_behavior
#
# and recommends several -nstruct attempts, picking the best. That is a
# different binary from the one LPF needs, so the equivalent is done here as a
# RosettaScripts protocol through the `rosetta_scripts` binary you already
# configured — which also means it works on the PyRosetta backend.
#
# This is not the same as window 0 of the trajectory: that pass
# (zero_minimize.xml) only settles the interface region under the LigandArea
# movemap. An unrelaxed start pose shows up as a huge window-0 energy.
# --------------------------------------------------------------------------

MINIMIZE_XML = """<ROSETTASCRIPTS>
    <SCOREFXNS>
        <ScoreFunction name="hard_rep" weights="ligand">
            <Reweight scoretype="fa_rep" weight="0.80"/>
            <Reweight scoretype="fa_elec" weight="0.42"/>
            <Reweight scoretype="fa_sol" weight="0.65"/>
            <Reweight scoretype="fa_intra_rep" weight="1.0"/>
$coord_cst_reweight$
        </ScoreFunction>
    </SCOREFXNS>
    <MOVERS>
$coord_cst_mover$
        <MinMover name="minimize" scorefxn="hard_rep" chi="1" bb="$bb$"
                  jump="ALL" tolerance="0.0001" max_iter="2000"/>
        <InterfaceScoreCalculator name="add_scores" chains="$chain$"
                                  scorefxn="hard_rep"/>
    </MOVERS>
    <PROTOCOLS>
$coord_cst_add$
        <Add mover_name="minimize"/>
        <Add mover_name="add_scores"/>
    </PROTOCOLS>
</ROSETTASCRIPTS>
"""


def write_minimize_xml(path, ligand_chain="X", minimize_backbone=True,
                       ca_restraint_sd=0.5):
    """
    Write the minimisation protocol used by minimize_start_pose().

    `ca_restraint_sd` adds coordinate restraints so the fold cannot drift. The
    tutorial's `minimize` command has none — pass 0 to match it exactly — but a
    free all-atom minimisation of a whole protein can distort it, and the
    tutorial itself says manual inspection is "highly recommended". The default
    keeps the backbone near its input positions while letting the interface
    settle.
    """
    xml = MINIMIZE_XML
    if ca_restraint_sd and float(ca_restraint_sd) > 0:
        xml = xml.replace(
            "$coord_cst_reweight$",
            '            <Reweight scoretype="coordinate_constraint" weight="1.0"/>')
        xml = xml.replace(
            "$coord_cst_mover$",
            f'        <AddConstraints name="add_coord_cst">\n'
            f'            <CoordinateConstraintGenerator name="coord_cst"\n'
            f'                sd="{float(ca_restraint_sd):.3f}" ca_only="true"\n'
            f'                residue_selector="prot"/>\n'
            f'        </AddConstraints>')
        xml = xml.replace("$coord_cst_add$",
                          '        <Add mover_name="add_coord_cst"/>')
        xml = xml.replace(
            "<MOVERS>",
            '<RESIDUE_SELECTORS>\n'
            '        <Chain name="lig" chains="$chain$"/>\n'
            '        <Not name="prot" selector="lig"/>\n'
            '    </RESIDUE_SELECTORS>\n\n    <MOVERS>')
    else:
        for tag in ("$coord_cst_reweight$", "$coord_cst_mover$", "$coord_cst_add$"):
            xml = xml.replace(tag + "\n", "")
            xml = xml.replace(tag, "")
    xml = xml.replace("$chain$", ligand_chain)
    xml = xml.replace("$bb$", "1" if minimize_backbone else "0")

    import xml.etree.ElementTree as ET
    ET.fromstring(xml)                      # fail here rather than inside Rosetta
    path = _abs(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(xml)
    return path


def _run_rosetta_raw(argv, cwd, log_path, err_path):
    """
    Run a Rosetta command, raising with the log tail if it fails.

    Deliberately not LPF's util.run_rosetta: this module stays importable
    without LigandPathFinder on sys.path.
    """
    with open(log_path, "w") as log_fh, open(err_path, "w") as err_fh:
        proc = subprocess.run(argv, cwd=cwd, stdout=log_fh, stderr=err_fh)
    if proc.returncode != 0:
        def tail(p, n=25):
            try:
                return "".join(open(p, errors="replace").readlines()[-n:]).rstrip()
            except OSError:
                return ""
        raise RuntimeError(
            f"Rosetta exited with status {proc.returncode}.\n"
            f"  command: {' '.join(argv)}\n  workdir: {cwd}\n"
            + (f"  {os.path.basename(err_path)}:\n"
               + textwrap.indent(tail(err_path), "    ") + "\n"
               if tail(err_path) else "")
            + (f"  {os.path.basename(log_path)}:\n"
               + textwrap.indent(tail(log_path), "    ")
               if tail(log_path) else ""))
    return proc


def read_score_file(path):
    """Parse a Rosetta score file into a list of dicts (floats where possible)."""
    lines = open(_abs(path), errors="replace").readlines()
    _require(len(lines) >= 3, f"score file has no data rows: {path}")
    rows = [l.strip().split() for l in lines[1:]]
    header = rows[0]
    out = []
    for r in rows[1:]:
        if len(r) != len(header):
            continue
        rec = {}
        for k, v in zip(header, r):
            try:
                rec[k] = float(v)
            except ValueError:
                rec[k] = v
        out.append(rec)
    _require(out, f"no usable rows in {path}")
    return out


def minimize_start_pose(start_pdb, out_pdb, rosetta, files_dir, ligand_dir,
                        work_dir=None, ligand_chain="X", nstruct=5,
                        extra_files=(), minimize_backbone=True,
                        ca_restraint_sd=0.5, n_proc=1, mpi_launcher="mpirun",
                        keep_work_dir=False, verbose=True):
    """
    Relax the start complex into the Rosetta force field, the tutorial's step 1.3.

    Runs `nstruct` independent minimisations and keeps the best by total_score
    (the tutorial's "run several attempts and pick the best"), then rebuilds the
    result into the canonical start.pdb layout LPF needs and re-validates it
    against lig.params.

    Returns the path to the minimised, validated start.pdb. Inspect it before
    trusting it — the tutorial is emphatic about that, and minimisation is the
    step most likely to need system-specific attention (metalloenzymes,
    non-standard protonation, structural waters).
    """
    start_pdb = _abs(start_pdb)
    files_dir, ligand_dir = _abs(files_dir), _abs(ligand_dir)
    work_dir = _abs(work_dir or os.path.join(os.path.dirname(_abs(out_pdb)),
                                             "_minimize"))
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(os.path.join(work_dir, "results"))
    os.makedirs(os.path.join(work_dir, "grid"))      # -qsar:grid_dir in options

    lig_params = os.path.join(ligand_dir, "lig.params")
    for src, dst in [(os.path.join(files_dir, "options"), "options"),
                     (lig_params, None),
                     (os.path.join(ligand_dir, "lig_conformers.pdb"), None),
                     (os.path.join(files_dir, "VRT1.params"), None)]:
        if not os.path.isfile(src):
            raise FileNotFoundError(f"minimize_start_pose: missing {src}")
        shutil.copy(src, os.path.join(work_dir, dst) if dst else work_dir)
    for f in extra_files:
        shutil.copy(_abs(f), work_dir)
    shutil.copy(start_pdb, os.path.join(work_dir, "input.pdb"))
    write_minimize_xml(os.path.join(work_dir, "minimize.xml"),
                       ligand_chain=ligand_chain,
                       minimize_backbone=minimize_backbone,
                       ca_restraint_sd=ca_restraint_sd)

    rb = _abs(rosetta)
    argv = []
    eff = min(int(n_proc), int(nstruct))
    if eff > 1:
        argv += [mpi_launcher, "-np", str(eff)]
    argv += [rb, "@options", "-nstruct", str(int(nstruct)),
             "-out:path:pdb", "./results", "-parser:protocol", "minimize.xml",
             "-in:file:s", "input.pdb"]
    if verbose:
        print(f"minimising {os.path.basename(start_pdb)} with {nstruct} attempt(s)")
        print("$ " + " ".join(argv))
    _run_rosetta_raw(argv, cwd=work_dir,
                     log_path=os.path.join(work_dir, "minimize.log"),
                     err_path=os.path.join(work_dir, "minimize.err"))

    rows = read_score_file(os.path.join(work_dir, "score.sc"))
    cst_cols = [c for c in ("atom_pair_constraint", "angle_constraint",
                            "dihedral_constraint", "coordinate_constraint")]

    def clean_total(r):
        return r["total_score"] - sum(float(r.get(c, 0.0)) for c in cst_cols)

    rows.sort(key=clean_total)
    best = None
    for r in rows:
        cand = os.path.join(work_dir, "results", str(r["description"]) + ".pdb")
        if os.path.isfile(cand):
            best = (cand, r)
            break
    if best is None:
        raise RuntimeError(
            f"minimisation produced a score file but no readable decoy PDB in "
            f"{work_dir}/results")
    best_pdb, best_row = best
    if verbose:
        scores = [clean_total(r) for r in rows]
        print(f"\n{len(rows)} attempt(s): total_score (constraint-free) "
              f"best {min(scores):.1f}, worst {max(scores):.1f}")
        print(f"keeping {os.path.basename(best_pdb)}")

    # Rosetta may reorder the ligand's atoms and emit its own TER/END records, so
    # rebuild into the layout LPF requires rather than trusting the raw output.
    out_pdb = _abs(out_pdb)
    assemble_start_pdb(best_pdb, best_pdb, out_pdb, params_path=lig_params,
                       ligand_chain=ligand_chain,
                       ligand_resnum=[a.resseq for a in read_pdb_atoms(best_pdb)
                                      if a.chain == ligand_chain][0],
                       ligand_chain_in_source=ligand_chain)
    if not keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    elif verbose:
        print(f"       working directory kept: {work_dir}")

    if verbose:
        before = read_pdb_atoms(start_pdb)
        after = read_pdb_atoms(out_pdb)
        common = {a.key: a.xyz for a in before}
        moved = [float(np.linalg.norm(a.xyz - common[a.key]))
                 for a in after if a.key in common]
        if moved:
            moved = np.array(moved)
            print(f"\natoms moved: mean {moved.mean():.2f} A, "
                  f"max {moved.max():.2f} A, {int((moved > 1.0).sum())} atoms > 1 A")
            if moved.max() > 5.0:
                _say(_WARN, "something moved more than 5 A — inspect the result "
                            "before using it. Consider ca_restraint_sd smaller, or "
                            "minimize_backbone=False.")
        print()
        print("Inspect this structure before running LPF on it. The tutorial calls "
              "manual verification\nhighly recommended, and relaxing an "
              "enzyme-substrate complex is the step most likely\nto need "
              "system-specific attention.")
    return out_pdb


# --------------------------------------------------------------------------
# guide path: where does the ligand leave?
#
# The coarse stage needs exactly two points: the ligand's NBR_ATOM position and
# a point roughly outside the protein. Everything between them is generated by
# bisection. These helpers propose the second point instead of making you pick
# coordinates by eye in PyMOL.
# --------------------------------------------------------------------------

@dataclass
class ExitRay:
    direction: np.ndarray
    bottleneck: float      # narrowest clearance to any protein heavy atom along the ray
    reach: float           # how far the ray got before clearance fell below clash_radius
    escaped: bool          # reached open solvent
    endpoint: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __repr__(self):
        d = ", ".join(f"{v:6.3f}" for v in self.direction)
        return (f"ExitRay(dir=[{d}], bottleneck={self.bottleneck:5.2f} A, "
                f"reach={self.reach:5.1f} A, escaped={self.escaped})")


def protein_heavy_coords(pdb, exclude_chain=None, include_hetero=True):
    """Heavy-atom coordinates of everything that is not the ligand."""
    pts = []
    for a in read_pdb_atoms(pdb):
        if exclude_chain is not None and a.chain == exclude_chain:
            continue
        if not include_hetero and a.record == "HETATM":
            continue
        if _infer_element(a.name, a.element, a.name_col13) in ("H", "D"):
            continue
        pts.append((a.x, a.y, a.z))
    _require(pts, f"no heavy atoms found in {pdb}")
    return np.array(pts, dtype=float)


def _fibonacci_directions(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])


def exit_direction_scan(protein_coords, start_point, n_dirs=512, max_dist=50.0,
                        sample_step=0.5, clash_radius=3.2, hint=None,
                        hint_cone_deg=None, open_clearance=7.0, top=8, verbose=True):
    """
    Cast rays from the binding pose and rank them by how wide the exit is.

    For each direction the ray is sampled every `sample_step` A; at each sample
    the distance to the nearest protein heavy atom is the local clearance. The
    ray's *bottleneck* is the smallest clearance before it either reaches open
    solvent (clearance > open_clearance) or gets pinched below `clash_radius`.

    Rays that escape are ranked by bottleneck, widest first: that bottleneck is
    roughly the radius of the narrowest point of the tunnel, so compare it with
    the radius of your ligand. `hint` + `hint_cone_deg` restrict the scan to a
    cone (use it when you already know which way the ligand should go).
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(protein_coords, dtype=float))
    start = np.asarray(start_point, dtype=float).reshape(3)

    dirs = _fibonacci_directions(int(n_dirs))
    if hint is not None:
        h = np.asarray(hint, dtype=float).reshape(3)
        _require(np.linalg.norm(h) > 0, "hint must be a non-zero vector")
        h = h / np.linalg.norm(h)
        if hint_cone_deg is not None:
            keep = dirs @ h >= math.cos(math.radians(hint_cone_deg))
            _require(keep.any(), f"no sampled direction lies within "
                                 f"{hint_cone_deg} deg of the hint")
            dirs = dirs[keep]

    ts = np.arange(sample_step, float(max_dist) + 1e-9, sample_step)
    rays = []
    for d in dirs:
        pts = start + np.outer(ts, d)
        clear, _ = tree.query(pts)
        bottleneck = float("inf")
        reach, escaped, endpoint = float(ts[-1]), False, pts[-1]
        for k, c in enumerate(clear):
            if c < clash_radius:
                reach, endpoint = float(ts[k]), pts[k]
                break
            bottleneck = min(bottleneck, float(c))
            if c > open_clearance:
                reach, escaped, endpoint = float(ts[k]), True, pts[k]
                break
        else:
            escaped = bool(clear[-1] > open_clearance)
        if not np.isfinite(bottleneck):
            bottleneck = float(clear[0])
        rays.append(ExitRay(direction=d, bottleneck=bottleneck, reach=reach,
                            escaped=escaped, endpoint=endpoint))

    rays.sort(key=lambda r: (not r.escaped, -r.bottleneck, r.reach))
    if verbose:
        n_esc = sum(r.escaped for r in rays)
        print(f"scanned {len(rays)} directions from "
              f"[{start[0]:.3f} {start[1]:.3f} {start[2]:.3f}]; "
              f"{n_esc} reach open solvent (clearance > {open_clearance} A)")
        print(f"  {'#':>3s} {'bottleneck':>11s} {'exit at':>9s}  direction")
        for i, r in enumerate(rays[:top]):
            d = " ".join(f"{v:6.3f}" for v in r.direction)
            print(f"  {i:>3d} {r.bottleneck:>9.2f} A {r.reach:>7.1f} A  [{d}]"
                  f"{'' if r.escaped else '   (blocked)'}")
        if n_esc == 0:
            _say(_WARN, "no ray escaped: the pose may be deeply buried. Raise max_dist, "
                        "lower open_clearance, or supply the exit direction by hand.")
    return rays


def exit_point_along(protein_coords, start_point, direction, clearance=8.0,
                     min_dist=12.0, max_dist=60.0, sample_step=0.5,
                     ligand_span=None):
    """
    Walk along `direction` until the point sits at least `clearance` A from any
    protein heavy atom (and at least `min_dist` A from the start), and return it.

    This is the second line of the coarse-stage `path` file. The coarse run
    jitters it anyway (endpoint_jitter), so it only has to be roughly right.

    `clearance` is measured from the TRACKED ATOM, not from the whole molecule,
    so for an elongated ligand the far end is still inside the protein when the
    tracked atom first reaches open solvent. Pass `ligand_span` and this reports
    whether `min_dist` is long enough for the whole body to get out — which
    matters for a peptide in a shallow surface groove, where the first escaping
    point can be only a few angstroms away.
    """
    if ligand_span:
        want = float(ligand_span) / 2.0 + 10.0
        if min_dist < want:
            _say(_WARN, f"min_dist={min_dist:.0f} A is short for a molecule spanning "
                        f"{float(ligand_span):.0f} A: the tracked atom would be clear "
                        f"while the far end is still buried. Consider "
                        f"min_dist={want:.0f} A so the whole body reaches solvent.")
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(protein_coords, dtype=float))
    start = np.asarray(start_point, dtype=float).reshape(3)
    d = np.asarray(direction, dtype=float).reshape(3)
    _require(np.linalg.norm(d) > 0, "direction must be a non-zero vector")
    d = d / np.linalg.norm(d)

    ts = np.arange(sample_step, float(max_dist) + 1e-9, sample_step)
    pts = start + np.outer(ts, d)
    dist, _ = tree.query(pts)
    for t, p, c in zip(ts, pts, dist):
        if t >= min_dist and c >= clearance:
            print(f"exit point at {t:.1f} A along the ray, {c:.1f} A clear of the protein")
            return p
    k = int(np.argmax(np.where(ts >= min_dist, dist, -1)))
    _say(_WARN, f"never reached {clearance} A clearance within {max_dist} A; using the "
                f"clearest point at {ts[k]:.1f} A ({dist[k]:.1f} A clear)")
    return pts[k]


def write_two_point_path(path_file, start_point, end_point):
    """Write the coarse stage's two-line `path` file (start, then exit direction)."""
    p0 = np.asarray(start_point, dtype=float).reshape(3)
    p1 = np.asarray(end_point, dtype=float).reshape(3)
    _require(np.linalg.norm(p1 - p0) > 1e-6, "the two path points are identical")
    path_file = _abs(path_file)
    os.makedirs(os.path.dirname(path_file) or ".", exist_ok=True)
    with open(path_file, "w") as fh:
        for p in (p0, p1):
            fh.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")
    print(f"path ({np.linalg.norm(p1 - p0):.1f} A apart) -> {path_file}")
    print(f"    {p0[0]:8.3f} {p0[1]:8.3f} {p0[2]:8.3f}   (ligand NBR_ATOM)")
    print(f"    {p1[0]:8.3f} {p1[1]:8.3f} {p1[2]:8.3f}   (exit direction)")
    return path_file


def write_path_points(path_file, points, as_pdb=False):
    """Write a multi-point guide path (text xyz, or a PDB of connected points)."""
    pts = np.asarray(points, dtype=float)
    _require(pts.ndim == 2 and pts.shape[1] == 3, f"points must be (N,3), got {pts.shape}")
    path_file = _abs(path_file)
    os.makedirs(os.path.dirname(path_file) or ".", exist_ok=True)
    if as_pdb:
        with open(path_file, "w") as fh:
            for i, (x, y, z) in enumerate(pts, start=1):
                fh.write(f"HETATM{i:5d}  C   GP  A{i:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
            for i in range(1, len(pts)):
                fh.write(f"CONECT{i:5d}{i+1:5d}\n")
    else:
        np.savetxt(path_file, pts, fmt="%.3f")
    print(f"{len(pts)} path points -> {path_file}")
    return path_file


def path_clearance_report(points, protein_coords, ligand_radius=None, warn_below=2.5):
    """
    Clearance of each guide point from the protein, with the tight spots flagged.

    Run this on coarsepath.txt before paying for a trajectory: a guide path that
    threads through the protein wall will stall the trajectory stage.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(points, dtype=float)
    tree = cKDTree(np.asarray(protein_coords, dtype=float))
    dist, _ = tree.query(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    print(f"{len(pts)} points, path length {seg.sum():.1f} A, "
          f"spacing {seg.min():.2f}-{seg.max():.2f} A (mean {seg.mean():.2f})")
    print(f"clearance to nearest protein heavy atom: min {dist.min():.2f} A, "
          f"median {np.median(dist):.2f} A, max {dist.max():.2f} A")
    tight = np.where(dist < warn_below)[0]
    if len(tight):
        _say(_WARN, f"{len(tight)} point(s) within {warn_below} A of the protein: "
                    f"{tight[:15].tolist()}{' ...' if len(tight) > 15 else ''}")
    if ligand_radius:
        print(f"ligand radius ~{ligand_radius:.1f} A — the path's narrowest clearance is "
              f"{dist.min():.1f} A; the protein has to open up for the rest.")
    return dist


def read_guide_points(path):
    """
    Read a guide path as an (N,3) array, from text or PDB.

    Mirrors LPF's util.read_guide_path: a whitespace-delimited `x y z` text file,
    or a PDB whose ATOM/HETATM coordinates are the points. Use this rather than
    np.loadtxt so a CAVER tunnel PDB works wherever a text path does.
    """
    p = _abs(path)
    _require(os.path.isfile(p), f"guide path not found: {p}")
    if p.lower().endswith(".pdb"):
        pts = np.array([a.xyz for a in read_pdb_atoms(p)], dtype=float)
    else:
        pts = np.atleast_2d(np.loadtxt(p))
    _require(pts.ndim == 2 and pts.shape[1] == 3,
             f"{p} is not an (N,3) list of points (got shape {pts.shape})")
    return pts


def check_path_direction(points, start_point, auto_reverse=False):
    """
    Verify an externally supplied path runs AWAY from where the ligand actually is.

    Both LPF stages overwrite the path's first point with the ligand's current
    NBR_ATOM position whenever the two are more than step/5 apart
    (CoarsePathRunner.__init__, TrajectoryRunner.prepare_path). That silently
    repairs a path whose start is merely imprecise — but it also means a path
    supplied back-to-front has its *outer* end replaced by the binding-site
    position, producing a route that goes nowhere sensible without any error.

    Tunnel finders do not guarantee an orientation, so a CAVER or AQUA-DUCT
    export can easily come out outside-in. This compares both ends against the
    ligand's real position and says which way it runs.

    Returns (points, was_reversed); with auto_reverse=True the points come back
    flipped so they run outward.
    """
    pts = np.asarray(points, dtype=float)
    _require(pts.ndim == 2 and pts.shape[1] == 3,
             f"points must be (N,3), got {pts.shape}")
    _require(len(pts) >= 2, "need at least two points")
    s = np.asarray(start_point, dtype=float).reshape(3)

    d_first = float(np.linalg.norm(pts[0] - s))
    d_last = float(np.linalg.norm(pts[-1] - s))
    print(f"ligand NBR_ATOM to path start {d_first:6.2f} A, to path end {d_last:6.2f} A")

    if d_last >= d_first:
        _say(_OK, "the path runs away from the ligand, as LPF expects")
        return pts, False

    _say(_BAD, f"this path appears REVERSED: its last point is nearer the ligand "
               f"({d_last:.2f} A) than its first ({d_first:.2f} A).")
    print("       LPF would overwrite point 0 with the ligand's position, so the "
          "outer end would be replaced by the binding site and the route would "
          "be nonsense,")
    print("       without raising anything. Reverse it before use.")
    if auto_reverse:
        _say(_OK, "reversed for you")
        return pts[::-1].copy(), True
    return pts, True


def check_path_anchor(points, start_point, dock_zero=False, tolerance=2.0,
                      verbose=True):
    """
    Verify the guide path actually starts where the ligand is — or that dock_zero
    is on, which is the case where it need not.

    Both stages replace the path's first point with the ligand's NBR_ATOM
    position whenever the two are more than step/5 apart, *unless* dock_zero is
    set. For a path that merely starts imprecisely that is a silent repair. For
    a path that starts somewhere else entirely it is a trap: the first point
    moves tens of angstroms to the ligand while the second stays put, so the
    first segment becomes a huge jump against ~step-sized spacing elsewhere, and
    the smoothing spline through that is meaningless.

    This is exactly the configuration upstream examples/trajectory ships — a
    path 30 A from the ligand — and it sets dock_zero=true, which docks the
    ligand onto the path's first point before the algorithm starts.

    Returns True when the combination is sound.
    """
    pts = np.asarray(points, dtype=float)
    _require(pts.ndim == 2 and len(pts) >= 2, "need an (N,3) path with 2+ points")
    s = np.asarray(start_point, dtype=float).reshape(3)
    d = float(np.linalg.norm(pts[0] - s))
    seg = float(np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    if d <= tolerance:
        if verbose:
            _say(_OK, f"the path starts on the ligand ({d:.2f} A from NBR_ATOM)")
        return True
    if dock_zero:
        if verbose:
            _say(_OK, f"the path starts {d:.1f} A from the ligand, and dock_zero is "
                      f"on — the ligand will be docked onto the path's first point "
                      f"before the run starts")
        return True
    if verbose:
        _say(_BAD, f"the path starts {d:.1f} A from the ligand's NBR_ATOM and "
                   f"dock_zero is off.")
        print(f"       LPF will move point 0 onto the ligand, leaving a {d:.0f} A "
              f"first segment")
        print(f"       against {seg:.2f} A spacing elsewhere — the spline through "
              f"that is meaningless.")
        print("       Fix one of:")
        print("         * turn dock_zero on, to dock the ligand onto the path's "
              "first point")
        print("           (needs zero_dock.xml and nstruct_zero; this is what "
              "upstream")
        print("           examples/trajectory does)")
        print("         * supply a path that starts at the ligand's NBR_ATOM")
        print("         * use a start pose whose ligand sits at the path's start")
    return False


def coarse_path_medoid(path_files, n_resample=40):
    """
    Pick the medoid of several coarse-path replicates.

    The coarse stage is stochastic (and jitters its endpoint), so the paper's
    recipe is to run it several times and carry the medoid forward. Paths are
    resampled to a common length by arc length, then compared by mean
    point-to-point distance.
    """
    paths = []
    for f in path_files:
        p = np.loadtxt(_abs(f))
        _require(p.ndim == 2 and p.shape[1] == 3, f"{f} is not an (N,3) path")
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        if s[-1] <= 0:
            continue
        u = np.linspace(0, s[-1], n_resample)
        paths.append(np.column_stack([np.interp(u, s, p[:, k]) for k in range(3)]))
    _require(len(paths) >= 1, "no usable paths")
    if len(paths) == 1:
        print("only one path given; returning it")
        return _abs(path_files[0]), 0, np.zeros((1, 1))
    A = np.stack(paths)
    n = len(A)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(np.linalg.norm(A[i] - A[j], axis=1).mean())
    k = int(np.argmin(D.sum(axis=1)))
    print(f"{n} replicates, mean pairwise distance {D[np.triu_indices(n, 1)].mean():.2f} A")
    for i, f in enumerate(path_files):
        mark = "  <- medoid" if i == k else ""
        print(f"  {os.path.basename(os.path.dirname(_abs(f))) or i:>20s}  "
              f"mean distance to others {D[i].sum() / (n - 1):6.2f} A{mark}")
    return _abs(path_files[k]), k, D


# --------------------------------------------------------------------------
# RosettaScripts protocols and the Rosetta options file
#
# The shipped XMLs hardcode chain="X" in LigandArea, SCORINGGRIDS, Transform,
# StartFrom and InterfaceScoreCalculator, and size the scoring grid at 20 A.
# Rosetta never reads input.ligand_chain from run.json, so a different chain or
# a larger "ligand" needs the XMLs rewritten. That is what this does: it starts
# from the upstream files and retargets them, rather than shipping forks that
# would drift from the repository.
# --------------------------------------------------------------------------

_XML_FILES_COARSE = ("coarsepath.xml",)
_XML_FILES_TRAJ = ("main_standard.xml", "zero_minimize.xml", "stats.xml", "zero_dock.xml")
#: Alternative per-window protocols. Not written by default: main_rigid.xml
#: deliberately zeroes every interface cutoff and main_extended.xml deliberately
#: widens them, so blanket retuning would erase the only thing that makes them
#: different from main_standard.xml. Ask for them explicitly if you want them.
_XML_FILES_OPTIONAL = ("main_extended.xml", "main_rigid.xml")
_NEEDS_PLACEHOLDER = ("coarsepath.xml", "main_standard.xml", "main_extended.xml",
                      "main_rigid.xml", "zero_dock.xml")

#: Files whose character would be destroyed by the peptide tuning rules.
_TUNING_SENSITIVE = ("main_rigid.xml", "main_extended.xml")

#: The per-window protocols the trajectory stage actually runs.
_MAIN_LIKE = ("main_standard.xml", "zero_dock.xml")

#: Upstream values, unchanged. Small-molecule ligands need no retuning.
LIGAND_XML_TUNING = {"grid_width": None, "rules": []}

#: Peptide-sized "ligands": wider interface detection so more of the protein can
#: respond, a looser C-alpha restraint sd so the backbone can actually open,
#: a bigger Transform box, and more docking cycles per window. The scoring grid
#: is sized from the molecule itself (see auto_grid_width) — width=20 does not
#: contain a peptide, and a ligand poking outside its grid scores nonsense.
PEPTIDE_XML_TUNING = {
    "grid_width": None,
    "rules": [
        # Wider interface detection: a peptide contacts far more of the protein
        # than a small molecule, and only residues inside these cutoffs are
        # allowed to repack or move at all.
        {"tag": "LigandArea", "name": "inhibitor_dock_sc", "attrs": {"cutoff": "8.0"}},
        {"tag": "LigandArea", "name": "inhibitor_final_sc", "attrs": {"cutoff": "8.0"}},
        # coarsepath.xml alone sets add_nbr_radius="true", which adds the residue's
        # NBR_RADIUS to the cutoff. That is a couple of angstroms for a small
        # ligand but 20+ for a peptide, which would drag the whole protein into
        # every repack. Turn it off and rely on the explicit cutoff.
        {"tag": "LigandArea", "name": None, "attrs": {"add_nbr_radius": "false"},
         "files": ("coarsepath.xml",)},
        {"tag": "LigandArea", "name": "inhibitor_final_bb",
         "attrs": {"cutoff": "10.0", "Calpha_restraints": "0.5"},
         # not stats.xml (a rescoring pass) and not zero_minimize.xml (which only
         # settles the supplied pose and should stay tight)
         "files": _MAIN_LIKE + ("coarsepath.xml",)},
        {"tag": "DetectProteinLigandInterface", "name": "pack_interface",
         "attrs": {"cut1": "7.0", "cut2": "9.0", "cut3": "11.0", "cut4": "13.0"},
         "files": _MAIN_LIKE},

        # Per-window sampling: a bigger translation box, more cycles, and only a
        # small initial reorientation — spinning a peptide freely inside a
        # channel just produces clashes.
        {"tag": "Transform", "name": "transform",
         "attrs": {"box_size": "8", "cycles": "500", "repeats": "2",
                   "initial_angle_perturb": "5"},
         "files": _MAIN_LIKE},
        {"tag": "Transform", "name": "transform_2",
         "attrs": {"box_size": "8", "cycles": "200", "repeats": "2"},
         "files": _MAIN_LIKE},
        {"tag": "HighResDocker", "name": "high_res_docker", "attrs": {"cycles": "15"},
         "files": _MAIN_LIKE + ("stats.xml",)},

        # The coarse stage is meant to be cheap, so its cycle counts stay as
        # upstream; only the 90-degree initial reorientation is dialled back,
        # because it is hopeless for an elongated molecule in a tunnel.
        {"tag": "Transform", "name": "transform",
         "attrs": {"initial_angle_perturb": "20"},
         "files": ("coarsepath.xml",)},
    ],
}


def auto_grid_width(span, box_size=8.0, pad=4.0, minimum=20.0, maximum=64.0):
    """
    Size SCORINGGRIDS so the molecule plus its Transform translation box fits.

    The grid is centred on the ligand; anything outside it is not scored, so a
    peptide in a width=20 grid is partly invisible to the docking filters.
    Memory grows with width^3, hence the cap.
    """
    w = float(span) + 2.0 * float(box_size) + float(pad)
    w = max(minimum, math.ceil(w / 2.0) * 2.0)
    if w > maximum:
        _say(_WARN, f"computed grid width {w:.0f} A exceeds the {maximum:.0f} A cap; "
                    f"using {maximum:.0f}. Grid memory scales with width^3 — if the "
                    f"molecule really is this big, raise `maximum` deliberately.")
        w = maximum
    print(f"SCORINGGRIDS width = {w:.0f} A  (molecule span {float(span):.1f} A "
          f"+ 2x{float(box_size):.0f} A Transform box + {float(pad):.0f} A pad)")
    return w


def write_protocol_xmls(out_dir, src_dir, chain="X", tuning=None, grid_width=None,
                        files=None, src_chain="X", verbose=True):
    """
    Copy the upstream RosettaScripts protocols into `out_dir`, retargeted.

    Rewrites every chain / ligand_chain / chains attribute from `src_chain` to
    `chain`, optionally sets the SCORINGGRIDS width, and applies `tuning`
    rules. Also copies VRT1.params (the virtual residue type used for every
    constraint anchor). Returns {basename: path}.
    """
    import xml.etree.ElementTree as ET

    src_dir, out_dir = _abs(src_dir), _abs(out_dir)
    _require(os.path.isdir(src_dir), f"no such directory: {src_dir}")
    os.makedirs(out_dir, exist_ok=True)
    tuning = tuning or {"grid_width": None, "rules": []}
    if grid_width is None:
        grid_width = tuning.get("grid_width")
    rules = tuning.get("rules", [])
    wanted = tuple(files) if files else (_XML_FILES_COARSE + _XML_FILES_TRAJ)
    risky = [f for f in wanted if f in _TUNING_SENSITIVE and rules]
    if risky:
        _say(_WARN, f"{', '.join(risky)} differ from main_standard.xml only in the "
                    f"attributes these tuning rules overwrite; they will come out "
                    f"nearly identical to it.")

    written, changes = {}, []
    for base in wanted:
        src = os.path.join(src_dir, base)
        if not os.path.isfile(src):
            if base in _XML_FILES_OPTIONAL:
                continue
            raise FileNotFoundError(f"protocol not found upstream: {src}")
        raw = open(src, "r", errors="replace").read()
        had_placeholder = "$replace$" in raw
        tree = ET.ElementTree(ET.fromstring(raw))
        root = tree.getroot()

        for el in root.iter():
            for attr in ("chain", "ligand_chain", "chains"):
                if el.get(attr) == src_chain and chain != src_chain:
                    el.set(attr, chain)
                    changes.append(f"{base}: {el.tag}[{attr}] {src_chain} -> {chain}")
            if el.tag == "SCORINGGRIDS" and grid_width:
                old = el.get("width")
                el.set("width", str(int(float(grid_width))))
                changes.append(f"{base}: SCORINGGRIDS[width] {old} -> {el.get('width')}")
            for rule in rules:
                if el.tag != rule["tag"]:
                    continue
                if rule.get("name") is not None and el.get("name") != rule["name"]:
                    continue
                if rule.get("files") and base not in rule["files"]:
                    continue
                for k, v in rule["attrs"].items():
                    old = el.get(k)
                    if old == str(v):
                        continue
                    el.set(k, str(v))
                    changes.append(f"{base}: {el.tag}"
                                   f"[{rule.get('name') or '-'}].{k} {old} -> {v}")

        dest = os.path.join(out_dir, base)
        tree.write(dest, encoding="unicode", xml_declaration=False)
        text = open(dest).read()
        if not text.endswith("\n"):
            with open(dest, "a") as fh:
                fh.write("\n")
        # the placeholder is where LPF injects <Coordinates/> tags; losing it
        # would make make_xml_strings() raise at the first window
        if had_placeholder and "$replace$" not in open(dest).read():
            raise RuntimeError(f"{base}: the $replace$ placeholder was lost on rewrite")
        if base in _NEEDS_PLACEHOLDER and "$replace$" not in open(dest).read():
            raise RuntimeError(f"{base} must contain $replace$ but does not")
        ET.parse(dest)      # fails loudly if we produced invalid XML
        written[base] = dest

    vrt = os.path.join(src_dir, "VRT1.params")
    _require(os.path.isfile(vrt), f"VRT1.params not found in {src_dir}")
    shutil.copy(vrt, os.path.join(out_dir, "VRT1.params"))
    written["VRT1.params"] = os.path.join(out_dir, "VRT1.params")

    print(f"wrote {len(written)} file(s) to {out_dir}")
    if verbose:
        for c in changes:
            print(f"    {c}")
        if not changes:
            print("    (upstream protocols copied unchanged)")
    return written


def write_options_file(path, extra_res=("lig.params", "VRT1.params"),
                       database=None, auto_setup_metals=False, extra_flags=(),
                       grid_dir="grid", restore_pre_talaris=True,
                       ignore_ligand_chi=True, mute=()):
    """
    Write the Rosetta @options file that LPF copies into every working directory.

    `database` matters whenever the rosetta_scripts binary has been lifted out
    of its source tree (a Drive-hosted binary on Colab, for instance): without
    -in:path:database Rosetta cannot find its chemical database and every
    window fails identically.

    `ignore_ligand_chi=True` keeps the upstream behaviour: ligand flexibility
    comes from lig_conformers.pdb, not from on-the-fly chi sampling. Leave it on
    for a peptide-as-ligand, where the number of torsions is large.
    """
    lines = [
        "# Rosetta options for Ligand PathFinder — written by lpf_nb.write_options_file",
        "",
        f"-in:file:extra_res_fa {' '.join(extra_res)}",
        "-in:file:load_PDB_components false",
    ]
    if database:
        db = _abs(database)
        _require(os.path.isdir(db), f"Rosetta database directory not found: {db}")
        lines.append(f"-in:path:database {db}")
    if auto_setup_metals:
        lines.append("-in:auto_setup_metals")
    lines += [
        "",
        f"-qsar:grid_dir {grid_dir}",
        "",
        "# extra rotamers for protein chi1/chi2, optimise hydrogens, allow HIS/ASN/GLN flips",
        "-packing:ex1",
        "-packing:ex2",
        "-packing:no_optH false",
        "-packing:flip_HNQ true",
        f"-packing:ignore_ligand_chi {'true' if ignore_ligand_chi else 'false'}",
        "-packing:linmem_ig 10",
        "",
        "-overwrite",
    ]
    if restore_pre_talaris:
        lines += [
            "",
            "# RosettaLigand is benchmarked against the pre-talaris scoring behaviour",
            "-mistakes:restore_pre_talaris_2013_behavior true",
        ]
    for m in mute:
        lines.append(f"-mute {m}")
    for f in extra_flags:
        lines.append(str(f))
    path = _abs(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    _say(_OK, f"options -> {path}")
    if database:
        print(f"       database: {_abs(database)}")
    print(f"       extra_res_fa: {' '.join(extra_res)}")
    return path


# --------------------------------------------------------------------------
# run.json
#
# The two stages share one schema (see docs/config_reference.md upstream). The
# four parameter sets below are the starting points the notebooks use.
#
# The LIGAND_* sets are the upstream demo values verbatim (demo/*/run.json) —
# the authors explicitly recommend leaving the coarse parameters as they are.
#
# The PEPTIDE_* sets are NOT from the paper. They are those same parameters
# scaled for a molecule an order of magnitude larger, with the reasoning given
# inline. Treat them as a sane place to start, then follow the retuning
# checklist in demo/tutorial.md section 5.
# --------------------------------------------------------------------------

#: Upstream demo/coarse_path/run.json. The coarse stage ignores ipdc_d0,
#: ipdc_sd and end_padding (it recomputes them per node), so they are left out
#: on purpose — including them only makes the script print warnings.
LIGAND_COARSE_PARAMS = {
    "step": 2.0,
    "conformer_subsampling": False,
    "rmsd_constraint": True,
    "rmsd_constraint_type": "harmonic",
    "rmsd_constraint_sd": 1.5,
    "rmsd_constraint_hydrogens": False,
    "in_plane_dev_constraint": True,
    "ipdc_type": "bounded",
    "plane_constraint_sd": 0.02,
    "plane_constraint_point_extent": 7,
    "angle_constraint": True,
    "angle_constraint_type": "bounded",
    "angle_constraint_bound": 2.25,
    "angle_constraint_sd": 0.50,
    "next_angle_constraint": True,
    "next_angle_constraint_type": "bounded",
    "next_angle_constraint_bound": 1.57,
    "next_angle_constraint_sd": 0.25,
    "memory_mode": False,
    "select_best_by": "total_score",
    "endpoint_jitter": True,
    "endpoint_jitter_radius": 5.0,
    "dock_zero": False,
}

#: Upstream demo/trajectory/run.json.
LIGAND_TRAJ_PARAMS = {
    "step": 0.5,
    "smooth": 0.5,
    "conformer_subsampling": True,
    "conformer_rmsd_threshold": 1.5,
    "rmsd_constraint": True,
    "rmsd_constraint_type": "harmonic",
    "rmsd_constraint_sd": 0.75,
    "rmsd_constraint_hydrogens": True,
    "in_plane_dev_constraint": True,
    "ipdc_type": "bounded",
    "ipdc_d0": 3.0,
    "ipdc_sd": 0.5,
    "end_padding": 5,
    "plane_constraint_sd": 0.02,
    "plane_constraint_point_extent": 5,
    "angle_constraint": True,
    "angle_constraint_type": "bounded",
    "angle_constraint_bound": 2.00,
    "angle_constraint_sd": 0.50,
    "next_angle_constraint": True,
    "next_angle_constraint_type": "bounded",
    "next_angle_constraint_bound": 1.57,
    "next_angle_constraint_sd": 0.25,
    "memory_mode": True,
    "run_stats": False,
    "select_best_by": "total_score",
    "select_next_by": "total_score",
    "noise": False,
    "outlier_rmsd": 2.0,
    "skip": True,
    "dock_zero": False,
}

PEPTIDE_COARSE_PARAMS = {
    # A peptide's tracked atom is dragged by a much heavier body; 2 A nodes would
    # multiply the (already expensive) node count for no extra routing detail.
    "step": 3.0,
    # ON for peptides, unlike upstream. The full library is large and most of it
    # is irrelevant to the current pose; the threshold is the paper's >50-atom
    # value (SI, benchmark settings), which also keeps the conformational ladder
    # connected.
    "conformer_subsampling": True,
    "conformer_rmsd_threshold": 5.0,
    "rmsd_constraint": True,
    "rmsd_constraint_type": "harmonic",
    # The effective weight is sd*sqrt(N), i.e. already normalised for atom
    # count, so this really is a looser shape restraint: the peptide has to
    # change conformation to get out.
    "rmsd_constraint_sd": 2.0,
    "rmsd_constraint_hydrogens": False,
    "in_plane_dev_constraint": True,
    "ipdc_type": "bounded",
    "plane_constraint_sd": 0.02,
    # Circumradius of the virtual triangle that defines the guide plane. It must
    # be comfortably larger than the molecule or the three dihedrals that hold
    # the ligand in that plane are badly conditioned.
    "plane_constraint_point_extent": 12,
    "angle_constraint": True,
    "angle_constraint_type": "bounded",
    "angle_constraint_bound": 2.00,
    "angle_constraint_sd": 0.50,
    "next_angle_constraint": True,
    "next_angle_constraint_type": "bounded",
    "next_angle_constraint_bound": 1.57,
    "next_angle_constraint_sd": 0.25,
    # TRUE for peptides, unlike upstream. With memory_mode off, every coarse
    # node restarts from the original protein conformation, so the channel never
    # opens and a peptide-sized body may find no route at all. The cost is that
    # induced distortion accumulates along a stochastic run — all the more
    # reason to run replicates and take the medoid.
    "memory_mode": True,
    "select_best_by": "total_score",
    "endpoint_jitter": True,
    "endpoint_jitter_radius": 5.0,
    "dock_zero": False,
}

PEPTIDE_TRAJ_PARAMS = {
    # 1.0 A rather than 0.5 A: it halves the number of windows, and each peptide
    # window is already several times more expensive than a small-molecule one.
    # Drop it back towards 0.5 if the peptide conformation jumps between windows.
    "step": 1.0,
    "smooth": 1.0,
    "conformer_subsampling": True,
    # the paper's value for ligands over 50 atoms, which every peptide is
    # (SI, benchmark settings) — not a scaled guess
    "conformer_rmsd_threshold": 5.0,
    "rmsd_constraint": True,
    "rmsd_constraint_type": "harmonic",
    "rmsd_constraint_sd": 1.25,
    # Excluded: a peptide's hydrogens roughly double the atom count and
    # restraining them adds stiffness without adding shape information.
    "rmsd_constraint_hydrogens": False,
    "in_plane_dev_constraint": True,
    "ipdc_type": "bounded",
    # Wider flat well: the tracked atom of a long molecule legitimately sits
    # further off the guide point while the rest of the body threads through.
    "ipdc_d0": 4.0,
    "ipdc_sd": 0.75,
    # More room to disengage at the end — the in-plane restraint is relaxed over
    # the last end_padding/step windows.
    "end_padding": 10,
    "plane_constraint_sd": 0.02,
    "plane_constraint_point_extent": 12,
    # Looser direction restraints (2.00 rad = 115 deg lower bound, 1.40 = 80 deg):
    # an elongated body cannot keep its tracked atom as straight as a small one.
    "angle_constraint": True,
    "angle_constraint_type": "bounded",
    "angle_constraint_bound": 1.80,
    "angle_constraint_sd": 0.60,
    "next_angle_constraint": True,
    "next_angle_constraint_type": "bounded",
    "next_angle_constraint_bound": 1.40,
    "next_angle_constraint_sd": 0.35,
    "memory_mode": True,
    "run_stats": False,
    "select_best_by": "total_score",
    "select_next_by": "total_score",
    "noise": False,
    # the paper's value for ligands over 50 atoms (SI, benchmark settings);
    # 2.0 A would throw away most decoys of a large, genuinely mobile body
    "outlier_rmsd": 5.0,
    "skip": True,
    "dock_zero": False,
}

#: The settings the paper's own 136-system Geometry-of-Tunnels benchmark used
#: (SI "Geometry of Tunnels benchmark / Shared run settings"), which are also
#: what upstream examples/trajectory/run.json carries. Use these rather than the
#: demo values when you are following a CAVER/AQUA-DUCT guide path: smooth 5.0
#: suits an externally computed tunnel, where the demo's 0.5 is tuned for a
#: coarse path that LPF produced itself.
BENCHMARK_TRAJ_PARAMS = {
    "step": 0.5,
    "smooth": 5.0,
    "conformer_subsampling": True,
    "conformer_rmsd_threshold": 3.0,
    "rmsd_constraint": True,
    "rmsd_constraint_type": "harmonic",
    "rmsd_constraint_sd": 0.66,
    "rmsd_constraint_hydrogens": False,
    "in_plane_dev_constraint": True,
    "ipdc_type": "bounded",
    "ipdc_d0": 2.0,
    "ipdc_sd": 0.5,
    "end_padding": 0,
    "plane_constraint_sd": 0.02,
    "plane_constraint_point_extent": 10,
    "angle_constraint": True,
    "angle_constraint_type": "bounded",
    "angle_constraint_bound": 2.5,
    "angle_constraint_sd": 0.5,
    "next_angle_constraint": True,
    "next_angle_constraint_type": "bounded",
    "next_angle_constraint_bound": 1.57,
    "next_angle_constraint_sd": 0.25,
    "memory_mode": True,
    "run_stats": False,
    "select_best_by": "total_score",
    "select_next_by": "total_score",
    "noise": False,
    "outlier_rmsd": 2.0,
    "skip": True,
    "dock_zero": False,
}

#: The authors' own threshold for what counts as an "excessively large" ligand,
#: and the two values they loosen for it (SI, benchmark settings): "Conformer
#: subsampling threshold 3.0 A and outlier RMSD 2.0 A for normal-sized ligands,
#: 5.0 A and 5.0 A respectively for excessively large ones (more than 50 atoms)."
LARGE_LIGAND_ATOMS = 50
LARGE_LIGAND_PARAMS = {"conformer_rmsd_threshold": 5.0, "outlier_rmsd": 5.0}


def apply_large_ligand_rule(params, n_atoms, verbose=True):
    """
    Apply the paper's >50-atom rule to a parameter set, in place on a copy.

    A peptide always trips this, and so do bulky small molecules. Using the
    authors' own numbers is better than scaling the demo values by eye, which is
    what these two keys used to be.
    """
    out = dict(params)
    if int(n_atoms) <= LARGE_LIGAND_ATOMS:
        if verbose:
            print(f"ligand has {n_atoms} atoms (<= {LARGE_LIGAND_ATOMS}): "
                  f"normal-sized, keeping conformer_rmsd_threshold="
                  f"{out.get('conformer_rmsd_threshold')} and outlier_rmsd="
                  f"{out.get('outlier_rmsd')}")
        return out
    changed = []
    for k, v in LARGE_LIGAND_PARAMS.items():
        if k not in out:
            continue                       # key the chosen stage does not use
        if out[k] != v:
            changed.append(f"{k} {out[k]} -> {v}")
        out[k] = v
    if out.get("conformer_rmsd_threshold"):
        out["conformer_subsampling"] = True
    if verbose:
        print(f"ligand has {n_atoms} atoms (> {LARGE_LIGAND_ATOMS}): applying the "
              f"paper's large-ligand settings")
        for c in changed:
            print(f"       {c}")
        if not changed:
            print("       (already at those values)")
    return out


_COARSE_ONLY_PARAMS = ("endpoint_jitter", "endpoint_jitter_radius")
_TRAJ_ONLY_PARAMS = ("smooth", "noise", "skip", "run_stats", "select_next_by",
                     "outlier_rmsd", "ipdc_d0", "ipdc_sd", "end_padding")


def build_run_config(stage, rosetta, files_dir, ligand_dir, guide_path, output_path,
                     start_pdb=None, protein_pdb=None, ligand_pose_pdb=None,
                     header_pdb=None, ligand_chain="X", params=None,
                     nstruct=10, nstruct_stats=10, nstruct_zero=20, n_proc=1,
                     mpi_launcher="mpirun", extra_files=(), rosetta_database=None,
                     archive=True, verbose=True, checkpoint_interval=10,
                     use_tmp=False, tmp_path="", protocol_names=None):
    """
    Assemble a run.json dict for one stage ('coarse' or 'trajectory').

    Every path is written absolute. The upstream convention of relative '../'
    paths depends on which directory you launch from; absolute paths behave the
    same whether the stage is launched from a notebook cell, a shell, or a
    cluster job, and `run_lpf_stage` records them in the run directory.

    `params` defaults to the matching LIGAND_* set; pass PEPTIDE_* (or your own
    dict) to override. Keys the chosen stage ignores are dropped, with a note.
    """
    _require(stage in ("coarse", "trajectory"), f"stage must be coarse/trajectory, got {stage!r}")
    files_dir, ligand_dir = _abs(files_dir), _abs(ligand_dir)
    proto = {
        "coarsepath": "coarsepath.xml", "main": "main_standard.xml",
        "zero": "zero_minimize.xml", "stats": "stats.xml", "zero_dock": "zero_dock.xml",
    }
    proto.update(protocol_names or {})

    base = dict(params if params is not None else
                (LIGAND_COARSE_PARAMS if stage == "coarse" else LIGAND_TRAJ_PARAMS))
    drop = _TRAJ_ONLY_PARAMS if stage == "coarse" else _COARSE_ONLY_PARAMS
    dropped = [k for k in drop if k in base]
    for k in dropped:
        base.pop(k)
    if dropped:
        print(f"note: dropped {len(dropped)} key(s) the {stage} stage ignores: "
              f"{', '.join(dropped)}")
    if not base.get("conformer_subsampling"):
        base.pop("conformer_rmsd_threshold", None)

    cfg = {
        "rosetta": _abs(rosetta),
        "general_files": files_dir,
        "extra_files": [_abs(f) for f in extra_files],
        "input": {
            "guide_path": _abs(guide_path),
            "ligand_files": ligand_dir,
            "ligand_chain": ligand_chain,
        },
        "parameters": base,
        "rosetta_scripts": {
            "options": os.path.join(files_dir, "options"),
            "n_proc": int(n_proc),
            "mpi_launcher": mpi_launcher,
        },
        "output": {
            "path": _abs(output_path),
            "verbose": bool(verbose),
            "archive": bool(archive),
        },
    }

    if start_pdb:
        cfg["input"]["start_pdb"] = _abs(start_pdb)
    else:
        _require(protein_pdb and ligand_pose_pdb,
                 "give start_pdb, or both protein_pdb and ligand_pose_pdb")
        cfg["input"]["start_protein_pdb"] = _abs(protein_pdb)
        cfg["input"]["start_ligand_pdb"] = _abs(ligand_pose_pdb)
    if header_pdb:
        cfg["input"]["header_pdb"] = _abs(header_pdb)

    rs = cfg["rosetta_scripts"]
    if stage == "coarse":
        rs["protocol"] = os.path.join(files_dir, proto["coarsepath"])
        rs["nstruct"] = int(nstruct)
    else:
        rs["protocol_zero"] = os.path.join(files_dir, proto["zero"])
        rs["protocol_main"] = os.path.join(files_dir, proto["main"])
        rs["protocol_stats"] = os.path.join(files_dir, proto["stats"])
        rs["nstruct_main"] = int(nstruct)
        rs["nstruct_stats"] = int(nstruct_stats)
        cfg["output"]["checkpoint_interval"] = int(checkpoint_interval)
        cfg["output"]["use_tmp"] = bool(use_tmp)
        cfg["output"]["tmp_path"] = str(tmp_path)
    if base.get("dock_zero"):
        rs["protocol_zero_dock"] = os.path.join(files_dir, proto["zero_dock"])
        rs["nstruct_zero"] = int(nstruct_zero)
    if rosetta_database:
        cfg["_rosetta_database_note"] = (
            "passed to Rosetta via -in:path:database in the options file, not via "
            "run.json; recorded here only for provenance: " + _abs(rosetta_database))
    return cfg


def write_run_config(path, cfg):
    path = _abs(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    _say(_OK, f"run.json -> {path}")
    return path


#: Seconds per decoy implied by the paper's benchmark: 2.7 min per frame at
#: nstruct=10 on one core, averaged over 136 protein/tunnel/ligand combinations
#: (SI "Run times"). The spread there is wide — 0.9 to 8.2 min per frame, i.e.
#: roughly 5 to 50 s per decoy — and the upstream BChE demo sits near the fast
#: end at about 4 s. So treat this as an order of magnitude, not a prediction.
SEC_PER_DECOY_BENCHMARK = 16.0


def estimate_cost(n_units, nstruct, sec_per_decoy=SEC_PER_DECOY_BENCHMARK, n_proc=1,
                  label="window", show_range=True):
    """
    Rough wall-clock estimate, anchored on the paper's own benchmark timings.

    The default comes from the SI's 136-system benchmark: 2.7 min per frame at
    nstruct=10 on a single core. Its range is wide (0.9-8.2 min per frame), so a
    span is printed alongside the point estimate. Recalibrate from your own
    first window and pass the result in; cost scales with protein size, with how
    much of the protein the LigandArea cutoffs let move, and with the number of
    conformers that survive subsampling.
    """
    per_proc = max(1, int(n_proc))
    total = n_units * nstruct * float(sec_per_decoy) / per_proc
    h = total / 3600.0
    print(f"~{n_units} {label}s x {nstruct} decoys x {sec_per_decoy:.0f} s "
          f"/ {per_proc} proc  ->  {h:.1f} h ({total/60:.0f} min)")
    if show_range and abs(sec_per_decoy - SEC_PER_DECOY_BENCHMARK) < 1e-9:
        lo = n_units * nstruct * 5.4 / per_proc / 3600.0
        hi = n_units * nstruct * 49.2 / per_proc / 3600.0
        print(f"       the benchmark's per-frame spread puts this anywhere from "
              f"{lo:.1f} h to {hi:.1f} h — time your first window to narrow it")
    if h > 11:
        _say(_WARN, "longer than a Colab session: enable output.checkpoint_interval, "
                    "put output.path on Drive, and re-run the cell to resume.")
    return total


def preflight_config(config_path, stage, lpf_repo=None, verbose=True):
    """
    Check a run.json the way the stage will, but report every problem at once.

    LPF raises on the first failure it meets, which on a fresh set-up means
    discovering the next missing file one run at a time. Returns (errors, warnings).
    """
    cfg = json.load(open(_abs(config_path)))
    errors, warns = [], []

    def err(m):
        errors.append(m)

    def warn(m):
        warns.append(m)

    req_top = ("rosetta", "general_files", "extra_files", "input", "parameters",
               "rosetta_scripts", "output")
    for k in req_top:
        if k not in cfg:
            err(f"missing top-level key: {k}")
    if errors:
        return _report_preflight(errors, warns, verbose)

    inp, par, rs, out = cfg["input"], cfg["parameters"], cfg["rosetta_scripts"], cfg["output"]

    if stage == "coarse":
        req_par = ("step", "conformer_subsampling", "memory_mode", "rmsd_constraint_sd",
                   "select_best_by", "in_plane_dev_constraint", "ipdc_type",
                   "plane_constraint_point_extent", "endpoint_jitter",
                   "endpoint_jitter_radius")
        req_rs = ("options", "protocol", "nstruct", "n_proc", "mpi_launcher")
    else:
        req_par = ("step", "smooth", "conformer_subsampling", "noise", "skip",
                   "memory_mode", "run_stats", "select_best_by", "select_next_by",
                   "outlier_rmsd", "plane_constraint_point_extent")
        req_rs = ("options", "protocol_zero", "protocol_main", "protocol_stats",
                  "nstruct_main", "nstruct_stats", "n_proc", "mpi_launcher")
    for k in ("guide_path", "ligand_files", "ligand_chain"):
        if k not in inp:
            err(f"input.{k} is required")
    for k in req_par:
        if k not in par:
            err(f"parameters.{k} is required by the {stage} stage")
    for k in req_rs:
        if k not in rs:
            err(f"rosetta_scripts.{k} is required by the {stage} stage")
    # util.ConstraintBuilder needs these whatever the stage
    for k in ("plane_constraint_sd", "next_angle_constraint", "angle_constraint",
              "rmsd_constraint", "in_plane_dev_constraint"):
        if k not in par:
            err(f"parameters.{k} is required (util.ConstraintBuilder)")
    if par.get("rmsd_constraint"):
        for k in ("rmsd_constraint_type", "rmsd_constraint_sd", "rmsd_constraint_hydrogens"):
            if k not in par:
                err(f"parameters.{k} is required when rmsd_constraint is true")
        if str(par.get("rmsd_constraint_type", "")).lower() == "bounded" \
                and "rmsd_constraint_bound" not in par:
            err("parameters.rmsd_constraint_bound is required for a bounded rmsd_constraint")
    if stage == "trajectory" and par.get("in_plane_dev_constraint"):
        for k in ("ipdc_d0", "ipdc_sd", "end_padding"):
            if k not in par:
                err(f"parameters.{k} is required by the trajectory stage")
    if stage == "coarse":
        for k in ("ipdc_d0", "ipdc_sd", "end_padding"):
            if k in par:
                warn(f"parameters.{k} has no effect in the coarse stage "
                     "(recomputed per node); LPF will print its own warning")
    for k in _COARSE_ONLY_PARAMS if stage == "trajectory" else ():
        if k in par:
            warn(f"parameters.{k} is a coarse-only key and is ignored here")

    # binary / MPI
    rb = _abs(cfg.get("rosetta", ""))
    if not os.path.isfile(rb):
        err(f"rosetta binary not found: {rb}")
    elif not os.access(rb, os.X_OK):
        err(f"rosetta binary is not executable: {rb}")
    n_proc = rs.get("n_proc")
    nstruct_key = "nstruct" if stage == "coarse" else "nstruct_main"
    if not isinstance(n_proc, int) or n_proc < 1:
        err(f"rosetta_scripts.n_proc must be a positive integer, got {n_proc!r}")
    else:
        if n_proc > 1 and ".mpi." not in os.path.basename(rb):
            err(f"n_proc={n_proc} needs an '.mpi.' Rosetta build; this is {os.path.basename(rb)}")
        if isinstance(rs.get(nstruct_key), int) and n_proc > rs[nstruct_key]:
            err(f"n_proc ({n_proc}) exceeds {nstruct_key} ({rs[nstruct_key]}); "
                "the extra ranks would sit idle and LPF refuses to start")

    # files
    needed = [("general_files/VRT1.params",
               os.path.join(cfg["general_files"], "VRT1.params")),
              ("ligand_files/lig.params",
               os.path.join(inp.get("ligand_files", ""), "lig.params")),
              ("ligand_files/lig_conformers.pdb",
               os.path.join(inp.get("ligand_files", ""), "lig_conformers.pdb")),
              ("input.guide_path", inp.get("guide_path", "")),
              ("rosetta_scripts.options", rs.get("options", ""))]
    for key in ("protocol", "protocol_zero", "protocol_main", "protocol_stats",
                "protocol_zero_dock"):
        if key in rs:
            needed.append((f"rosetta_scripts.{key}", rs[key]))
    for f in cfg.get("extra_files", []):
        needed.append(("extra_files entry", f))
    if inp.get("start_pdb"):
        needed.append(("input.start_pdb", inp["start_pdb"]))
    elif inp.get("start_protein_pdb"):
        needed.append(("input.start_protein_pdb", inp["start_protein_pdb"]))
        needed.append(("input.start_ligand_pdb", inp.get("start_ligand_pdb", "")))
    else:
        err("input needs start_pdb, or start_protein_pdb + start_ligand_pdb")
    if inp.get("header_pdb"):
        needed.append(("input.header_pdb", inp["header_pdb"]))
    for label, p in needed:
        if not p or not os.path.isfile(_abs(p)):
            err(f"{label} not found: {p}")

    if par.get("run_stats") and stage == "trajectory":
        if isinstance(rs.get("nstruct_stats"), int) and isinstance(n_proc, int) \
                and n_proc > rs["nstruct_stats"]:
            err(f"n_proc ({n_proc}) exceeds nstruct_stats ({rs['nstruct_stats']})")
    if par.get("dock_zero"):
        if not rs.get("protocol_zero_dock"):
            err("dock_zero is on but rosetta_scripts.protocol_zero_dock is unset")
        if not isinstance(rs.get("nstruct_zero"), int) or rs["nstruct_zero"] < 1:
            err(f"dock_zero needs a positive rosetta_scripts.nstruct_zero, "
                f"got {rs.get('nstruct_zero')!r}")

    # external tools
    if shutil.which("gzip") is None:
        err("gzip is not on PATH (LPF compresses decoys after every Rosetta call)")
    if out.get("archive") and shutil.which("zip") is None:
        err("output.archive is on but zip is not on PATH")

    # the ligand itself
    lig_params = os.path.join(inp.get("ligand_files", ""), "lig.params")
    start = inp.get("start_pdb") or inp.get("start_ligand_pdb")
    if os.path.isfile(_abs(lig_params)) and start and os.path.isfile(_abs(start)):
        try:
            names = params_atom_names(lig_params)
            lig = [a for a in read_pdb_atoms(start) if a.chain == inp["ligand_chain"]]
            if not lig:
                err(f"no atoms in chain {inp['ligand_chain']!r} of {start}")
            else:
                if any(a.record != "HETATM" for a in lig):
                    err(f"chain {inp['ligand_chain']!r} must be HETATM records only "
                        "(util.read_window ignores ATOM)")
                if len({(a.resseq, a.icode) for a in lig}) != 1:
                    err(f"chain {inp['ligand_chain']!r} must be exactly one residue")
                miss = sorted(set(names) - {a.name for a in lig})
                if miss:
                    err(f"start pose is missing params atoms: {miss[:8]}")
                nbr = params_nbr_atom(lig_params)
                if nbr not in {a.name for a in lig}:
                    err(f"NBR_ATOM {nbr!r} is absent from the start pose")
                if params_resname(lig_params) != lig[0].resname:
                    err(f"residue name mismatch: start pose {lig[0].resname!r} vs "
                        f"params {params_resname(lig_params)!r}")
        except (ValueError, FileNotFoundError) as exc:
            err(f"could not cross-check ligand against params: {exc}")

    # guide path shape
    gp = inp.get("guide_path", "")
    if gp and os.path.isfile(_abs(gp)):
        try:
            pts = (np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                             for l in open(_abs(gp)) if l[:6].strip() in ("ATOM", "HETATM")])
                   if gp.lower().endswith(".pdb") else np.atleast_2d(np.genfromtxt(_abs(gp))))
            if pts.ndim != 2 or pts.shape[1] != 3:
                err(f"guide_path is not an (N,3) list of points: shape {pts.shape}")
            elif stage == "coarse" and len(pts) < 2:
                err(f"the coarse stage needs at least 2 guide points, found {len(pts)}")
            elif stage == "trajectory":
                if len(pts) < 4:
                    err(f"the trajectory stage fits a cubic smoothing spline and needs at "
                        f"least 4 guide points, found {len(pts)} — run the coarse stage "
                        f"first, or supply a CAVER/AQUA-DUCT path")
                else:
                    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                    nw = int(length / max(par.get("step", 0.5), 1e-6))
                    print(f"guide path: {len(pts)} points, {length:.1f} A long "
                          f"-> roughly {nw} windows at step={par.get('step')} A")
                    if nw > 200:
                        warn(f"~{nw} windows is a long run; consider a larger step")
        except (ValueError, OSError) as exc:
            err(f"could not read guide_path {gp}: {exc}")

    # protocols vs chain
    protos = [rs[k] for k in ("protocol", "protocol_zero", "protocol_main",
                              "protocol_stats", "protocol_zero_dock") if k in rs]
    pat = re.compile(r'(?:ligand_)?chains?\s*=\s*"([^"]*)"')
    for p in protos:
        if not os.path.isfile(_abs(p)):
            continue
        declared = {c for m in pat.findall(open(_abs(p), errors="replace").read()) for c in m}
        if declared and inp.get("ligand_chain") not in declared:
            err(f"{os.path.basename(p)} targets chain(s) {''.join(sorted(declared))} but "
                f"input.ligand_chain is {inp.get('ligand_chain')!r} — Rosetta never reads "
                f"ligand_chain, so this would score the wrong chain")

    # options file sanity
    opt = rs.get("options", "")
    if opt and os.path.isfile(_abs(opt)):
        txt = open(_abs(opt), errors="replace").read()
        if "lig.params" not in txt:
            err(f"{os.path.basename(opt)} does not pass lig.params to -extra_res_fa")
        if "VRT1.params" not in txt:
            err(f"{os.path.basename(opt)} does not pass VRT1.params to -extra_res_fa "
                "(every constraint anchor is a VRT residue)")
        if "grid_dir" not in txt:
            warn(f"{os.path.basename(opt)} has no -qsar:grid_dir; LPF creates and deletes "
                 "a 'grid' directory per window and expects Rosetta to use it")
        for f in cfg.get("extra_files", []):
            b = os.path.basename(f)
            if b.endswith(".params") and b not in txt:
                err(f"extra_files has {b} but the options file never lists it in "
                    "-extra_res_fa, so Rosetta will not know that residue type")

    return _report_preflight(errors, warns, verbose)


def _report_preflight(errors, warns, verbose=True):
    if verbose:
        for w in warns:
            _say(_WARN, w)
        for e in errors:
            _say(_BAD, e)
        if not errors:
            _say(_OK, f"preflight passed{f' with {len(warns)} warning(s)' if warns else ''}")
        else:
            print(f"\n{len(errors)} problem(s) must be fixed before this stage will run.")
    return errors, warns


# --------------------------------------------------------------------------
# launching a stage
# --------------------------------------------------------------------------

_STAGE_SCRIPT = {"coarse": "lpf/lpf_coarse_path.py", "trajectory": "lpf/lpf_trajectory.py"}


def prepare_run_dir(run_dir, clean=False):
    """
    Make the run directory.

    `clean=True` wipes it first. The coarse stage has no resume support and
    creates one directory per node with os.mkdir, so re-running into a finished
    coarse directory fails with FileExistsError. The trajectory stage instead
    resumes from checkpoint.pkl, so leave `clean=False` to continue an
    interrupted trajectory.
    """
    run_dir = _abs(run_dir)
    if clean and os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
        print(f"removed previous contents of {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_lpf_stage(stage, lpf_repo, config_path, run_dir=None, overrides=(),
                  fresh=False, python_exe=None, log_path=None, echo=True,
                  check=True):
    """
    Run lpf_coarse_path.py or lpf_trajectory.py as a subprocess, streaming output.

    The upstream scripts are run unmodified. Output is echoed live (so a long
    run shows progress in the notebook) and tee'd to `log_path`.

    `overrides` are passed straight through as --set KEY=VALUE, e.g.
    ("parameters.step=0.75", "rosetta_scripts.nstruct_main=20").
    `fresh=True` (trajectory only) ignores an existing checkpoint.
    """
    _require(stage in _STAGE_SCRIPT, f"stage must be coarse/trajectory, got {stage!r}")
    script = os.path.join(_abs(lpf_repo), _STAGE_SCRIPT[stage])
    _require(os.path.isfile(script), f"LPF script not found: {script}")
    config_path = _abs(config_path)
    _require(os.path.isfile(config_path), f"run.json not found: {config_path}")
    run_dir = _abs(run_dir or os.path.dirname(config_path))
    os.makedirs(run_dir, exist_ok=True)

    argv = [python_exe or sys.executable, script, config_path]
    for o in overrides:
        argv += ["--set", str(o)]
    if fresh:
        _require(stage == "trajectory", "--fresh only exists for the trajectory stage")
        argv.append("--fresh")

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    print("$ cd " + run_dir)
    print("$ " + " ".join(argv) + "\n")

    log_fh = open(_abs(log_path), "w") if log_path else None
    tail = []
    try:
        proc = subprocess.Popen(argv, cwd=run_dir, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            if echo:
                print(line, end="")
            if log_fh:
                log_fh.write(line)
            tail.append(line)
            if len(tail) > 60:
                tail.pop(0)
        rc = proc.wait()
    finally:
        if log_fh:
            log_fh.close()

    if rc != 0:
        msg = (f"{os.path.basename(script)} exited with status {rc}. Last lines:\n"
               + textwrap.indent("".join(tail), "    "))
        if check:
            raise RuntimeError(msg)
        print(msg)
    else:
        _say(_OK, f"{stage} stage finished (exit 0)")
        if log_path:
            print(f"       log: {_abs(log_path)}")
    return rc


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def _loadtxt(path, cols=None):
    p = _abs(path)
    if not os.path.isfile(p):
        return None
    a = np.loadtxt(p, ndmin=2 if cols else 1)
    if cols and a.ndim == 2 and a.shape[1] != cols:
        a = a.reshape(-1, cols)
    return a


def load_run_scores(run_dir):
    """
    Read what a finished run wrote.

    interface_scores.txt (Rosetta's interface_delta_X per scored window, the
    file to analyse), total_scores.txt (whole-complex energy, mostly debug),
    nbr_trajectory.txt (the tracked atom's position per window), and the path
    construction intermediates.
    """
    rd = _abs(run_dir)
    data = {
        "interface_scores": _loadtxt(os.path.join(rd, "interface_scores.txt")),
        "total_scores": _loadtxt(os.path.join(rd, "total_scores.txt")),
        "nbr_trajectory": _loadtxt(os.path.join(rd, "nbr_trajectory.txt"), cols=3),
        "guide_path": _loadtxt(os.path.join(rd, "guide_path"), cols=3),
        "input_path": _loadtxt(os.path.join(rd, "input_path"), cols=3),
        "coarsepath": _loadtxt(os.path.join(rd, "coarsepath.txt"), cols=3),
        "run_dir": rd,
    }
    for k in ("interface_scores", "total_scores"):
        if data[k] is not None:
            data[k] = np.atleast_1d(data[k]).ravel()
    return data


def cumulative_path_length(points):
    """Distance travelled by the tracked atom, cumulative, starting at 0."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or len(p) == 0:
        return np.array([])
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def load_stats(run_dir):
    """
    Per-window score distributions from the optional statistics pass.

    Populated only when parameters.run_stats was true: nstruct_stats independent
    rescores of each selected pose, i.e. an error bar on the energy profile.
    """
    p = os.path.join(_abs(run_dir), "stats.pkl")
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as fh:
        st = pickle.load(fh)
    inter = st.get("interface_scores") or []
    if not len(inter):
        return None
    return {
        "interface_mean": np.array([np.mean(a) for a in inter]),
        "interface_std": np.array([np.std(a) for a in inter]),
        "interface_raw": inter,
        "total_raw": st.get("total_scores") or [],
    }


def plot_energy_profile(run_dir, x="distance", show_stats=True, title=None,
                        ax=None, color="#1f77b4", label=None, smooth_window=0):
    """
    Plot interface_delta_X along the trajectory — the main result of a run.

    x='distance' puts the cumulative distance travelled by the tracked atom on
    the x-axis (what the paper's figure shows); x='window' uses the window index.
    A single replicate is noisy: pool several before reading much into the shape.
    """
    import matplotlib.pyplot as plt
    d = load_run_scores(run_dir)
    y = d["interface_scores"]
    _require(y is not None and len(y), f"no interface_scores.txt in {_abs(run_dir)}")

    if x == "distance" and d["nbr_trajectory"] is not None \
            and len(d["nbr_trajectory"]) == len(y):
        xs = cumulative_path_length(d["nbr_trajectory"])
        xlabel = "distance travelled by the tracked atom (A)"
    else:
        xs = np.arange(len(y))
        xlabel = "scored window"

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0.0, lw=0.8, color="0.6", zorder=1)
    ax.plot(xs, y, "-o", ms=3.2, lw=1.3, color=color, zorder=3,
            label=label or "interface_delta_X")

    st = load_stats(run_dir) if show_stats else None
    if st is not None and len(st["interface_mean"]) == len(y):
        ax.fill_between(xs, st["interface_mean"] - st["interface_std"],
                        st["interface_mean"] + st["interface_std"],
                        color=color, alpha=0.18, lw=0, zorder=2,
                        label="statistics pass, +/-1 sd")
    if smooth_window and smooth_window > 1 and len(y) > smooth_window:
        k = int(smooth_window)
        sm = np.convolve(y, np.ones(k) / k, mode="valid")
        ax.plot(xs[k // 2: k // 2 + len(sm)], sm, "-", lw=2.0, color="#d62728",
                zorder=4, label=f"running mean ({k})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("interface energy (REU)")
    ax.set_title(title or f"Binding-energy profile — {os.path.basename(_abs(run_dir))}")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    ax.figure.tight_layout()
    return ax


def summarize_run(run_dir, stage="trajectory"):
    """Print what is in a run directory and whether it finished."""
    rd = _abs(run_dir)
    print(f"run directory: {rd}")
    if not os.path.isdir(rd):
        _say(_BAD, "directory does not exist")
        return {}
    d = load_run_scores(rd)
    info = {}

    meta = os.path.join(rd, "checkpoint_meta.json")
    if os.path.isfile(meta):
        m = json.load(open(meta))
        info["completed"] = bool(m.get("completed"))
        _say(_OK 
             if m.get("completed") else _WARN,
             f"checkpoint says the run is {'complete' if m.get('completed') else 'INCOMPLETE'} "
             f"— re-run the same cell to resume, or pass fresh=True to start over")

    if stage == "coarse":
        cp = d["coarsepath"]
        if cp is not None:
            length = float(np.linalg.norm(np.diff(cp, axis=0), axis=1).sum())
            seg = np.linalg.norm(np.diff(cp, axis=0), axis=1)
            print(f"coarsepath.txt: {len(cp)} points, {length:.1f} A long, "
                  f"spacing {seg.min():.2f}-{seg.max():.2f} A")
            info["n_points"], info["length"] = len(cp), length
        else:
            _say(_BAD, "no coarsepath.txt — the coarse stage did not finish")
        nodes = sorted(glob.glob(os.path.join(rd, "[0-9]*")))
        print(f"node directories/archives: {len(nodes)}")
    else:
        y = d["interface_scores"]
        if y is not None and len(y) and np.allclose(y, 0.0):
            _say(_BAD, "every interface score is exactly 0.0 — this is NOT a result.")
            print("       Rosetta's InterfaceScoreCalculator contributed nothing, so "
                  "the column was")
            print("       written as a placeholder. Check a window's score.sc for the "
                  "if_X_* family:")
            print("         head -2 <window>/score.sc | tail -1 | tr ' ' '\\n' | "
                  "grep if_X")
            print("       If it is absent, the protocol's InterfaceScoreCalculator did "
                  "not run or")
            print("       targeted the wrong chain. With the PyRosetta backend, probe "
                  "it directly:")
            print("         python pyrosetta_rosetta_scripts.py --probe "
                  "<window>/task.xml <window>/task.pdb")
            print()
        if y is not None and len(y):
            print(f"interface_scores.txt: {len(y)} scored windows, "
                  f"min {y.min():+.2f} REU (window {int(np.argmin(y))}), "
                  f"final {y[-1]:+.2f} REU, release {y[-1] - y.min():+.2f} REU")
            info["n_windows"], info["min"], info["final"] = len(y), float(y.min()), float(y[-1])
        else:
            _say(_BAD, "no interface_scores.txt")
        gp = d["guide_path"]
        if gp is not None and y is not None:
            print(f"guide path had {len(gp)} windows; {len(gp) - len(y)} were skipped "
                  f"(parameters.skip) or not reached")
        nt = d["nbr_trajectory"]
        if nt is not None and len(nt) > 1:
            print(f"tracked atom travelled {cumulative_path_length(nt)[-1]:.1f} A")
        if load_stats(rd) is not None:
            print("statistics pass present (stats.pkl)")
        wins = sorted(glob.glob(os.path.join(rd, "window_*")))
        print(f"window directories/archives: {len(wins)}")

    for f in ("lig_trajectory.pdb", "full_trajectory.pdb", "trajectory.pdb"):
        p = os.path.join(rd, f)
        if os.path.isfile(p):
            n = sum(1 for l in open(p) if l.startswith("ENDMDL"))
            print(f"{f}: {n} frame(s), {os.path.getsize(p) / 1e6:.1f} MB")
    return info


def relocate_project(project_dir, remap=(), old_root=None, dry_run=False,
                     verbose=True):
    """
    Repair a project directory that has been moved.

    Every path this tooling writes is absolute — deliberately, so a run behaves
    the same however it is launched — which means moving the directory
    afterwards leaves stale paths in project.json, each run's run.json, the
    checkpoints, and the PyRosetta launcher. This rewrites them.

    The old location is read from project.json's own `project_dir` field, so you
    do not have to remember it. `remap` takes extra ("old", "new") pairs for
    files that lived outside the project, such as the structure you passed to
    --target.

    Returns {"changed": [paths], "stale": [paths still referencing the old root]}.
    """
    project_dir = _abs(project_dir)
    man_path = os.path.join(project_dir, "project.json")
    _require(os.path.isfile(man_path), f"no project.json in {project_dir}")
    man = json.load(open(man_path))
    old = _abs(old_root or man.get("project_dir") or project_dir)

    pairs = [(old, project_dir)] + [(_abs(a), _abs(b)) for a, b in remap]
    pairs = [(a, b) for a, b in pairs if a != b]
    if not pairs:
        if verbose:
            _say(_OK, f"nothing to do — project.json already says {project_dir}")
        return {"changed": [], "stale": []}
    if verbose:
        print(f"rewriting {len(pairs)} path mapping(s):")
        for a, b in pairs:
            print(f"   {a}\n     -> {b}")
        if dry_run:
            print("\n--dry-run: nothing will be written\n")

    def swap(text):
        for a, b in pairs:
            text = text.replace(a, b)
        return text

    changed = []
    texts = [man_path]
    for pat in ("run.json", "checkpoint_meta.json", "rosetta_scripts.*"):
        texts += glob.glob(os.path.join(project_dir, "*", pat))
        texts += glob.glob(os.path.join(project_dir, "*", "*", pat))
    for f in sorted(set(texts)):
        if not os.path.isfile(f) or f.endswith(".py"):
            continue
        try:
            before = open(f, errors="replace").read()
        except OSError:
            continue
        after = swap(before)
        if after != before:
            if not dry_run:
                with open(f, "w") as fh:
                    fh.write(after)
            changed.append(f)

    for f in sorted(glob.glob(os.path.join(project_dir, "*", "checkpoint.pkl"))):
        try:
            with open(f, "rb") as fh:
                state = pickle.load(fh)
        except Exception as exc:
            _say(_WARN, f"could not read {f}: {exc}")
            continue
        touched = False
        for key in ("rootdir", "prev_path"):
            v = state.get(key)
            if isinstance(v, str) and swap(v) != v:
                state[key] = swap(v)
                touched = True
        if touched:
            if not dry_run:
                with open(f, "wb") as fh:
                    pickle.dump(state, fh)
            changed.append(f)

    stale = []
    for root, _dirs, files in os.walk(project_dir):
        for name in files:
            if not name.endswith((".json", ".pkl", ".pyrosetta", ".txt")):
                continue
            f = os.path.join(root, name)
            try:
                blob = open(f, "rb").read()
            except OSError:
                continue
            if any(a.encode() in blob for a, _ in pairs):
                stale.append(f)

    if verbose:
        print(f"\n{len(changed)} file(s) {'would be ' if dry_run else ''}rewritten")
        if stale and not dry_run:
            _say(_WARN, f"{len(stale)} file(s) still mention the old path:")
            for f in stale[:8]:
                print(f"       {os.path.relpath(f, project_dir)}")
        elif not dry_run:
            logs = []
            for root, _d, files in os.walk(project_dir):
                logs += [os.path.join(root, n) for n in files if n.endswith(".log")]
            n_logs = sum(1 for f in logs
                         if any(a.encode() in open(f, "rb").read() for a, _ in pairs))
            _say(_OK, "no file that is read back mentions the old path any more")
            if n_logs:
                print(f"       ({n_logs} .log file(s) still do, deliberately — they "
                       f"are a record of where the run actually happened)")
    return {"changed": changed, "stale": stale}


def rescore_windows(run_dir, rosetta, files_dir, ligand_dir, protocol=None,
                    ligand_chain="X", extra_files=(), n_proc=1,
                    mpi_launcher="mpirun", out_name="interface_scores_rescored.txt",
                    keep_work=False, install=False, verbose=True):
    """
    Recompute the interface energy of a finished run, without redoing the docking.

    For salvaging a trajectory whose interface_scores.txt came out all zeros: each
    window's selected pose (best.pdb) is rescored once with a scoring-only
    protocol, so the cost is one Rosetta call per window instead of nstruct
    docking runs — minutes rather than hours.

    `protocol` defaults to zero_score.xml from the files directory (upstream
    ships it: a single InterfaceScoreCalculator and nothing else). Archived
    windows (`output.archive`) are read straight out of their zips.

    Writes `out_name` next to the original scores and returns the array. The
    original interface_scores.txt is left alone unless `install` is set, in
    which case it is backed up to interface_scores_original.txt and replaced, so
    that plot_energy_profile, summarize_run and cluster_trajectories — all of
    which read interface_scores.txt — pick the new values up.
    """
    run_dir, files_dir, ligand_dir = _abs(run_dir), _abs(files_dir), _abs(ligand_dir)
    protocol = _abs(protocol or os.path.join(files_dir, "zero_score.xml"))
    _require(os.path.isfile(protocol),
             f"no scoring protocol at {protocol}. Copy zero_score.xml from the "
             f"repository's rosettaXML_files/, or pass protocol=...")

    plain = sorted(glob.glob(os.path.join(run_dir, "window_*")))
    plain = [d for d in plain if os.path.isdir(d)]
    zips = sorted(glob.glob(os.path.join(run_dir, "window_*.zip")))
    units = [(d, None) for d in plain] + [(z, z) for z in zips]
    units.sort(key=lambda t: os.path.basename(t[0]).split(".")[0])
    _require(units, f"no window_* directories or archives in {run_dir}")
    if verbose:
        print(f"{len(units)} window(s) to rescore "
              f"({len(plain)} unpacked, {len(zips)} archived)")
        print(f"protocol: {os.path.basename(protocol)} (scoring only, 1 decoy each)")

    work_root = os.path.join(run_dir, "_rescore")
    if os.path.isdir(work_root):
        shutil.rmtree(work_root)
    os.makedirs(work_root)

    scores, missing = [], []
    for idx, (unit, zpath) in enumerate(units):
        name = os.path.basename(unit).split(".")[0]
        wd = os.path.join(work_root, name)
        os.makedirs(os.path.join(wd, "results"))
        os.makedirs(os.path.join(wd, "grid"))

        if zpath:
            with zipfile.ZipFile(zpath) as z:
                hit = next((m for m in z.namelist()
                            if m.endswith("/best.pdb") or m == "best.pdb"), None)
                if hit is None:
                    missing.append(name)
                    continue
                with z.open(hit) as src, open(os.path.join(wd, "input.pdb"), "wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            best = os.path.join(unit, "best.pdb")
            if not os.path.isfile(best):
                missing.append(name)
                continue
            shutil.copy(best, os.path.join(wd, "input.pdb"))

        for src, dst in [(os.path.join(files_dir, "options"), "options"),
                         (protocol, "task.xml"),
                         (os.path.join(ligand_dir, "lig.params"), None),
                         (os.path.join(ligand_dir, "lig_conformers.pdb"), None),
                         (os.path.join(files_dir, "VRT1.params"), None)]:
            if not os.path.isfile(src):
                raise FileNotFoundError(f"rescore_windows: missing {src}")
            shutil.copy(src, os.path.join(wd, dst) if dst else wd)
        for f in extra_files:
            shutil.copy(_abs(f), wd)

        argv = []
        if int(n_proc) > 1:
            argv += [mpi_launcher, "-np", "1"]      # one decoy; MPI is pointless
        argv += [_abs(rosetta), "@options", "-nstruct", "1",
                 "-out:path:pdb", "./results", "-parser:protocol", "task.xml",
                 "-in:file:s", "input.pdb"]
        _run_rosetta_raw(argv, cwd=wd,
                         log_path=os.path.join(wd, "task.log"),
                         err_path=os.path.join(wd, "task.err"))

        rows = read_score_file(os.path.join(wd, "score.sc"))
        r = rows[0]
        if_cols = [c for c in r if c.startswith("if_") and c.endswith("_constraint")]
        clean = float(r.get("interface_delta_" + ligand_chain, float("nan"))) \
            - sum(float(r.get(c, 0.0)) for c in if_cols)
        scores.append(clean)
        if verbose:
            n_if = sum(1 for c in r if c.startswith("if_"))
            print(f"  {name}: interface_delta_{ligand_chain} = {clean:+.3f} REU "
                  f"({n_if} if_{ligand_chain}_* terms)"
                  + ("   <-- still no interface terms!" if n_if == 0 else ""))

    if missing and verbose:
        _say(_WARN, f"{len(missing)} window(s) had no best.pdb and were skipped: "
                    f"{missing[:6]}")
    _require(scores, "nothing could be rescored")

    arr = np.array(scores)
    out = os.path.join(run_dir, out_name)
    np.savetxt(out, arr)

    if install:
        live = os.path.join(run_dir, "interface_scores.txt")
        backup = os.path.join(run_dir, "interface_scores_original.txt")
        if os.path.isfile(live) and not os.path.isfile(backup):
            shutil.copy(live, backup)
            if verbose:
                print(f"original backed up to {os.path.basename(backup)}")
        n_live = len(np.atleast_1d(np.loadtxt(live)).ravel()) \
            if os.path.isfile(live) else len(arr)
        if n_live != len(arr):
            _say(_WARN, f"the original has {n_live} values and the rescore has "
                        f"{len(arr)} — NOT installing, since the window ordering "
                        f"may differ. Compare {os.path.basename(out)} by hand.")
        else:
            np.savetxt(live, arr)
            if verbose:
                _say(_OK, "installed as interface_scores.txt — analysis and "
                          "clustering now use the rescored values")
    if not keep_work:
        shutil.rmtree(work_root, ignore_errors=True)
    if verbose:
        print()
        if np.allclose(arr, 0.0):
            _say(_BAD, "the rescore is also all zeros — InterfaceScoreCalculator is "
                       "still contributing nothing. Probe it directly before "
                       "spending more time (see the shim's --probe).")
        else:
            print(f"rescored {len(arr)} window(s): bound {arr[0]:+.2f} REU, "
                  f"final {arr[-1]:+.2f} REU, min {arr.min():+.2f} REU")
        print(f"-> {out}")
        print("   The original interface_scores.txt is untouched; compare them.")
    return arr


def read_ligand_frames(lig_trajectory_pdb, ligand_chain=None):
    """
    Read lig_trajectory.pdb into a list of (n_atoms, 3) arrays, one per frame.

    LPF separates frames with a bare ENDMDL. Atom order is constant within a run
    because every frame is written from the same residue type.
    """
    frames, cur = [], []
    with open(_abs(lig_trajectory_pdb), errors="replace") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                if cur:
                    frames.append(np.array(cur, dtype=float))
                cur = []
            elif line[:6].strip() in ("ATOM", "HETATM") and len(line) >= 54:
                if ligand_chain and line[21] != ligand_chain:
                    continue
                cur.append([float(line[30:38]), float(line[38:46]),
                            float(line[46:54])])
    if cur:
        frames.append(np.array(cur, dtype=float))
    _require(frames, f"no frames parsed from {lig_trajectory_pdb}")
    sizes = {len(f) for f in frames}
    _require(len(sizes) == 1,
             f"{lig_trajectory_pdb} has frames with different atom counts: {sizes}")
    return frames


def path_distance_matrix(frame_sets, missing_penalty=0.5):
    """
    Pairwise distances between trajectories, as the paper defines them.

    From the SI ("Clustering of sampled paths"): the frame-by-frame coordinate
    RMSD of the ligand, summed over frames with linearly decreasing weights, so
    that differences early in the path — while the ligand is still inside the
    tunnel — count for more than differences in bulk solvent. Where one path is
    shorter, each missing frame contributes a fixed penalty of `missing_penalty`
    angstroms.

    No superposition is done: these are coordinate RMSDs in the shared frame of
    the protein, which is what makes two paths through different tunnels far
    apart. Weights run linearly from 1 at the first frame to 0 at the last and
    are normalised to sum to 1, so a distance reads in angstroms. The SI states
    the weighting is linear but not its normalisation, so that part is this
    implementation's choice.
    """
    n = len(frame_sets)
    _require(n >= 2, "need at least two trajectories")
    atoms = {fs[0].shape[0] for fs in frame_sets}
    _require(len(atoms) == 1,
             f"the trajectories have different ligand atom counts {atoms}; they "
             f"must be the same ligand")
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = frame_sets[i], frame_sets[j]
            n_common, n_max = min(len(a), len(b)), max(len(a), len(b))
            w = np.linspace(1.0, 0.0, n_max, endpoint=False)
            w = w / w.sum()
            d = 0.0
            for f in range(n_common):
                diff = a[f] - b[f]
                d += w[f] * float(np.sqrt((diff ** 2).sum(axis=1).mean()))
            for f in range(n_common, n_max):
                d += w[f] * float(missing_penalty)
            D[i, j] = D[j, i] = d
    return D


def replicate_convergence(run_dirs, n_boot=300, step=None, plot=True, seed=0,
                          verbose=True):
    """
    Measure how many replicates you actually need, instead of guessing.

    For each ensemble size n, `n_boot` bootstrap resamples of n runs are drawn
    (with replacement) and the mean interface profile computed for each. The spread of those means — averaged
    over frames — is how much your answer would move if you had happened to run
    a different n replicates. When that spread falls below the size of the
    feature you care about (a 1 REU barrier, say), adding more buys you little.

    Decreases roughly as 1/sqrt(n), which is drawn as a reference, so a curve
    that sits above it means the replicates disagree structurally — different
    exit routes — and the right response is clustering, not more averaging.

    Returns {"n": [...], "spread": [...], "sem_full": array, "profiles": (N,k)}.
    """
    profiles = []
    for rd in run_dirs:
        f = os.path.join(_abs(rd), "interface_scores.txt")
        if not os.path.isfile(f):
            continue
        y = np.atleast_1d(np.loadtxt(f)).ravel()
        if len(y):
            profiles.append(y)
    _require(len(profiles) >= 3,
             f"need at least 3 finished runs to say anything about convergence, "
             f"found {len(profiles)}")

    k = min(len(y) for y in profiles)
    A = np.stack([y[:k] for y in profiles])
    N = len(A)
    if verbose:
        lens = [len(y) for y in profiles]
        print(f"{N} replicates, {min(lens)}-{max(lens)} windows each; "
              f"comparing the first {k}")
        if np.allclose(A, 0.0):
            _say(_BAD, "every profile is all zeros — fix the scoring before "
                       "asking about convergence")

    rng = np.random.default_rng(seed)
    ladder = [n for n in (2, 3, 5, 8, 12, 20, 30, 50, 75, 100) if n < N] + [N]
    spread = []
    for n in ladder:
        # bootstrap WITH replacement. Drawing without it would shrink the spread
        # to zero as n approached N (there is only one subset of size N), which
        # is a property of the sampling scheme, not of the data — it made the
        # curve turn back up at the last point.
        means = np.array([A[rng.integers(0, N, size=n)].mean(axis=0)
                          for _ in range(int(n_boot))])
        spread.append(float(means.std(axis=0).mean()))

    sem_full = A.std(axis=0, ddof=1) / np.sqrt(N)
    if verbose:
        print(f"\n  {'n':>5s}{'spread of the mean (REU)':>28s}")
        for n, sp in zip(ladder, spread):
            bar = "#" * max(1, int(round(sp / max(spread) * 28))) if max(spread) else ""
            print(f"  {n:>5d}{sp:>26.3f}  {bar}")
        print(f"\nfull ensemble (n={N}): mean SEM {sem_full.mean():.3f} REU, "
              f"worst frame {sem_full.max():.3f} REU")
        print("\nRead it as: to resolve a feature of size D, you want the spread")
        print("comfortably below D. The paper's tutorial runs 30 replicates in its")
        print("worked example; its clustering discards clusters of 10 or fewer, so")
        print("an ensemble meant for clustering has to be a good deal larger than 10.")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].plot(ladder, spread, "-o", ms=4, color="#1f77b4", label="measured")
        ref = np.array(spread[0]) * np.sqrt(ladder[0] / np.array(ladder, dtype=float))
        ax[0].plot(ladder, ref, "--", color="0.5", lw=1.2, label=r"$1/\sqrt{n}$")
        ax[0].set_xlabel("replicates pooled (n)")
        ax[0].set_ylabel("spread of the mean profile (REU)")
        ax[0].set_xscale("log")
        ax[0].set_title("how much the answer still moves")
        ax[0].grid(alpha=0.25, lw=0.6)
        ax[0].legend(frameon=False, fontsize=9)

        x = np.arange(k) * (float(step) if step else 1.0)
        m = A.mean(axis=0)
        ax[1].axhline(0.0, lw=0.8, color="0.6")
        ax[1].fill_between(x, m - sem_full, m + sem_full, alpha=0.25, lw=0,
                           color="#1f77b4")
        ax[1].plot(x, m, lw=1.6, color="#1f77b4")
        ax[1].set_xlabel("progression along the path"
                         + (" (A)" if step else " (window)"))
        ax[1].set_ylabel("interface energy (REU)")
        ax[1].set_title(f"pooled mean +/- SEM, n={N}")
        ax[1].grid(alpha=0.25, lw=0.6)
        fig.tight_layout()

    return {"n": ladder, "spread": spread, "sem_full": sem_full, "profiles": A}


def cluster_trajectories(run_dirs, ligand_chain="X", missing_penalty=0.5,
                         min_cluster_size=10, step=None, plot=True,
                         random_state=0, verbose=True):
    """
    Cluster pooled trajectory replicates and average their energy profiles.

    This is the analysis the paper reports for an ensemble of runs (SI,
    "Clustering of sampled paths"): the negative of the path distance matrix is
    handed to affinity propagation as a precomputed affinity, the interface
    profile is averaged within each cluster, clusters of `min_cluster_size` or
    fewer paths are discarded, and the retained averages are plotted against
    progression along the path, with the most populated cluster in bold.

    One replicate tells you very little; this is what turns thirty of them into
    a result. `step` is the per-frame spacing in angstroms for the x-axis and is
    read from each run's run.json when not given.

    Returns a dict with the labels, the distance matrix, the per-cluster mean
    profiles and the exemplar run of each cluster.
    """
    try:
        from sklearn.cluster import AffinityPropagation
    except ImportError:
        raise ImportError(
            "clustering needs scikit-learn: pip install scikit-learn\n"
            "(the distance matrix itself is available without it via "
            "path_distance_matrix)")

    runs, frame_sets, profiles, steps = [], [], [], []
    for rd in run_dirs:
        rd = _abs(rd)
        lig = os.path.join(rd, "lig_trajectory.pdb")
        sc = os.path.join(rd, "interface_scores.txt")
        if not (os.path.isfile(lig) and os.path.isfile(sc)):
            if verbose:
                _say(_WARN, f"skipping {rd}: no lig_trajectory.pdb / "
                            f"interface_scores.txt (did the run finish?)")
            continue
        frames = read_ligand_frames(lig, ligand_chain)
        prof = np.atleast_1d(np.loadtxt(sc)).ravel()
        if len(prof) != len(frames) and verbose:
            _say(_WARN, f"{os.path.basename(rd)}: {len(frames)} frames but "
                        f"{len(prof)} scores — using the shorter")
        k = min(len(prof), len(frames))
        runs.append(rd)
        frame_sets.append(frames[:k])
        profiles.append(prof[:k])
        cfg = os.path.join(rd, "run.json")
        if os.path.isfile(cfg):
            try:
                steps.append(float(json.load(open(cfg))["parameters"]["step"]))
            except (KeyError, ValueError):
                pass

    _require(len(runs) >= 2, f"need at least two finished runs, found {len(runs)}")
    if step is None:
        step = steps[0] if steps else 1.0
        if steps and len(set(steps)) > 1:
            _say(_WARN, f"the runs used different step sizes {sorted(set(steps))}; "
                        f"using {step} for the x-axis")
    if verbose:
        print(f"{len(runs)} finished runs, "
              f"{frame_sets[0][0].shape[0]} ligand atoms, "
              f"{min(len(f) for f in frame_sets)}-{max(len(f) for f in frame_sets)} "
              f"frames, step {step} A")

    D = path_distance_matrix(frame_sets, missing_penalty=missing_penalty)
    ap = AffinityPropagation(affinity="precomputed", random_state=random_state)
    labels = ap.fit_predict(-D)
    exemplars = getattr(ap, "cluster_centers_indices_", None)

    sizes = {int(c): int((labels == c).sum()) for c in set(labels)}
    kept = [c for c, n in sizes.items() if n > min_cluster_size]
    kept.sort(key=lambda c: -sizes[c])
    if verbose:
        iu = np.triu_indices(len(runs), 1)
        print(f"mean pairwise distance {D[iu].mean():.2f} A "
              f"(min {D[iu].min():.2f}, max {D[iu].max():.2f})")
        print(f"{len(sizes)} cluster(s): sizes "
              f"{sorted(sizes.values(), reverse=True)}")
        print(f"{len(kept)} with more than {min_cluster_size} member(s) retained"
              + (f", {len(sizes) - len(kept)} discarded" if len(kept) < len(sizes)
                 else ""))
        if not kept:
            _say(_WARN, f"every cluster has {min_cluster_size} or fewer members. "
                        f"The paper pooled ~30+ replicates; with fewer, lower "
                        f"min_cluster_size to see anything.")

    means = {}
    for c in kept:
        idx = [i for i, l in enumerate(labels) if l == c]
        n_max = max(len(profiles[i]) for i in idx)
        stack = np.full((len(idx), n_max), np.nan)
        for row, i in enumerate(idx):
            stack[row, :len(profiles[i])] = profiles[i]
        means[c] = {
            "mean": np.nanmean(stack, axis=0),
            "sem": np.nanstd(stack, axis=0) / np.sqrt(np.maximum(
                1, (~np.isnan(stack)).sum(axis=0))),
            "n": len(idx),
            "runs": [os.path.basename(runs[i]) for i in idx],
            "exemplar": (os.path.basename(runs[int(exemplars[c])])
                         if exemplars is not None and c < len(exemplars) else None),
        }

    if plot and kept:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        ax.axhline(0.0, lw=0.8, color="0.6", zorder=1)
        for rank, c in enumerate(kept):
            m = means[c]["mean"]
            x = np.arange(len(m)) * float(step)
            if rank == 0:
                ax.plot(x, m, lw=2.4, color="#1f77b4", zorder=4,
                        label=f"cluster {c} (n={means[c]['n']})")
            else:
                ax.plot(x, m, lw=1.0, alpha=0.55, zorder=3,
                        label=f"cluster {c} (n={means[c]['n']})")
        ax.set_xlabel("progression along the path (A)")
        ax.set_ylabel("interface energy (REU)")
        ax.set_title(f"cluster-averaged profiles, {len(runs)} replicates")
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()

    return {"runs": runs, "labels": labels, "distances": D, "sizes": sizes,
            "kept": kept, "means": means, "step": step}


def normalize_multimodel_pdb(in_pdb, out_pdb):
    """
    Add MODEL/ENDMDL records to an LPF trajectory PDB.

    LPF separates frames with a bare ENDMDL and no opening MODEL, which most
    viewers will not split into frames.
    """
    n = 0
    with open(_abs(in_pdb)) as fh, open(_abs(out_pdb), "w") as out:
        out.write(f"MODEL     {n + 1:4d}\n")
        n = 1
        for line in fh:
            if line.startswith("ENDMDL"):
                out.write("ENDMDL\n")
                n += 1
                out.write(f"MODEL     {n:4d}\n")
            else:
                out.write(line)
        out.write("ENDMDL\n")
    print(f"{n} frame(s) -> {_abs(out_pdb)}")
    return _abs(out_pdb)


def view_trajectory(run_dir, ligand_chain="X", mode="overlay", stride=1,
                    width=760, height=520, protein_style="cartoon"):
    """
    Show the trajectory with py3Dmol.

    mode='overlay' draws the protein once and every ligand pose on top, coloured
    from blue (bound) to red (released) — the static figure in the paper.
    mode='frames' gives an animation instead.
    """
    import py3Dmol
    rd = _abs(run_dir)
    full = os.path.join(rd, "full_trajectory.pdb")
    lig = os.path.join(rd, "lig_trajectory.pdb")
    _require(os.path.isfile(full), f"no full_trajectory.pdb in {rd}")

    view = py3Dmol.view(width=width, height=height)
    if mode == "frames":
        norm = normalize_multimodel_pdb(full, os.path.join(rd, "_frames.pdb"))
        view.addModelsAsFrames(open(norm).read(), "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})
        view.setStyle({"chain": ligand_chain}, {"stick": {"radius": 0.18}})
        view.animate({"loop": "forward", "interval": 120})
    else:
        first, buf = [], []
        for line in open(full):
            if line.startswith("ENDMDL"):
                first = buf
                break
            buf.append(line)
        view.addModel("".join(l for l in first if l[21:22] != ligand_chain), "pdb")
        view.setStyle({protein_style: {"color": "lightgrey"}})
        frames, cur = [], []
        src = lig if os.path.isfile(lig) else full
        for line in open(src):
            if line.startswith("ENDMDL"):
                if cur:
                    frames.append(cur)
                cur = []
            elif line[:6].strip() in ("ATOM", "HETATM") and line[21:22] == ligand_chain:
                cur.append(line)
        if cur:
            frames.append(cur)
        sel = frames[::max(1, int(stride))]
        for i, fr in enumerate(sel):
            t = i / max(1, len(sel) - 1)
            view.addModel("".join(fr), "pdb")
            view.setStyle({"model": -1},
                          {"stick": {"radius": 0.16,
                                     "color": f"rgb({int(255*t)},{int(60+40*t)},{int(255*(1-t))})"}})
        print(f"{len(frames)} ligand poses, showing {len(sel)} "
              f"(blue = bound, red = released)")
    view.zoomTo()
    return view.show()


def zip_results(run_dir, out_zip=None, include_windows=False, extra=()):
    """
    Package the analysis files (and optionally every window) into one archive.

    Window directories hold the decoy archives and dominate the size, so they
    are excluded unless you ask for them.
    """
    rd = _abs(run_dir)
    out_zip = _abs(out_zip or rd.rstrip("/") + "_results.zip")
    keep = ("interface_scores.txt", "total_scores.txt", "nbr_trajectory.txt",
            "coarsepath.txt", "guide_path", "guide_path.pdb", "spline", "normals",
            "input_path", "lig_trajectory.pdb", "full_trajectory.pdb",
            "trajectory.pdb", "stats.pkl", "checkpoint_meta.json", "run.json",
            "lig.resfile", "coarse.log", "trajectory.log")
    n = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in keep + tuple(extra):
            p = os.path.join(rd, f)
            if os.path.isfile(p):
                z.write(p, os.path.join(os.path.basename(rd), f))
                n += 1
        if include_windows:
            for root, _, files in os.walk(rd):
                for f in files:
                    p = os.path.join(root, f)
                    rel = os.path.relpath(p, rd)
                    if rel in keep:
                        continue
                    z.write(p, os.path.join(os.path.basename(rd), rel))
                    n += 1
    print(f"{n} file(s) -> {out_zip}  ({os.path.getsize(out_zip) / 1e6:.1f} MB)")
    return out_zip


# --------------------------------------------------------------------------
# chemistry: conformer libraries and Rosetta params
#
# LPF needs two ligand files: lig.params (the Rosetta residue type, with
# NBR_ATOM and a PDB_ROTAMERS pointer) and lig_conformers.pdb (the rotamer
# library). Both come from Rosetta's own molfile_to_params.py, fed an SD file
# whose FIRST entry is the bound pose.
#
# For a peptide the same machinery applies, with the peptide treated as one
# large flexible ligand: all of its backbone and side-chain freedom has to live
# in the conformer library, because the options file sets
# -packing:ignore_ligand_chi true and Rosetta will not sample those torsions
# on the fly.
# --------------------------------------------------------------------------

def _rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.warning")
        return Chem, AllChem
    except ImportError:
        raise ImportError(
            "RDKit is needed to build conformer libraries: `pip install rdkit`.\n"
            "Alternatively prepare the SD file yourself (Avogadro / OpenBabel / BCL, "
            "as demo/tutorial.md 1.2 describes) and point the notebook at it.")


def mol_from_sdf(path, sanitize=True, keep_all=False):
    """Read an SD file. Returns the first molecule (its conformers merged in)."""
    Chem, _ = _rdkit()
    mols = [m for m in Chem.SDMolSupplier(_abs(path), removeHs=False,
                                          sanitize=sanitize) if m is not None]
    _require(mols, f"no readable molecule in {path}")
    if keep_all:
        return mols
    base = mols[0]
    for extra in mols[1:]:
        if extra.GetNumAtoms() == base.GetNumAtoms():
            base.AddConformer(extra.GetConformer(), assignId=True)
    n_h = sum(1 for a in base.GetAtoms() if a.GetAtomicNum() == 1)
    formula = __import__("rdkit.Chem.rdMolDescriptors",
                         fromlist=["x"]).CalcMolFormula(base)
    print(f"{os.path.basename(_abs(path))}: {base.GetNumAtoms()} atoms "
          f"({n_h} explicit H), {base.GetNumConformers()} conformer(s), "
          f"formula {formula}")
    if n_h == 0:
        _say(_WARN, "this file has NO explicit hydrogens. Rosetta needs them — "
                    "the tutorial asks for an SD file 'with protons and correctly "
                    "specified bond orders'. They will be added for you "
                    "(generate_conformers add_hydrogens='auto'); check the result, "
                    "or supply a protonated file to control the states yourself.")
    return base


def mol_from_smiles_3d(smiles, n_confs=1, seed=0xF00D, add_hs=True):
    """Build a 3D molecule from SMILES. Only useful when you have no bound pose."""
    Chem, AllChem = _rdkit()
    m = Chem.MolFromSmiles(smiles)
    _require(m is not None, f"RDKit could not parse SMILES: {smiles!r}")
    if add_hs:
        m = Chem.AddHs(m)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = int(seed)
    _require(AllChem.EmbedMultipleConfs(m, numConfs=int(n_confs), params=ps),
             "embedding failed")
    AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=500)
    _say(_WARN, "this geometry is a free-solution guess, NOT a bound pose. LPF starts "
                "from a functionally relevant binding pose and does not find one for "
                "you — dock or predict the complex first.")
    return m


# --------------------------------------------------------------------------
# peptide preparation
#
# A peptide pulled out of a complex is usually not chemically complete: the
# terminal carboxyl is often truncated, side chains can be missing atoms, and
# the ionisable groups carry no charge information. None of that matters while
# it is "residues 1-9 of chain B"; all of it matters the moment it becomes a
# single Rosetta ligand residue with its own partial charges.
# --------------------------------------------------------------------------

#: Heavy atoms of each standard residue, excluding the terminal OXT.
_RESIDUE_HEAVY_ATOMS = {
    "GLY": "N CA C O",
    "ALA": "N CA C O CB",
    "SER": "N CA C O CB OG",
    "CYS": "N CA C O CB SG",
    "THR": "N CA C O CB OG1 CG2",
    "VAL": "N CA C O CB CG1 CG2",
    "LEU": "N CA C O CB CG CD1 CD2",
    "ILE": "N CA C O CB CG1 CG2 CD1",
    "MET": "N CA C O CB CG SD CE",
    "PRO": "N CA C O CB CG CD",
    "PHE": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ",
    "TYR": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH",
    "TRP": "N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "ASP": "N CA C O CB CG OD1 OD2",
    "GLU": "N CA C O CB CG CD OE1 OE2",
    "ASN": "N CA C O CB CG OD1 ND2",
    "GLN": "N CA C O CB CG CD OE1 NE2",
    "HIS": "N CA C O CB CG ND1 CD2 CE1 NE2",
    "LYS": "N CA C O CB CG CD CE NZ",
    "ARG": "N CA C O CB CG CD NE CZ NH1 NH2",
}
_RESIDUE_HEAVY_ATOMS = {k: set(v.split()) for k, v in _RESIDUE_HEAVY_ATOMS.items()}


def peptide_completeness_report(pep_pdb, verbose=True):
    """
    Check a peptide chain residue by residue before converting it to a ligand.

    Reports missing heavy atoms per residue and the state of the C-terminus.
    Returns {"missing": {resid: [atoms]}, "needs_oxt": bool, "needs_o": bool,
    "n_residues": int, "ok": bool}.
    """
    atoms = read_pdb_atoms(pep_pdb)
    by_res, order = {}, []
    for a in atoms:
        if _infer_element(a.name, a.element, a.name_col13) in ("H", "D"):
            continue
        rid = (a.chain, a.resseq, a.icode, a.resname.upper())
        if rid not in by_res:
            by_res[rid] = set()
            order.append(rid)
        by_res[rid].add(a.name)

    _require(order, f"no heavy atoms found in {pep_pdb}")
    last = order[-1]
    missing, unknown = {}, []
    for i, rid in enumerate(order):
        rn = rid[3]
        if rn not in _RESIDUE_HEAVY_ATOMS:
            unknown.append(rid)
            continue
        want = set(_RESIDUE_HEAVY_ATOMS[rn])
        lack = sorted(want - by_res[rid])
        if lack:
            missing[rid] = lack

    needs_o = rid_last_lacks = "O" not in by_res[last]
    needs_oxt = "OXT" not in by_res[last]

    if verbose:
        print(f"{len(order)} residue(s): "
              + "-".join(f"{r[3]}{r[1]}" for r in order[:12])
              + (" ..." if len(order) > 12 else ""))
        if unknown:
            _say(_WARN, "non-standard residue(s) "
                        + ", ".join(f"{r[3]}{r[1]}" for r in unknown)
                        + " — completeness not checked, and a sequence template "
                          "cannot be built for them")
        for rid, lack in missing.items():
            terminal_only = rid == last and set(lack) <= {"O"}
            _say(_WARN if terminal_only else _BAD,
                 f"{rid[3]}{rid[1]} is missing {lack}"
                 + ("  (terminal carboxyl — repairable)" if terminal_only else
                    "  (rebuild this side chain upstream: PyMOL, pdbfixer, "
                    "Rosetta FastRelax)"))
        if needs_oxt:
            _say(_WARN, f"the C-terminal residue {last[3]}{last[1]} has no OXT, so the "
                        "carboxyl is incomplete — call complete_c_terminus()")
        real_problems = {k: v for k, v in missing.items()
                         if not (k == last and set(v) <= {"O"})}
        if not real_problems and not unknown and not needs_oxt and not needs_o:
            _say(_OK, "peptide is chemically complete")
    real_problems = {k: v for k, v in missing.items()
                     if not (k == last and set(v) <= {"O"})}
    return {"missing": missing, "needs_oxt": needs_oxt, "needs_o": needs_o,
            "unknown": unknown, "n_residues": len(order),
            "ok": not real_problems and not unknown}


def complete_c_terminus(pdb_in, pdb_out, co_bond=1.25, verbose=True):
    """
    Add a missing terminal O and/or OXT to the last residue, as a planar carboxyl.

    Both oxygens are placed in the N-CA-C plane, 120 degrees either side of the
    CA-C bond, which is the standard carboxylate geometry (what pdbfixer or
    PyMOL would build). Anything more than the terminal oxygens — a missing side
    chain — is not invented here; fix that upstream.
    """
    atoms = read_pdb_atoms(pdb_in)
    heavy = [a for a in atoms if _infer_element(a.name, a.element, a.name_col13)
             not in ("H", "D")]
    _require(heavy, f"no heavy atoms in {pdb_in}")
    last_rid = (heavy[-1].chain, heavy[-1].resseq, heavy[-1].icode)
    res = {a.name: a for a in atoms
           if (a.chain, a.resseq, a.icode) == last_rid}
    have_o, have_oxt = "O" in res, "OXT" in res
    if have_o and have_oxt:
        if verbose:
            _say(_OK, "C-terminus already has both O and OXT; nothing to do")
        if _abs(pdb_in) != _abs(pdb_out):
            shutil.copy(_abs(pdb_in), _abs(pdb_out))
        return _abs(pdb_out)

    for req in ("N", "CA", "C"):
        _require(req in res, f"terminal residue has no {req} atom; cannot build the "
                             f"carboxyl geometry")
    C, CA, N = res["C"].xyz, res["CA"].xyz, res["N"].xyz

    def unit(v):
        n = np.linalg.norm(v)
        _require(n > 1e-6, "degenerate backbone geometry at the C-terminus")
        return v / n

    v_ca = unit(CA - C)
    normal = np.cross(CA - C, N - CA)
    if np.linalg.norm(normal) < 1e-6:            # N, CA, C collinear
        normal = np.cross(v_ca, np.array([0.0, 0.0, 1.0]))
    normal = unit(normal)
    u = unit(np.cross(normal, v_ca))
    cos120, sin120 = math.cos(math.radians(120.0)), math.sin(math.radians(120.0))

    added = []
    template = res["C"]
    if not have_o and not have_oxt:
        pos = {"O": C + co_bond * (cos120 * v_ca + sin120 * u),
               "OXT": C + co_bond * (cos120 * v_ca - sin120 * u)}
    elif have_o:
        b = -(unit(CA - C) + unit(res["O"].xyz - C))
        pos = {"OXT": C + co_bond * unit(b)}
    else:
        b = -(unit(CA - C) + unit(res["OXT"].xyz - C))
        pos = {"O": C + co_bond * unit(b)}

    out, inserted = [], False
    last_index = max(i for i, a in enumerate(atoms)
                     if (a.chain, a.resseq, a.icode) == last_rid)
    for i, a in enumerate(atoms):
        out.append(a)
        if i == last_index:
            for nm, xyz in pos.items():
                out.append(Atom(record=template.record, name=nm, resname=template.resname,
                                chain=template.chain, resseq=template.resseq,
                                icode=template.icode, x=float(xyz[0]), y=float(xyz[1]),
                                z=float(xyz[2]), occ=1.0, bfac=template.bfac,
                                element="O"))
                added.append(nm)
            inserted = True
    _require(inserted, "internal error: terminal residue not located")
    write_pdb_atoms(pdb_out, out)
    if verbose:
        _say(_OK, f"built {', '.join(added)} on {template.resname}{template.resseq} "
                  f"-> {_abs(pdb_out)}")
        for nm, xyz in pos.items():
            print(f"       {nm}: [{xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}], "
                  f"{np.linalg.norm(xyz - C):.2f} A from C, "
                  f"{GeometryAngle(CA, C, xyz):.1f} deg from CA-C")
    return _abs(pdb_out)


def GeometryAngle(p1, p2, p3):
    """Angle p1-p2-p3 in degrees."""
    a, b = np.asarray(p1) - np.asarray(p2), np.asarray(p3) - np.asarray(p2)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


_PROTONATION_PATTERNS = (
    # (name, SMARTS, atom index in the match to change, charge, explicit H)
    ("carboxylate (Asp/Glu/C-terminus)", "[CX3](=[OX1])[OX2H1]", 2, -1, 0),
    ("primary amine (Lys/N-terminus)", "[NX3;H2;!$([NX3][CX3]=[OX1]);!$([NX3]C=N)]",
     0, +1, 3),
    ("guanidinium (Arg)", "[NX3][CX3](=[NX2;H2])[NX3]", 2, +1, 2),
    # a proline N-terminus is a secondary amine, so the H2 pattern above misses it
    ("secondary amine N-terminus (Pro)",
     "[NX3;H1;R;!$([NX3][CX3]=[OX1]);!$([NX3]C=N)]", 0, +1, 2),
)


def set_physiological_protonation(mol, verbose=True):
    """
    Put the ionisable groups in their pH-7 states before hydrogens are added.

    RDKit's sequence template produces a neutral peptide: -COOH, -NH2 on lysine,
    neutral guanidine. As a Rosetta ligand those neutral forms would carry the
    wrong partial charges at every interface it crosses, which is exactly what
    the energy profile measures. Histidine is left alone — its tautomer is a
    judgement call, and Rosetta cannot be told which one to use for a ligand.
    """
    Chem, _ = _rdkit()
    # The patterns below set explicit hydrogen counts, which would collide with
    # hydrogens that are already real atoms in the graph (valence errors). So
    # work on the heavy-atom graph and put the hydrogens back afterwards if the
    # caller had them.
    had_explicit_h = any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())
    m = Chem.RWMol(Chem.RemoveHs(mol) if had_explicit_h else Chem.Mol(mol))
    counts = {}
    for label, smarts, idx, charge, n_h in _PROTONATION_PATTERNS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        hits = m.GetSubstructMatches(patt)
        for hit in hits:
            a = m.GetAtomWithIdx(hit[idx])
            if a.GetFormalCharge() == charge:
                continue
            a.SetFormalCharge(charge)
            a.SetNumExplicitHs(n_h)
            a.SetNoImplicit(True)
            counts[label] = counts.get(label, 0) + 1
    out = m.GetMol()
    Chem.SanitizeMol(out)
    if had_explicit_h:
        out = Chem.AddHs(out, addCoords=out.GetNumConformers() > 0)
    if verbose:
        if counts:
            for label, n in counts.items():
                print(f"       {n} x {label}")
        else:
            print("       no ionisable groups changed")
        has_his = out.HasSubstructMatch(Chem.MolFromSmarts("c1cnc[nH]1")) or \
            out.HasSubstructMatch(Chem.MolFromSmarts("c1c[nH]cn1"))
        if has_his:
            _say(_WARN, "histidine present: left neutral, and its tautomer is whatever "
                        "the template produced. Check it if it is near the interface.")
        print(f"       net formal charge: {Chem.GetFormalCharge(out):+d}")
    return out


def peptide_mol_from_pdb(pep_pdb, sequence=None, template_smiles=None,
                         add_hydrogens=True, physiological_protonation=True,
                         verbose=True):
    """
    Turn a peptide chain PDB into an RDKit molecule with correct bond orders,
    keeping the bound coordinates.

    A PDB carries no bond orders, so they are transferred from a template built
    from the sequence (RDKit's MolFromSequence), which assumes complete
    residues and a free C-terminal -COOH. Run peptide_completeness_report()
    first, and complete_c_terminus() if the terminal oxygens are missing;
    otherwise the heavy-atom graphs will not match and the transfer fails.

    If the template match fails, bond perception from geometry is tried as a
    fallback. If that fails too, prepare the SD file yourself as the upstream
    tutorial describes (section 1.2.1) and skip this function.
    """
    Chem, AllChem = _rdkit()
    pep_pdb = _abs(pep_pdb)

    # strip hydrogens at the file level: MolFromPDBFile's removeHs is ineffective
    # when sanitisation is off, and leftover hydrogens break the template match
    # against a heavy-atom-only sequence template
    all_atoms = read_pdb_atoms(pep_pdb)
    heavy_atoms = [a for a in all_atoms
                   if _infer_element(a.name, a.element, a.name_col13) not in ("H", "D")]
    _require(heavy_atoms, f"no heavy atoms in {pep_pdb}")
    heavy_pdb = os.path.join(os.path.dirname(pep_pdb),
                             "_heavy_" + os.path.basename(pep_pdb))
    write_pdb_atoms(heavy_pdb, heavy_atoms)

    raw = Chem.MolFromPDBFile(heavy_pdb, removeHs=False, sanitize=False)
    _require(raw is not None, f"RDKit could not read {heavy_pdb}")
    try:
        Chem.SanitizeMol(raw, Chem.SanitizeFlags.SANITIZE_ALL
                         ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                         ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        pass
    n_heavy = raw.GetNumAtoms()

    seq = sequence
    if seq is None and template_smiles is None:
        chains = {a.chain for a in heavy_atoms}
        _require(len(chains) == 1,
                 f"{os.path.basename(pep_pdb)} holds chains {sorted(chains)}; "
                 "extract a single peptide chain first")
        seq, resids = chain_sequence(pep_pdb, chains.pop())
        if "X" in seq:
            bad = [r for r, c in zip(resids, seq) if c == "X"]
            raise ValueError(
                f"non-standard residue(s) {bad} cannot be templated from sequence. "
                f"Pass template_smiles=... or supply the SD file directly.")
        if verbose:
            print(f"sequence from PDB: {seq}  ({len(seq)} residues, {n_heavy} heavy atoms)")

    mol = None
    if template_smiles is not None:
        templ = Chem.MolFromSmiles(template_smiles)
        _require(templ is not None, f"bad template_smiles: {template_smiles!r}")
    elif seq is not None:
        templ = Chem.MolFromSequence(seq)
        _require(templ is not None, f"RDKit could not build a template for {seq!r}")
    else:
        templ = None

    if templ is not None:
        if templ.GetNumAtoms() != n_heavy:
            diff = templ.GetNumAtoms() - n_heavy
            _say(_WARN, f"template has {templ.GetNumAtoms()} heavy atoms, the structure "
                        f"has {n_heavy} ({diff:+d}). Run peptide_completeness_report() — "
                        f"the usual causes are a missing terminal O/OXT or an "
                        f"incomplete side chain.")
        try:
            mol = AllChem.AssignBondOrdersFromTemplate(templ, raw)
            if verbose:
                _say(_OK, "bond orders transferred from the sequence template")
        except Exception as exc:
            _say(_WARN, f"template match failed ({exc}); trying bond perception")

    if mol is None:
        try:
            from rdkit.Chem import rdDetermineBonds
            mol = Chem.Mol(raw)
            rdDetermineBonds.DetermineBondOrders(mol, charge=0)
            _say(_WARN, "bond orders perceived from geometry — CHECK them (aromatic "
                        "rings, carboxylates, His tautomer) before trusting the params")
        except Exception as exc:
            raise ValueError(
                f"could not establish bond orders for {os.path.basename(pep_pdb)}: "
                f"{exc}\nRun peptide_completeness_report(), repair what it lists "
                f"(complete_c_terminus() handles the terminal oxygens), or prepare the "
                f"SD file yourself as in demo/tutorial.md 1.2.1 and pass it in.") from None

    Chem.SanitizeMol(mol)
    if physiological_protonation:
        if verbose:
            print("protonation states at pH 7:")
        mol = set_physiological_protonation(mol, verbose=verbose)
    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=True)
    _require(mol.GetNumConformers() >= 1, "no coordinates survived the conversion")
    if verbose:
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        print(f"peptide molecule: {mol.GetNumAtoms()} atoms ({n_heavy} heavy), "
              f"formula {CalcMolFormula(mol)}, net charge "
              f"{Chem.GetFormalCharge(mol):+d}")
        print("note: from here the peptide is ONE Rosetta ligand residue. All of its "
              "backbone and side-chain flexibility has to come from the conformer "
              "library built next.")
    return mol


def _rotatable_bonds(mol, include_amide=False):
    """Single, acyclic bonds between non-terminal heavy atoms; amides optional."""
    Chem, _ = _rdkit()
    out = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.SINGLE or b.IsInRing():
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        if a1.GetDegree() < 2 or a2.GetDegree() < 2:
            continue
        if not include_amide and _is_amide_bond(a1, a2):
            continue
        out.append((a1.GetIdx(), a2.GetIdx()))
    return out


def _is_amide_bond(a1, a2):
    for c, n in ((a1, a2), (a2, a1)):
        if c.GetSymbol() == "C" and n.GetSymbol() == "N":
            for b in c.GetBonds():
                if b.GetBondTypeAsDouble() == 2.0 and \
                        b.GetOtherAtom(c).GetSymbol() in ("O", "S"):
                    return True
    return False


def _torsion_quad(mol, i, j):
    """Pick a (a,i,j,b) atom quadruple defining the dihedral about bond i-j."""
    a = next((n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
              if n.GetIdx() != j and n.GetAtomicNum() > 1), None)
    b = next((n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors()
              if n.GetIdx() != i and n.GetAtomicNum() > 1), None)
    return None if a is None or b is None else (a, i, j, b)


def _lpf_rmsd(ref, mob, nbr_idx):
    """
    The RMSD metric LPF itself uses (util.ConformerSampler.optimal_align_rmsd):
    translate both so NBR_ATOM is at the origin, apply the optimal rotation, then
    plain RMSD over all atoms. Matching it here means conformer_rmsd_threshold
    in run.json means the same thing as the numbers reported below.
    """
    from scipy.spatial.transform import Rotation
    ref = np.asarray(ref, dtype=float) - np.asarray(ref[nbr_idx], dtype=float)
    mob = np.asarray(mob, dtype=float) - np.asarray(mob[nbr_idx], dtype=float)
    rot = Rotation.align_vectors(ref, mob)[0]
    return float(np.sqrt(np.sum((rot.apply(mob) - ref) ** 2) / len(ref)))


def _conf_coords(mol, cid):
    return np.array(mol.GetConformer(cid).GetPositions(), dtype=float)


def _prune_conformers(mol, nbr_idx, threshold, keep_first=True, verbose=True):
    """Greedy RMSD pruning in LPF's own metric, always keeping conformer 0."""
    ids = [c.GetId() for c in mol.GetConformers()]
    if threshold <= 0 or len(ids) < 2:
        return ids
    coords = {i: _conf_coords(mol, i) for i in ids}
    keep = [ids[0]] if keep_first else []
    for i in ids[1:] if keep_first else ids:
        if all(_lpf_rmsd(coords[k], coords[i], nbr_idx) >= threshold for k in keep):
            keep.append(i)
    if verbose:
        print(f"pruned {len(ids)} -> {len(keep)} conformers at "
              f"{threshold:.2f} A RMSD separation")
    return keep


def generate_conformers(mol, n_confs=200, prune_rmsd=0.5, nbr_idx=0, seed=0xF00D,
                        minimize=True, max_iters=400, keep_input_pose=True,
                        add_hydrogens="auto", verbose=True):
    """
    Build a small-molecule conformer library by ETKDG embedding (the role BCL
    plays in the upstream tutorial).

    The input conformer — your bound pose — is kept as conformer 0, because
    molfile_to_params writes the FIRST SD entry to lig.pdb, and that file is
    what becomes the ligand in start.pdb.

    `add_hydrogens="auto"` adds explicit hydrogens when the molecule has none,
    which is the common case for a ligand exported from PyMOL. Without them the
    Rosetta residue type is built from heavy atoms only: a hydroxyl cannot
    donate a hydrogen bond, packing and fa_rep are wrong, and every energy the
    run reports is wrong with them. Pass True to force, False to leave alone.
    """
    Chem, AllChem = _rdkit()
    work = Chem.Mol(mol)
    n_h = sum(1 for a in work.GetAtoms() if a.GetAtomicNum() == 1)
    if add_hydrogens is True or (add_hydrogens == "auto" and n_h == 0):
        before = work.GetNumAtoms()
        work = Chem.AddHs(work, addCoords=work.GetNumConformers() > 0)
        if verbose:
            _say(_OK, f"added {work.GetNumAtoms() - before} explicit hydrogen(s): "
                      f"{before} -> {work.GetNumAtoms()} atoms. Rosetta needs them; "
                      f"supply a protonated SD file if you want to set the "
                      f"protonation states yourself.")
    bound = _conf_coords(work, work.GetConformer().GetId()) if work.GetNumConformers() else None
    if keep_input_pose:
        _require(bound is not None, "keep_input_pose needs the input to have coordinates")

    ps = AllChem.ETKDGv3()
    ps.randomSeed = int(seed)
    ps.pruneRmsThresh = -1.0
    ps.useSmallRingTorsions = True
    extra = Chem.Mol(work)
    extra.RemoveAllConformers()
    cids = AllChem.EmbedMultipleConfs(extra, numConfs=int(n_confs), params=ps)
    _require(len(cids), "ETKDG embedding produced no conformers")
    if minimize:
        AllChem.MMFFOptimizeMoleculeConfs(extra, maxIters=int(max_iters))

    out = Chem.Mol(work)
    out.RemoveAllConformers()
    if keep_input_pose:
        c = Chem.Conformer(work.GetNumAtoms())
        for i, p in enumerate(bound):
            c.SetAtomPosition(i, p.tolist())
        out.AddConformer(c, assignId=True)
    for cid in cids:
        out.AddConformer(extra.GetConformer(cid), assignId=True)
    if verbose:
        print(f"embedded {len(cids)} conformers"
              + (" (+ the input bound pose as conformer 0)" if keep_input_pose else ""))
    keep = _prune_conformers(out, nbr_idx, prune_rmsd, keep_first=keep_input_pose,
                             verbose=verbose)
    final = Chem.Mol(out)
    final.RemoveAllConformers()
    for cid in keep:
        final.AddConformer(out.GetConformer(cid), assignId=True)
    return final


def generate_peptide_conformers(mol, n_local=120, n_medium=120, n_global=60,
                                sigma_local_deg=12.0, sigma_medium_deg=45.0,
                                n_torsions_local=3, n_torsions_medium=8,
                                prune_rmsd=0.6, nbr_idx=0, seed=0xF00D,
                                minimize=True, max_iters=150, verbose=True):
    """
    Build a peptide conformer library as a ladder from the bound pose outwards.

    Free embedding alone is the wrong tool here. LPF filters the library to
    conformers within conformer_rmsd_threshold of the *current* pose at every
    window, so what it needs is a library whose members are reachable from one
    another in small RMSD steps — a connected ladder from the bound
    conformation to extended ones. A purely local ensemble would never let the
    peptide unfold on the way out; a purely random one leaves gaps the
    subsampling filter cannot cross.

    So three populations are mixed:
      n_local   small perturbations of a few torsions   (stays near the bound pose)
      n_medium  larger perturbations of more torsions   (the middle of the ladder)
      n_global  free ETKDG embeddings                   (extended / collapsed states)

    Check the result with conformer_library_report() before spending Rosetta time.
    """
    Chem, AllChem = _rdkit()
    from rdkit.Chem import rdMolTransforms as rmt

    _require(mol.GetNumConformers() >= 1, "the input molecule needs the bound pose")
    rng = np.random.default_rng(int(seed))
    base = Chem.Mol(mol)
    base_cid = base.GetConformer().GetId()
    bound = _conf_coords(base, base_cid)

    rb = _rotatable_bonds(base, include_amide=False)
    quads = [q for q in (_torsion_quad(base, i, j) for i, j in rb) if q]
    if verbose:
        print(f"{len(quads)} rotatable torsions (amide bonds excluded)")
    if not quads:
        _say(_WARN, "no rotatable torsions found; falling back to free embedding only")
        n_local = n_medium = 0

    out = Chem.Mol(base)
    out.RemoveAllConformers()
    c0 = Chem.Conformer(base.GetNumAtoms())
    for i, p in enumerate(bound):
        c0.SetAtomPosition(i, p.tolist())
    out.AddConformer(c0, assignId=True)      # conformer 0 = the bound pose

    mp = AllChem.MMFFGetMoleculeProperties(base) if minimize else None
    if minimize and mp is None:
        _say(_WARN, "MMFF typing failed; conformers will not be minimised")
        minimize = False

    n_fail = 0
    for n_make, sigma, n_tors in ((n_local, sigma_local_deg, n_torsions_local),
                                  (n_medium, sigma_medium_deg, n_torsions_medium)):
        for _ in range(int(n_make)):
            tmp = Chem.Mol(base)
            conf = tmp.GetConformer(base_cid)
            pick = rng.choice(len(quads), size=min(int(n_tors), len(quads)), replace=False)
            for k in pick:
                a, b, c, d = quads[int(k)]
                try:
                    cur = rmt.GetDihedralDeg(conf, a, b, c, d)
                    rmt.SetDihedralDeg(conf, a, b, c, d,
                                       cur + float(rng.normal(0.0, sigma)))
                except Exception:
                    n_fail += 1
            if minimize:
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(
                        tmp, AllChem.MMFFGetMoleculeProperties(tmp), confId=base_cid)
                    ff.Minimize(maxIts=int(max_iters))
                except Exception:
                    n_fail += 1
            out.AddConformer(tmp.GetConformer(base_cid), assignId=True)

    if int(n_global) > 0:
        ps = AllChem.ETKDGv3()
        ps.randomSeed = int(seed) + 1
        ps.pruneRmsThresh = -1.0
        # Plain ETKDG cannot embed a large flexible chain from distance bounds
        # alone — for a 20-residue peptide it returns nothing at all, silently
        # leaving this population empty. Random starting coordinates and a
        # bigger iteration budget are the standard remedy.
        if base.GetNumAtoms() > 120:
            ps.useRandomCoords = True
            ps.maxIterations = 2000
            if verbose:
                print(f"       {base.GetNumAtoms()} atoms: using random starting "
                      f"coordinates for the free embeddings")
        g = Chem.Mol(base)
        g.RemoveAllConformers()
        gids = AllChem.EmbedMultipleConfs(g, numConfs=int(n_global), params=ps)
        if minimize and len(gids):
            AllChem.MMFFOptimizeMoleculeConfs(g, maxIters=int(max_iters))
        for cid in gids:
            out.AddConformer(g.GetConformer(cid), assignId=True)
        if verbose:
            print(f"local {int(n_local)} + medium {int(n_medium)} + "
                  f"global {len(gids)} = {out.GetNumConformers() - 1} "
                  f"conformers around the bound pose")
        if int(n_global) > 0 and len(gids) == 0:
            _say(_WARN, "every free embedding failed, so the library has no "
                        "globally different conformations — only perturbations of "
                        "the bound pose. Expect conformer_library_report to call it "
                        "disconnected.")
    if n_fail and verbose:
        _say(_WARN, f"{n_fail} torsion/minimisation operation(s) failed and were skipped")

    keep = _prune_conformers(out, nbr_idx, prune_rmsd, keep_first=True, verbose=verbose)
    final = Chem.Mol(out)
    final.RemoveAllConformers()
    for cid in keep:
        final.AddConformer(out.GetConformer(cid), assignId=True)
    return final


def conformer_library_report(mol, nbr_idx=0, thresholds=(1.5, 2.0, 3.0, 4.0),
                             plot=False):
    """
    Diagnose a conformer library the way LPF will use it.

    For each candidate conformer_rmsd_threshold this reports:
      * how many conformers survive subsampling at the bound pose — the number
        of rotamers Rosetta actually gets in window 0;
      * whether the library is *connected* at that threshold, i.e. whether
        single-linkage chains of steps smaller than the threshold reach from the
        bound pose to the most distant conformer. If it is not connected, the
        peptide can never walk from the bound conformation to the far ones,
        however many conformers the file contains.
    """
    n = mol.GetNumConformers()
    _require(n >= 2, "need at least two conformers to report on")
    ids = [c.GetId() for c in mol.GetConformers()]
    coords = {i: _conf_coords(mol, i) for i in ids}
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = _lpf_rmsd(coords[ids[i]], coords[ids[j]], nbr_idx)
    to_bound = D[0]
    print(f"{n} conformers; RMSD to the bound pose: "
          f"min {to_bound[1:].min():.2f}, median {np.median(to_bound[1:]):.2f}, "
          f"max {to_bound.max():.2f} A")
    print(f"  {'threshold':>10s}{'kept at bound pose':>20s}{'library connected':>20s}")
    far = int(np.argmax(to_bound))
    out = {}
    for t in thresholds:
        kept = int((to_bound < t).sum())
        adj = D < t
        seen, stack = {0}, [0]
        while stack:
            u = stack.pop()
            for v in np.where(adj[u])[0]:
                if v not in seen:
                    seen.add(int(v))
                    stack.append(int(v))
        conn = far in seen
        out[t] = (kept, conn, len(seen))
        print(f"  {t:>10.2f}{kept:>20d}{('yes' if conn else 'NO'):>20s}"
              f"   ({len(seen)}/{n} reachable)")
        if kept <= 2:
            _say(_WARN, f"at threshold {t} only {kept} conformer(s) survive — the "
                        "molecule will behave as a near-rigid body")
        if not conn:
            _say(_WARN, f"at threshold {t} the library is disconnected: the far "
                        "conformations are unreachable from the bound pose")
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].hist(to_bound[1:], bins=30, color="#1f77b4")
        ax[0].set_xlabel("RMSD to bound pose (A)")
        ax[0].set_ylabel("conformers")
        ax[0].set_title("ladder from the bound pose")
        im = ax[1].imshow(D, cmap="viridis")
        ax[1].set_title("pairwise RMSD (A)")
        fig.colorbar(im, ax=ax[1])
        fig.tight_layout()
    return D, out


def write_conformer_sdf(mol, path, aromatic_bonds=True, verbose=True):
    """
    Write the multi-conformer SD file that molfile_to_params consumes.

    `aromatic_bonds=True` writes aromatic bonds as MDL type 4 rather than a
    Kekule structure, which is what molfile_to_params expects ("Expects that
    the input ligand has already had aromaticity perceived") and what the
    upstream tutorial does by hand to its paraoxon SDF.
    """
    Chem, _ = _rdkit()
    path = _abs(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    w = Chem.SDWriter(path)
    if aromatic_bonds:
        w.SetKekulize(False)
    n = 0
    for c in mol.GetConformers():
        w.write(mol, confId=c.GetId())
        n += 1
    w.close()
    if verbose:
        print(f"{n} conformer(s) -> {path}"
              + ("  (aromatic bonds as type 4)" if aromatic_bonds else ""))
        print("       conformer 0 is the bound pose; molfile_to_params writes it to "
              "lig.pdb, which becomes the ligand in start.pdb")
    return path


# --------------------------------------------------------------------------
# molfile_to_params: SD file -> lig.params + lig_conformers.pdb + lig.pdb
# --------------------------------------------------------------------------

_MFTP_BASE = ("https://raw.githubusercontent.com/RosettaCommons/rosetta/main/"
              "source/scripts/python/public")
_MFTP_PARTS = ("molfile_to_params.py",
               "rosetta_py/__init__.py",
               "rosetta_py/io/__init__.py",
               "rosetta_py/io/mdl_molfile.py",
               "rosetta_py/utility/__init__.py",
               "rosetta_py/utility/rankorder.py",
               "rosetta_py/utility/r3.py")


def ensure_molfile_to_params(dest_dir, *hints, rosetta_bin=None, allow_download=True):
    """
    Return a usable molfile_to_params.py, preferring the one in your Rosetta tree.

    If none is found and `allow_download` is set, the script and the four
    rosetta_py modules it imports are fetched from the RosettaCommons
    repository into `dest_dir`. Those files are part of Rosetta and are covered
    by Rosetta's licence, not by LigandPathFinder's — you need the same
    academic/non-commercial licence that lets you run the binary.
    """
    local = find_molfile_to_params(*hints, rosetta_bin=rosetta_bin)
    if local:
        _say(_OK, f"molfile_to_params.py from your Rosetta tree: {local}")
        return local
    if not allow_download:
        raise FileNotFoundError(
            "molfile_to_params.py not found. It ships with the Rosetta source tree at "
            "main/source/scripts/python/public/molfile_to_params.py — point "
            "find_molfile_to_params() at it, or set allow_download=True.")

    import urllib.request
    dest_dir = _abs(dest_dir)
    print(f"fetching molfile_to_params.py + its rosetta_py modules from "
          f"RosettaCommons into {dest_dir}")
    print("       (these are Rosetta files, used under your Rosetta licence)")
    for rel in _MFTP_PARTS:
        out = os.path.join(dest_dir, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        urllib.request.urlretrieve(f"{_MFTP_BASE}/{rel}", out)
    script = os.path.join(dest_dir, "molfile_to_params.py")
    proc = subprocess.run([sys.executable, script, "--help"],
                          capture_output=True, text=True, cwd=dest_dir)
    if "molfile_to_params" not in (proc.stdout + proc.stderr):
        raise RuntimeError(f"downloaded molfile_to_params.py does not run:\n"
                           f"{(proc.stderr or proc.stdout)[:1500]}")
    _say(_OK, f"molfile_to_params.py ready: {script}")
    return script


def run_molfile_to_params(sdf, out_dir, script=None, resname="lig", prefix="lig",
                          nbr_atom_index=None, max_confs=None, recharge=None,
                          keep_names=False, extra_args=(), python_exe=None,
                          clobber=True, verbose=True):
    """
    Run Rosetta's molfile_to_params.py on an SD file.

    `prefix` must stay "lig": LPF hardcodes the filenames lig.params and
    lig_conformers.pdb (lpf_trajectory._setup_window / _read_nbr_atom).
    `resname` is the 3-character PDB residue name Rosetta will use ("pep" reads
    nicely for a peptide).

    `nbr_atom_index` is a 1-based index into the SD file's atom order and sets
    NBR_ATOM directly, which also lets molfile_to_params compute a matching
    NBR_RADIUS. Rosetta overlays every rotamer on NBR_ATOM and on the
    orientation of its bonded atoms, so choose a central, well-connected atom —
    for a peptide a backbone atom of a middle residue, not a terminal one.

    Returns {"params", "conformers", "lig_pdb", "stdout"}.
    """
    sdf, out_dir = _abs(sdf), _abs(out_dir)
    _require(os.path.isfile(sdf), f"SD file not found: {sdf}")
    _require(prefix == "lig",
             "prefix must be 'lig' — LPF looks for lig.params and lig_conformers.pdb "
             "by those exact names")
    _require(len(resname) <= 3, f"resname must be at most 3 characters, got {resname!r}")
    os.makedirs(out_dir, exist_ok=True)
    script = script or ensure_molfile_to_params(os.path.join(out_dir, "_mftp"))

    local_sdf = os.path.join(out_dir, os.path.basename(sdf))
    if _abs(local_sdf) != sdf:
        shutil.copy(sdf, local_sdf)

    argv = [python_exe or sys.executable, _abs(script),
            "-n", resname, "-p", prefix, "--conformers-in-one-file"]
    if clobber:
        argv.append("--clobber")
    if nbr_atom_index is not None:
        argv += [f"--nbr_atom={int(nbr_atom_index)}"]
    if max_confs is not None:
        argv += ["-m", str(int(max_confs))]
    if recharge is not None:
        argv += [f"--recharge={recharge}"]
    if keep_names:
        argv.append("--keep-names")
    argv += list(extra_args) + [os.path.basename(local_sdf)]

    if verbose:
        print("$ " + " ".join(argv))
    env = dict(os.environ)
    env["PYTHONPATH"] = (os.path.dirname(_abs(script)) + os.pathsep
                         + env.get("PYTHONPATH", ""))
    proc = subprocess.run(argv, cwd=out_dir, capture_output=True, text=True, env=env)
    blob = (proc.stdout or "") + (proc.stderr or "")
    if verbose:
        print(textwrap.indent(blob.strip()[:3000], "       "))
    if proc.returncode != 0:
        raise RuntimeError(
            f"molfile_to_params.py exited {proc.returncode}:\n"
            + textwrap.indent(blob[-3000:], "    ")
            + "\nCommon causes: a Kekule SD file where aromatic bonds are expected "
              "(write it with aromatic_bonds=True), an atom-name clash, or more "
              "rotatable bonds than it will accept.")

    params = os.path.join(out_dir, f"{prefix}.params")
    confs = os.path.join(out_dir, f"{prefix}_conformers.pdb")
    ligpdb = os.path.join(out_dir, f"{prefix}.pdb")
    missing = [p for p in (params, confs, ligpdb) if not os.path.isfile(p)]
    if missing:
        raise RuntimeError(
            f"molfile_to_params.py reported success but did not write "
            f"{[os.path.basename(m) for m in missing]}.\nIts output was:\n"
            + textwrap.indent(blob[-2000:], "    "))

    names = params_atom_names(params)
    nconf = count_conformers(confs)
    txt = open(params).read()
    _say(_OK, f"{os.path.basename(params)}: residue {params_resname(params)}, "
              f"{len(names)} atoms, NBR_ATOM {params_nbr_atom(params)}, "
              f"NBR_RADIUS {params_nbr_radius(params):.3f}")
    print(f"       {os.path.basename(confs)}: {nconf} rotamer(s)")
    if "PDB_ROTAMERS" not in txt:
        _say(_BAD, "lig.params has no PDB_ROTAMERS line, so Rosetta will ignore the "
                   "conformer library entirely")
    n_chi = txt.count("\nCHI ")
    if n_chi > 24:
        print(f"       {n_chi} CHI records — the options file sets "
              f"-packing:ignore_ligand_chi true, so these are not sampled on the fly; "
              f"all flexibility comes from the {nconf} rotamers")
    return {"params": params, "conformers": confs, "lig_pdb": ligpdb, "stdout": blob}


def params_bonds(params_path):
    """Heavy-atom connectivity from the params BOND records."""
    bonds = []
    for ln in _params_lines(params_path):
        if ln.startswith("BOND"):
            parts = ln.split()
            if len(parts) >= 3:
                bonds.append((parts[1], parts[2]))
    return bonds


def map_params_atoms_to_source(lig_pdb, source_pdb, tol=0.02):
    """
    Match molfile_to_params' invented atom names back to the original PDB.

    molfile_to_params renames and reorders atoms (C1, N1, ... in atom-tree
    order), which for a peptide collapsed into one residue means you lose track
    of which atom was which residue's CA. lig.pdb holds the first SD conformer —
    the bound pose — with the new names at the original coordinates, so the two
    files can be matched on coordinates.

    Returns {params_atom_name: (resname, resseq, original_atom_name)}.
    """
    from scipy.spatial import cKDTree
    lig = read_pdb_atoms(_abs(lig_pdb))
    src = read_pdb_atoms(_abs(source_pdb))
    _require(lig and src, "both files must contain atoms")
    tree = cKDTree(np.array([a.xyz for a in src]))
    mapping, unmatched = {}, []
    for a in lig:
        d, i = tree.query(a.xyz)
        if d <= tol:
            s = src[int(i)]
            mapping[a.name] = (s.resname, s.resseq, s.name)
        else:
            unmatched.append((a.name, float(d)))
    print(f"matched {len(mapping)}/{len(lig)} params atoms back to {os.path.basename(_abs(source_pdb))}")
    if unmatched:
        hs = [n for n, _ in unmatched if _infer_element(n) == "H"]
        other = [(n, d) for n, d in unmatched if _infer_element(n) != "H"]
        if hs:
            print(f"       {len(hs)} unmatched hydrogen(s) — expected if the source "
                  f"file had none or they were rebuilt")
        if other:
            _say(_WARN, f"unmatched heavy atoms: {other[:6]}")
    return mapping


def describe_atom(params_atom_name, mapping):
    """'C17  =  CA of LEU 5' — for reporting which atom the path actually tracks."""
    m = mapping.get(params_atom_name)
    if not m:
        return f"{params_atom_name} (no match in the source structure)"
    resname, resseq, orig = m
    return f"{params_atom_name} = {orig} of {resname}{resseq}"


def suggest_nbr_atom_index(mol, nbr_candidates=None, verbose=True):
    """
    Pick the most central heavy atom of an RDKit molecule, as a 1-based SD index
    suitable for run_molfile_to_params(nbr_atom_index=...).

    Scored by maximum distance to any other heavy atom, over every conformer,
    with terminal atoms (fewer than two heavy neighbours) rejected: Rosetta
    aligns rotamers on NBR_ATOM *and* the orientation of its bonded atoms, so a
    tip atom gives an unstable reference frame.
    """
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    _require(heavy, "no heavy atoms")
    ok = []
    for i in heavy:
        deg = sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() > 1)
        if deg >= 2:
            ok.append(i)
    pool = ok or heavy
    if nbr_candidates:
        pool = [i for i in pool if i in set(nbr_candidates)] or pool

    worst = {i: 0.0 for i in pool}
    for c in mol.GetConformers():
        xyz = np.array(c.GetPositions())[heavy]
        pos = {a: k for k, a in enumerate(heavy)}
        for i in pool:
            worst[i] = max(worst[i], float(np.linalg.norm(xyz - xyz[pos[i]], axis=1).max()))
    ranked = sorted(worst.items(), key=lambda kv: kv[1])
    best = ranked[0][0]
    if verbose:
        print(f"{'SD index':>9s} {'element':>8s} {'heavy nbrs':>11s}  "
              f"max distance to any heavy atom (A)")
        for i, d in ranked[:6]:
            a = mol.GetAtomWithIdx(i)
            deg = sum(1 for n in a.GetNeighbors() if n.GetAtomicNum() > 1)
            print(f"{i + 1:>9d} {a.GetSymbol():>8s} {deg:>11d}  {d:8.2f}"
                  f"{'   <- chosen' if i == best else ''}")
        print(f"pass nbr_atom_index={best + 1} to run_molfile_to_params()")
    return best + 1


def conformer_subsampling_preview(pose_pdb, conformers_pdb, ligand_chain, params_path,
                                  thresholds=(1.5, 2.0, 3.0, 4.0)):
    """
    How many library conformers survive LPF's filter at a given pose.

    Mirrors util.ConformerSampler.subsample_pdb exactly: each library conformer
    is optimally superposed on the pose about NBR_ATOM and kept if the RMSD is
    below conformer_rmsd_threshold, with the pose itself always appended. If
    this returns 1-2 at your chosen threshold, the molecule is effectively rigid
    for that window however large the library is.
    """
    nbr = params_nbr_atom(params_path)
    names = params_atom_names(params_path)
    pose = {a.name: (a.x, a.y, a.z)
            for a in read_pdb_atoms(pose_pdb) if a.chain == ligand_chain}
    _require(pose, f"no atoms in chain {ligand_chain!r} of {pose_pdb}")
    missing = [n for n in names if n not in pose]
    _require(not missing, f"pose is missing params atoms: {missing[:6]}")
    ref = np.array([pose[n] for n in names])
    nbr_idx = names.index(nbr)

    lib = read_conformers(conformers_pdb)
    _require(lib, f"no conformers in {conformers_pdb}")
    rms = []
    for b in lib:
        if any(n not in b for n in names):
            continue
        rms.append(_lpf_rmsd(ref, np.array([b[n] for n in names]), nbr_idx))
    rms = np.array(rms)
    print(f"{len(rms)} library conformers vs this pose (NBR_ATOM {nbr}): "
          f"RMSD min {rms.min():.2f}, median {np.median(rms):.2f}, max {rms.max():.2f} A")
    print(f"  {'conformer_rmsd_threshold':>26s}{'rotamers Rosetta gets':>24s}")
    out = {}
    for t in thresholds:
        n = int((rms < t).sum()) + 1          # LPF always appends the pose itself
        out[t] = n
        print(f"  {t:>26.2f}{n:>24d}")
        if n <= 2:
            _say(_WARN, f"at {t} A the ligand is effectively rigid in this window")
    return rms, out
