#!/usr/bin/env python3
"""
Self-test for this directory. Run it after editing lpf_nb.py or regenerating
the notebooks:

    python selftest.py              # everything
    python selftest.py notebooks    # just one group

Checks, none of which need Rosetta:
  sources     every module parses
  notebooks   all four notebooks are valid, syntactically sound, and the Colab
              pair carry byte-identical copies of lpf_nb.py and the shim
  execute     every notebook code cell actually runs, with Rosetta stubbed out
  cli         every lpf_run.py subcommand defines each option its stage
              functions touch, and every --help renders
  shim        pyrosetta_rosetta_scripts.py --self-test
  roundtrip   PDB writing reproduces the upstream demo byte-for-byte on the
              fields LPF and Rosetta parse, and LPF's own reader agrees
"""
import ast
import base64
import contextlib
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SOURCES = ("lpf_nb.py", "lpf_run.py", "pyrosetta_rosetta_scripts.py",
           "build_notebooks.py", "make_tutorial.py", "selftest.py")
NOTEBOOKS = sorted(glob.glob(os.path.join(HERE, "notebooks", "LPF_*.ipynb"))
                   or glob.glob(os.path.join(HERE, "LPF_*.ipynb")))
DEMO_START = None
for cand in (os.environ.get("LPF_REPO", ""),
             os.path.join(HERE, "LigandPathFinder"),
             os.path.join(os.path.dirname(HERE), "LigandPathFinder")):
    if not cand:
        continue
    p = os.path.join(cand, "demo/coarse_path/start.pdb")
    if os.path.isfile(p):
        DEMO_START = p
        REPO = cand
        break

results = []


def check(name, fn):
    print(f"\n--- {name} ---")
    try:
        ok, note = fn()
    except Exception:
        traceback.print_exc()
        ok, note = False, "raised"
    results.append((name, ok, note))
    print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({note})" if note else ""))
    return ok


def t_sources():
    bad = []
    for f in SOURCES:
        p = os.path.join(HERE, f)
        if not os.path.isfile(p):
            bad.append(f"{f} missing")
            continue
        try:
            ast.parse(open(p).read())
        except SyntaxError as e:
            bad.append(f"{f}: {e}")
    for b in bad:
        print("  ", b)
    return not bad, f"{len(SOURCES)} modules"


def t_notebooks():
    import nbformat
    nb_ref = open(os.path.join(HERE, "lpf_nb.py"), "rb").read()
    sh_ref = open(os.path.join(HERE, "pyrosetta_rosetta_scripts.py"), "rb").read()
    ok = bool(NOTEBOOKS)
    for f in NOTEBOOKS:
        nb = nbformat.read(f, as_version=4)
        nbformat.validate(nb)
        bad = []
        for c in nb.cells:
            if c.cell_type == "code":
                try:
                    ast.parse(c.source)
                except SyntaxError as e:
                    bad.append(str(e))
        emb = "n/a"
        for c in nb.cells:
            if "_LPF_NB_B64" in c.source:
                blobs = re.findall(r'=\s*\(\n((?:\s*"[A-Za-z0-9+/=]+"\n)+)\)', c.source)
                got = [zlib.decompress(base64.b64decode(
                    "".join(re.findall(r'"([^"]+)"', b)))) for b in blobs]
                emb = ("identical" if len(got) == 2 and got[0] == nb_ref
                       and got[1] == sh_ref else "MISMATCH")
        ok &= not bad and emb != "MISMATCH"
        print(f"   {os.path.basename(f):28s} {len(nb.cells):3d} cells | "
              f"syntax {len(bad)} errors | embedded {emb}")
    return ok, f"{len(NOTEBOOKS)} notebooks"


def t_cli():
    import lpf_run
    tree = ast.parse(open(os.path.join(HERE, "lpf_run.py")).read())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def attrs(fn, seen=None):
        seen = seen or set()
        if fn in seen or fn not in funcs:
            return set()
        seen.add(fn)
        out, calls = set(), set()
        for node in ast.walk(funcs[fn]):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id == "args" and not node.attr.startswith("_"):
                out.add(node.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.add(node.func.id)
        for c in calls:
            out |= attrs(c, seen)
        return out

    stages = {
        "prep": (["stage_prep"], ["prep", "--project", "/tmp/x", "--target", "a.pdb"]),
        "coarse": (["stage_coarse"], ["coarse", "--project", "/tmp/x"]),
        "traj": (["stage_traj"], ["traj", "--project", "/tmp/x"]),
        "analyze": (["stage_analyze"], ["analyze", "--project", "/tmp/x"]),
        "doctor": (["stage_doctor"], ["doctor"]),
        "cluster": (["stage_cluster"], ["cluster", "/tmp/a", "/tmp/b"]),
        "rescore": (["stage_rescore"], ["rescore", "--project", "/tmp/x"]),
        "relocate": (["stage_relocate"], ["relocate", "--project", "/tmp/x"]),
        "demo": (["stage_demo", "stage_traj", "stage_analyze"],
                 ["demo", "--project", "/tmp/x"]),
        "run": (["stage_prep", "stage_coarse", "stage_traj", "stage_analyze"],
                ["run", "--project", "/tmp/x", "--target", "a.pdb"]),
    }
    ap = lpf_run.build_parser()
    ok = True
    for stage, (fns, argv) in stages.items():
        need = set()
        for f in fns:
            need |= attrs(f)
        with contextlib.redirect_stdout(io.StringIO()):
            ns = ap.parse_args(argv)
        missing = sorted(a for a in need if not hasattr(ns, a))
        helped = subprocess.run([sys.executable, os.path.join(HERE, "lpf_run.py"),
                                 stage, "--help"],
                                capture_output=True).returncode == 0
        ok &= not missing and helped
        print(f"   {stage:8s} {len(need):3d} args | "
              f"{'ok' if not missing else 'MISSING ' + ', '.join(missing)} | "
              f"--help {'ok' if helped else 'FAIL'}")
    return ok, f"{len(stages)} subcommands"


def t_shim():
    proc = subprocess.run([sys.executable,
                           os.path.join(HERE, "pyrosetta_rosetta_scripts.py"),
                           "--self-test"], capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1:] or [""]
    print("   ", tail[0])
    return "self-test: PASS" in proc.stdout, None


def t_roundtrip():
    if not DEMO_START:
        print("    (no LigandPathFinder checkout — skipped. Set LPF_REPO=/path"
          "/to/LigandPathFinder, or clone it beside this repo.)")
        return True, "skipped"
    import lpf_nb as L
    atoms = L.read_pdb_atoms(DEMO_START)
    orig = [l.rstrip("\n") for l in open(DEMO_START)
            if l[:6].strip() in ("ATOM", "HETATM")]
    bad = 0
    for o, a in zip(orig, atoms):
        new = L.format_atom_line(a).rstrip("\n")
        for lo, hi in ((0, 6), (12, 16), (16, 17), (17, 20), (21, 22),
                       (22, 26), (30, 54), (54, 66), (76, 78)):
            if o[lo:hi] != new[lo:hi]:
                bad += 1
                break
    print(f"    {len(orig)} atoms, {bad} field mismatches")
    sys.path.insert(0, os.path.join(REPO, "lpf"))
    from util import PDBHandler
    tmp = "/tmp/_selftest_rt.pdb"
    L.write_pdb_atoms(tmp, atoms)
    import numpy as np
    a1 = PDBHandler.read_window(DEMO_START, "X", "O1")
    a2 = PDBHandler.read_window(tmp, "X", "O1")
    agree = (np.allclose(a1[0], a2[0]) and a1[1] == a2[1] and a1[2] == a2[2]
             and a1[3] == a2[3])
    os.remove(tmp)
    print(f"    LPF's own read_window agrees: {agree}")
    return bad == 0 and agree, f"{len(orig)} atoms"


def t_execute():
    """Run every notebook code cell, with the two Rosetta calls stubbed."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    if not DEMO_START:
        print("    (no LigandPathFinder checkout — skipped. Set LPF_REPO=/path"
          "/to/LigandPathFinder, or clone it beside this repo.)")
        return True, "skipped"

    pep_target = "/tmp/_selftest_1cka.pdb"
    if not os.path.isfile(pep_target):
        try:
            import urllib.request
            urllib.request.urlretrieve(
                "https://files.rcsb.org/download/1CKA.pdb", pep_target)
        except Exception as exc:
            print(f"    (could not fetch 1CKA for the peptide notebooks: {exc})")
            pep_target = None

    stub = "/tmp/_selftest_rosetta_scripts.static.linuxgccrelease"
    with open(stub, "w") as fh:
        fh.write('#!/bin/sh\necho "core.init: Rosetta version stub"\n')
    os.chmod(stub, 0o755)

    def fake_coarse(run_dir, guide_path, n=16):
        p = np.loadtxt(guide_path)
        t = np.linspace(0, 1, n)[:, None]
        perp = np.cross(p[-1] - p[0], [0, 0, 1.0])
        perp /= np.linalg.norm(perp)
        pts = p[0] + (p[-1] - p[0]) * t + perp * (2.5 * np.sin(np.pi * t))
        np.savetxt(os.path.join(run_dir, "coarsepath.txt"), pts)
        for f in ("trajectory.pdb", "lig_trajectory.pdb", "full_trajectory.pdb"):
            open(os.path.join(run_dir, f), "w").write(
                "HETATM    1  C   GP  A   1       0.000   0.000   0.000"
                "  1.00  0.00           C\nENDMDL\n")
        os.makedirs(os.path.join(run_dir, "0"), exist_ok=True)

    def fake_traj(run_dir, guide_path, start_pdb, lig_chain, step):
        pts = np.loadtxt(guide_path)
        s = np.concatenate([[0], np.cumsum(
            np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        nw = max(5, int(s[-1] / step))
        u = np.linspace(0, s[-1], nw)
        gp = np.column_stack([np.interp(u, s, pts[:, k]) for k in range(3)])
        for n, a in (("guide_path", gp), ("input_path", pts),
                     ("nbr_trajectory.txt", gp)):
            np.savetxt(os.path.join(run_dir, n), a)
        x = np.linspace(0, 1, nw)
        y = -8 * np.exp(-4 * x) + 0.6 * np.sin(9 * x)
        np.savetxt(os.path.join(run_dir, "interface_scores.txt"), y)
        np.savetxt(os.path.join(run_dir, "total_scores.txt"), y * 40 - 900)
        lines = [l for l in open(start_pdb)
                 if l[:6].strip() in ("ATOM", "HETATM")]
        with open(os.path.join(run_dir, "full_trajectory.pdb"), "w") as ft, \
                open(os.path.join(run_dir, "lig_trajectory.pdb"), "w") as lt:
            for i in range(nw):
                d = gp[i] - gp[0]
                for l in lines:
                    if l[21] == lig_chain:
                        c = np.array([float(l[30:38]), float(l[38:46]),
                                      float(l[46:54])]) + d
                        l = l[:30] + f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}" + l[54:]
                        lt.write(l)
                    ft.write(l)
                ft.write("ENDMDL\n")
                lt.write("ENDMDL\n")
                w = os.path.join(run_dir, f"window_{i:0{len(str(nw))}d}")
                os.makedirs(w, exist_ok=True)
                shutil.copy(os.path.join(run_dir, "lig_trajectory.pdb"),
                            os.path.join(w, "best.pdb"))
        json.dump({"version": 1, "completed": True,
                   "rootdir": os.path.abspath(run_dir)},
                  open(os.path.join(run_dir, "checkpoint_meta.json"), "w"))

    base = {"ROSETTA_BACKEND": "binary", "INSTALL_PYROSETTA": False,
            "ROSETTA_SCRIPTS": stub, "ROSETTA_DATABASE": "", "LPF_REPO": REPO,
            "SEC_PER_DECOY_GUESS": 4.0}
    ok = True
    cwd = os.getcwd()
    for f in NOTEBOOKS:
        name = os.path.basename(f)
        is_pep = "Peptide" in name
        if is_pep and not pep_target:
            print(f"   {name:28s} skipped (no 1CKA)")
            continue
        ov = dict(base, PROJECT_DIR=f"/tmp/_selftest_proj_{'pep' if is_pep else 'lig'}")
        if is_pep:
            ov.update(PROTEIN_SOURCE=pep_target, PEPTIDE_CHAIN_IN_SOURCE="B",
                      KEEP_CHAINS="A", N_LOCAL=20, N_MEDIUM=20, N_GLOBAL=8,
                      RANDOM_SEED=5, N_REPLICATES=1, COARSE_STEP=3.0, TRAJ_STEP=1.5)
        else:
            ov.update(USE_SHIPPED_DEMO=True, N_REPLICATES=2, COARSE_STEP=2.0,
                      TRAJ_STEP=0.8)
        wd = f"/tmp/_selftest_nb_{name.replace('.ipynb', '')}"
        shutil.rmtree(wd, ignore_errors=True)
        os.makedirs(wd)
        nb = json.load(open(f))
        if "_LPF_NB_B64" not in json.dumps(nb):
            for mod in ("lpf_nb.py", "pyrosetta_rosetta_scripts.py"):
                shutil.copy(os.path.join(HERE, mod), wd)
            # mimic the repo layout: notebook in a subdirectory, modules above it
            os.makedirs(os.path.join(wd, "notebooks"), exist_ok=True)
        os.chdir(wd)
        ns = {"__name__": "__main__", "_FAKE_COARSE": fake_coarse,
              "_FAKE_TRAJ": fake_traj}
        ran = skipped = 0
        failed = None
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            src = "".join(c["source"])
            if '-m", "pip' in src or "from google.colab" in src:
                skipped += 1
                continue
            for k, v in ov.items():
                src = re.sub(rf"^{k} = .*$", f"{k} = {v!r}", src, flags=re.M)
            src = src.replace(
                'L.run_lpf_stage("coarse", LPF_REPO, cfg_path, run_dir,\n'
                '                        log_path=os.path.join(run_dir, "coarse.log"))',
                "_FAKE_COARSE(run_dir, GUIDE_PATH)")
            src = re.sub(r'L\.run_lpf_stage\("trajectory".*?\)\)',
                         "_FAKE_TRAJ(TRAJ_RUN, TRAJ_PATH, START_PDB, LIGAND_CHAIN, "
                         "TRAJ_STEP)", src, flags=re.S)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(src, f"<{name} cell {i}>", "exec"), ns)
                ran += 1
            except Exception as exc:
                failed = f"cell {i}: {type(exc).__name__}: {exc}"
                break
        os.chdir(cwd)
        shutil.rmtree(wd, ignore_errors=True)
        shutil.rmtree(ov["PROJECT_DIR"], ignore_errors=True)
        ok &= failed is None
        print(f"   {name:28s} {ran:3d} cells run, {skipped} skipped"
              + (f" | FAILED {failed}" if failed else ""))
    os.remove(stub)
    return ok, f"{len(NOTEBOOKS)} notebooks"


def t_tutorial():
    """Every flag in TUTORIAL.md exists, every parser option is documented, and
    every runnable command in it parses."""
    import argparse, contextlib, io, re, shlex
    md_path = os.path.join(HERE, "TUTORIAL.md")
    if not os.path.isfile(md_path):
        print("    (TUTORIAL.md not built — run make_tutorial.py)")
        return True, "skipped"
    import lpf_run
    md = open(md_path).read()
    parser = lpf_run.build_parser()
    sub = next(a for a in parser._actions
               if isinstance(a, argparse._SubParsersAction))
    real = {s for p in list(sub.choices.values()) + [parser]
            for a in p._actions for s in a.option_strings}
    shim = {"--probe", "--options", "--self-test", "--write-wrapper",
            "--python", "--version"}
    doc = ((set(re.findall(r"`(--[a-z0-9-]+)`", md))
            | set(re.findall(r"(?<![\w-])(--[a-z0-9-]+)(?=[\s\\])", md)))
           - {"---"})
    ghost = sorted(f for f in doc if f not in real and f not in shim)
    allreal = {s for p in sub.choices.values() for a in p._actions
               for s in a.option_strings if s.startswith("--") and s != "--help"}
    undoc = sorted(f for f in allreal if f not in doc)

    cmds, bad = [], []
    for b in re.findall(r"```bash\n(.*?)```", md, re.S):
        b = re.sub(r"\\\n\s*", " ", b)
        for line in b.split("\n"):
            line = re.sub(r"\s*#.*$", "", line.strip())
            if line.startswith("python lpf_run.py"):
                cmds.append(line)
    need = {"prep": ["--target", "x.pdb"], "run": ["--target", "x.pdb"]}
    for c in cmds:
        argv = shlex.split(c)[2:]
        if not argv:
            continue
        extra = []
        if "--project" not in argv and argv[0] != "doctor":
            extra += ["--project", "/tmp/x"]
        for i in range(0, len(need.get(argv[0], [])), 2):
            if need[argv[0]][i] not in argv:
                extra += need[argv[0]][i:i + 2]
        try:
            with contextlib.redirect_stderr(io.StringIO()), \
                    contextlib.redirect_stdout(io.StringIO()):
                parser.parse_args(argv + extra)
        except SystemExit:
            bad.append(c[:70])
    print(f"    {len(doc)} flags documented, {len(cmds)} runnable commands")
    print(f"    flags not in the parser: {ghost or 'none'}")
    print(f"    parser options undocumented: {undoc or 'none'}")
    print(f"    commands that fail to parse: {bad or 'none'}")
    return not (ghost or undoc or bad), f"{len(cmds)} commands"


GROUPS = {"sources": t_sources, "notebooks": t_notebooks, "cli": t_cli,
          "shim": t_shim, "roundtrip": t_roundtrip, "execute": t_execute,
          "tutorial": t_tutorial}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(GROUPS)
    unknown = [w for w in wanted if w not in GROUPS]
    if unknown:
        sys.exit(f"unknown group(s) {unknown}; choose from {list(GROUPS)}")
    for w in wanted:
        check(w, GROUPS[w])
    print(f"\n{'=' * 60}")
    for name, ok, note in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({note})" if note else ""))
    failed = [n for n, ok, _ in results if not ok]
    print(f"{'=' * 60}")
    print("ALL PASS" if not failed else f"FAILED: {', '.join(failed)}")
    sys.exit(1 if failed else 0)
