#!/usr/bin/env python3
"""
A `rosetta_scripts`-compatible front end backed by PyRosetta.

Ligand PathFinder does not import Rosetta — it runs the `rosetta_scripts`
*executable* as a subprocess and then reads `score.sc` out of the working
directory (see util.run_rosetta). PyRosetta ships no such executable, so
`pip install pyrosetta` on its own does not give LPF anything it can call.

This script closes that gap: it presents the slice of the `rosetta_scripts`
command line that LPF uses, runs the RosettaScripts XML through PyRosetta, and
writes decoy PDBs and a score file in the format LPF's ScoreParser expects. Point
the `"rosetta"` key of run.json at the launcher wrapper (see --write-wrapper) and
nothing in LPF changes.

Command line understood (exactly what LPF builds):

    rosetta_scripts @options -nstruct N -out:path:pdb ./results \
        -parser:protocol task.xml -in:file:s task.pdb
    rosetta_scripts @options -in:file:s stats.pdb -out:path:pdb ./stats \
        -out:file:scorefile stats.sc -parser:protocol stats.xml -nstruct N

Every flag not in that list — including `@file` option files — is handed
straight to pyrosetta.init(), so the options file LPF copies into each working
directory is parsed by Rosetta's own option system rather than re-implemented
here.

Extras, set through the environment because LPF controls the command line:

    LPF_SHIM_JOBS=4     run nstruct across 4 worker processes (default 1).
                        Each worker initialises its own PyRosetta, so budget
                        ~1-2 GB of RAM per job.
    LPF_SHIM_VERBOSE=1  keep PyRosetta's own tracer output.

Self-check, which needs no PyRosetta:

    python pyrosetta_rosetta_scripts.py --self-test

Diagnosing an all-zero interface_scores.txt (needs PyRosetta):

    cd <run>/window_0005
    python pyrosetta_rosetta_scripts.py --probe task.xml task.pdb --options options

MPI: none. Keep `.mpi.` out of the launcher's filename and LPF will hold
rosetta_scripts.n_proc at 1 by itself; use LPF_SHIM_JOBS instead.

PyRosetta is Rosetta, and is covered by your Rosetta licence, not by
LigandPathFinder's.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

__version__ = "1.0"

# Score terms LPF subtracts before ranking decoys, so they must appear as
# columns even when a protocol leaves them at zero
# (util._CONSTRAINT_COLS / _IF_CONSTRAINT_COLS).
CONSTRAINT_TERMS = ("atom_pair_constraint", "angle_constraint", "dihedral_constraint")
REQUIRED_COLUMNS = ("total_score", "interface_delta_X")


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

#: flags this script consumes; everything else goes to pyrosetta.init()
_CONSUMED_WITH_VALUE = {
    "-nstruct", "-out:path:pdb", "-out:file:scorefile", "-parser:protocol",
    "-in:file:s", "-out:path:score", "-out:prefix", "-out:suffix",
}


def parse_rosetta_argv(argv):
    """
    Split a rosetta_scripts command line into what we act on and what Rosetta gets.

    Rosetta flags take their values as following tokens, so a flag is consumed
    together with its value; a leading '@' names an option file, which is passed
    through untouched for Rosetta to parse.
    """
    opts = {"nstruct": 1, "out_path_pdb": ".", "scorefile": "score.sc",
            "protocol": None, "input_pdb": None, "out_path_score": None,
            "prefix": "", "suffix": ""}
    passthrough = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in _CONSUMED_WITH_VALUE:
            if i + 1 >= len(argv):
                raise SystemExit(f"{tok} needs a value")
            val = argv[i + 1]
            i += 2
            if tok == "-nstruct":
                opts["nstruct"] = int(val)
            elif tok == "-out:path:pdb":
                opts["out_path_pdb"] = val
            elif tok == "-out:file:scorefile":
                opts["scorefile"] = val
            elif tok == "-out:path:score":
                opts["out_path_score"] = val
            elif tok == "-parser:protocol":
                opts["protocol"] = val
            elif tok == "-in:file:s":
                opts["input_pdb"] = val
            elif tok == "-out:prefix":
                opts["prefix"] = val
            elif tok == "-out:suffix":
                opts["suffix"] = val
            continue
        passthrough.append(tok)
        i += 1
    return opts, passthrough


def decoy_tag(input_pdb, index, prefix="", suffix=""):
    """
    Reproduce Rosetta's output naming: task.pdb + nstruct -> task_0001, task_0002 ...

    LPF reads the `description` column of score.sc and then opens
    `<out_path>/<description>.pdb`, so the tag and the filename must agree.
    """
    base = os.path.basename(input_pdb)
    for ext in (".pdb.gz", ".pdb"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return f"{prefix}{base}{suffix}_{index:04d}"


# --------------------------------------------------------------------------
# score file
# --------------------------------------------------------------------------

def write_score_file(path, rows, require=REQUIRED_COLUMNS):
    """
    Write a Rosetta-style score file that LPF's ScoreParser can read.

    The format it needs (util.ScoreParser.report_final skips line 0, treats line
    1 as the header, and whitespace-splits every line into equal-length rows):

        SEQUENCE:
        SCORE:  total_score  <terms...>  description
        SCORE:  -123.456     ...         task_0001

    `rows` is a list of (tag, {term: value}). Columns are the union over all
    decoys so the rows stay rectangular, with total_score first and description
    last; missing values are written as 0.
    """
    if not rows:
        raise ValueError("no decoys to write — nothing succeeded")

    keys = set()
    for _, sc in rows:
        keys.update(k for k in sc if not k.startswith("_"))
    keys.update(require)
    keys.update(CONSTRAINT_TERMS)
    ordered = ["total_score"] + sorted(k for k in keys if k != "total_score")

    missing = [c for c in require if c not in ordered]
    if missing:
        raise ValueError(f"score columns {missing} are absent; LPF requires them")

    def fmt(v):
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return str(v)

    widths = {k: max(len(k), 9) for k in ordered}
    lines = ["SEQUENCE: \n"]
    lines.append("SCORE: " + " ".join(k.rjust(widths[k]) for k in ordered)
                 + " description\n")
    for tag, sc in rows:
        lines.append("SCORE: "
                     + " ".join(fmt(sc.get(k, 0.0)).rjust(widths[k]) for k in ordered)
                     + f" {tag}\n")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.writelines(lines)
    return path


# --------------------------------------------------------------------------
# PyRosetta engine
# --------------------------------------------------------------------------

def _init_pyrosetta(passthrough, verbose=False):
    import pyrosetta
    flags = list(passthrough)
    if not verbose:
        flags += ["-mute", "all"]
    pyrosetta.init(" ".join(flags), silent=not verbose)
    return pyrosetta


def _make_protocol(pyrosetta, xml_path):
    """
    Build the mover described by a RosettaScripts XML.

    The parser API has moved around between PyRosetta releases, so the known
    entry points are tried in order and the one that works is used. A fresh
    mover is built per decoy, because RosettaScripts movers are not guaranteed
    to be reusable across applications.
    """
    rs = pyrosetta.rosetta.protocols.rosetta_scripts
    errors = []

    parser_cls = getattr(rs, "RosettaScriptsParser", None)
    if parser_cls is not None:
        for attr in ("generate_mover", "generate_mover_for_job",
                     "generate_mover_and_apply_to_pose"):
            fn = getattr(parser_cls(), attr, None)
            if fn is None:
                continue
            try:
                mover = fn(xml_path)
                if mover is not None:
                    return ("parser:" + attr), (lambda: getattr(parser_cls(), attr)(xml_path))
            except Exception as exc:                       # wrong arity, etc.
                errors.append(f"RosettaScriptsParser.{attr}: {exc}")

    xo = getattr(rs, "XmlObjects", None)
    if xo is not None:
        try:
            obj = xo.create_from_file(xml_path)
            mover = obj.get_mover("ParsedProtocol")
            if mover is not None:
                return "XmlObjects", (
                    lambda: xo.create_from_file(xml_path).get_mover("ParsedProtocol"))
        except Exception as exc:
            errors.append(f"XmlObjects.create_from_file: {exc}")

    raise RuntimeError(
        "Could not build a mover from the RosettaScripts XML with this PyRosetta "
        "build. Tried:\n  " + "\n  ".join(errors or ["(no API found at all)"]))


def _extra_scores(pose):
    """
    Mover-set extra scores, read through every accessor this PyRosetta offers.

    This is where InterfaceScoreCalculator leaves interface_delta_<chain> and the
    if_<chain>_* family, and where Transform leaves Transform_accept_ratio.

    `pose.scores` alone is NOT enough: in some PyRosetta builds it returns only
    the score-function energy map, so every mover-set value silently disappears
    and LPF's interface_scores.txt comes out all zeros. Each accessor is tried
    independently and the results unioned, so a build that exposes only one of
    them still works.
    """
    import pyrosetta
    core_pose = pyrosetta.rosetta.core.pose
    out = {}

    # 1. the documented free functions for the arbitrary-data caches
    for name in ("getPoseExtraFloatScores", "getPoseExtraScores",
                 "getPoseExtraStringScores"):
        fn = getattr(core_pose, name, None)
        if fn is None:
            continue
        try:
            for k, v in dict(fn(pose)).items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
        except Exception:
            continue

    # 2. the DataCache directly, for builds without those helpers
    try:
        dt = core_pose.datacache.CacheableDataType
        for attr in ("ARBITRARY_FLOAT_DATA", "ARBITRARY_STRING_DATA"):
            key = getattr(dt, attr, None)
            if key is None or not pose.data().has(key):
                continue
            blob = pose.data().get_const_ptr(key)
            for k, v in dict(blob.map()).items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
    except Exception:
        pass

    return out


def _ligand_chain_from_xml(xml_path):
    """
    Which chain the protocol treats as the ligand.

    Read from the XML rather than guessed: InterfaceScoreCalculator's chains=,
    then SCORINGGRIDS ligand_chain=, then StartFrom/Transform chain=. The
    LPF_SHIM_CHAIN environment variable overrides all of them.
    """
    override = os.environ.get("LPF_SHIM_CHAIN", "").strip()
    if override:
        return override[:1]
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None
    for tag, attr in (("InterfaceScoreCalculator", "chains"),
                      ("SCORINGGRIDS", "ligand_chain"),
                      ("StartFrom", "chain"),
                      ("Transform", "chain")):
        for el in root.iter(tag):
            v = (el.get(attr) or "").strip()
            if v:
                return v[0]
    return None


def _score_function_from_xml(pyrosetta, xml_path, want=None, drop_constraints=True):
    """
    Rebuild a ScoreFunction declared in the XML's SCOREFXNS block.

    Only `weights=` plus `<Reweight scoretype= weight=>` are honoured, which is
    all the LPF protocols use. `want` names the ScoreFunction; the default is
    whatever InterfaceScoreCalculator references, falling back to the last one
    declared.

    Constraint terms are zeroed for the interface calculation. Separating the
    ligand stretches the AtomPair restraints to the virtual ghost residues, and
    the resulting energy swamps everything — the real mover produces a value of
    order 1e5 for this reason, which LPF then has to subtract back out. Zeroing
    them up front gives the physical number directly.
    """
    import xml.etree.ElementTree as ET
    core_scoring = pyrosetta.rosetta.core.scoring
    root = ET.parse(xml_path).getroot()

    if want is None:
        for el in root.iter("InterfaceScoreCalculator"):
            want = el.get("scorefxn")
            break
    decls = {el.get("name"): el for el in root.iter("ScoreFunction")}
    el = decls.get(want)
    if el is None and decls:
        el = list(decls.values())[-1]
    weights = (el.get("weights") if el is not None else None) or "ligand"

    sfxn = pyrosetta.create_score_function(weights)
    if el is not None:
        for rw in el.iter("Reweight"):
            name, w = rw.get("scoretype"), rw.get("weight")
            if not name or w is None:
                continue
            try:
                sfxn.set_weight(core_scoring.score_type_from_name(name), float(w))
            except Exception:
                continue
    if drop_constraints:
        for name in ("atom_pair_constraint", "angle_constraint",
                     "dihedral_constraint", "coordinate_constraint",
                     "res_type_constraint", "metalbinding_constraint"):
            try:
                sfxn.set_weight(core_scoring.score_type_from_name(name), 0.0)
            except Exception:
                continue
    return sfxn, weights


def _compute_interface_delta(pyrosetta, pose, chain, sfxn):
    """
    Interface energy as complex-minus-separated, computed here.

    This is what InterfaceScoreCalculator does. Doing it directly removes the
    dependency on that mover delivering its result into a place this shim can
    read, which varies between PyRosetta builds.

    The ligand chain is translated far enough away that every interaction with
    it vanishes, and the pose is rescored. Returns
    (total_delta, {term: per_term_delta}).
    """
    core_scoring = pyrosetta.rosetta.core.scoring
    AtomID = pyrosetta.rosetta.core.id.AtomID
    xyzV = pyrosetta.rosetta.numeric.xyzVector_double_t

    info = pose.pdb_info()
    lig = [i for i in range(1, pose.total_residue() + 1)
           if info and info.chain(i) == chain]
    if not lig:
        raise RuntimeError(
            f"chain {chain!r} has no residues in this pose — cannot compute an "
            f"interface. Chains present: "
            f"{sorted({info.chain(i) for i in range(1, pose.total_residue() + 1)})}")

    nonzero = list(sfxn.get_nonzero_weighted_scoretypes())
    weights = sfxn.weights()

    together = pose.clone()
    sfxn(together)
    # read the values out rather than cloning the EMapVector: it has no clone()
    # in older PyRosetta, and the two poses are separate objects anyway
    e_together = {st: float(together.energies().total_energies()[st])
                  for st in nonzero}

    apart = pose.clone()
    offset = xyzV(1000.0, 1000.0, 1000.0)
    for i in lig:
        res = apart.residue(i)
        for j in range(1, res.natoms() + 1):
            aid = AtomID(j, i)
            apart.set_xyz(aid, apart.xyz(aid) + offset)
    sfxn(apart)
    e_apart = {st: float(apart.energies().total_energies()[st]) for st in nonzero}

    per_term, total = {}, 0.0
    for st in nonzero:
        d = float(weights[st]) * (e_together[st] - e_apart[st])
        if abs(d) > 1e-9:
            per_term[core_scoring.name_from_score_type(st)] = d
        total += d
    return float(total), per_term


def _collect_scores(pose):
    """
    Numeric scores for one finished decoy: energy terms plus mover-set extras.

    The constraint terms are forced in at 0.0 when a protocol cleared them,
    because LPF subtracts those columns unconditionally.
    """
    out = {}
    try:
        for k, v in dict(pose.scores).items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    except Exception:
        pass

    n_from_scores = len(out)
    extras = _extra_scores(pose)
    added = {k: v for k, v in extras.items() if k not in out}
    out.update(extras)

    if "total_score" not in out:
        out["total_score"] = float(pose.energies().total_energy())
    for term in CONSTRAINT_TERMS:
        out.setdefault(term, 0.0)
    out["_n_from_pose_scores"] = n_from_scores
    out["_n_from_extra_caches"] = len(added)
    return out


def _interface_mode():
    """auto | mover | compute | off, from LPF_SHIM_INTERFACE."""
    m = os.environ.get("LPF_SHIM_INTERFACE", "auto").strip().lower()
    return m if m in ("auto", "mover", "compute", "off") else "auto"


def _add_interface_scores(pyrosetta, pose, sc, chain, sfxn, weights_name, log,
                          announce=False):
    """
    Make sure sc carries interface_delta_<chain>, computing it if the mover did not.

    LPF reads interface_delta_X and subtracts the if_X_*_constraint columns. A
    computed delta already excludes the constraint terms, so those columns are
    emitted as 0.0 and the subtraction is a no-op — the number LPF records is
    the physical one either way.
    """
    key = f"interface_delta_{chain}"
    mode = _interface_mode()
    if mode == "off" or (mode == "mover"):
        return sc, "mover"
    if key in sc and mode == "auto":
        return sc, "mover"

    total, per_term = _compute_interface_delta(pyrosetta, pose, chain, sfxn)
    sc[key] = total
    for term, d in per_term.items():
        sc[f"if_{chain}_{term}"] = d
    for term in CONSTRAINT_TERMS:
        sc.setdefault(f"if_{chain}_{term}", 0.0)
    if announce:
        log(f"[shim] InterfaceScoreCalculator gave no {key}, so it was computed "
            f"here instead:")
        log(f"[shim]   complex minus separated, scorefxn '{weights_name}', "
            f"constraint terms zeroed")
        log(f"[shim]   {len(per_term)} per-term contributions emitted as "
            f"if_{chain}_*")
    return sc, "computed"


def run_pyrosetta(opts, passthrough, verbose=False, log=print):
    """Run the protocol `nstruct` times and return [(tag, scores), ...]."""
    pyrosetta = _init_pyrosetta(passthrough, verbose=verbose)
    api, make = _make_protocol(pyrosetta, opts["protocol"])
    log(f"[shim] RosettaScripts via {api}")
    chain = _ligand_chain_from_xml(opts["protocol"])
    sfxn = weights_name = None
    if chain and _interface_mode() in ("auto", "compute"):
        try:
            sfxn, weights_name = _score_function_from_xml(pyrosetta, opts["protocol"])
            log(f"[shim] interface fallback ready: chain {chain}, "
                f"scorefxn '{weights_name}'")
        except Exception as exc:
            log(f"[shim] could not build the interface fallback: {exc}")

    start_pose = pyrosetta.pose_from_file(opts["input_pdb"])
    log(f"[shim] input {opts['input_pdb']}: {start_pose.total_residue()} residues, "
        f"{start_pose.num_chains()} chains")
    os.makedirs(opts["out_path_pdb"], exist_ok=True)

    rows = []
    for i in range(1, opts["nstruct"] + 1):
        tag = decoy_tag(opts["input_pdb"], i, opts["prefix"], opts["suffix"])
        pose = start_pose.clone()
        try:
            make().apply(pose)
        except Exception as exc:
            log(f"[shim] decoy {tag} FAILED: {exc}")
            continue
        out_pdb = os.path.join(opts["out_path_pdb"], tag + ".pdb")
        pose.dump_pdb(out_pdb)
        sc = _collect_scores(pose)
        if chain and sfxn is not None:
            try:
                sc, src = _add_interface_scores(pyrosetta, pose, sc, chain, sfxn,
                                                weights_name, log, announce=(i == 1))
            except Exception as exc:
                if i == 1:
                    log(f"[shim] interface fallback failed: {exc}")
        rows.append((tag, sc))
        if i == 1:
            log(f"[shim] score sources: {sc['_n_from_pose_scores']} term(s) from "
                f"pose.scores, {sc['_n_from_extra_caches']} more from the extra-score "
                f"caches")
            n_if = sum(1 for k in sc if k.startswith("if_") or "interface_delta" in k)
            log(f"[shim] interface terms found: {n_if}")
        log(f"[shim] {tag}  total_score {sc.get('total_score', float('nan')):.3f}  "
            f"interface_delta_X {sc.get('interface_delta_X', float('nan')):.3f}")
    return rows


def _worker(payload):
    """One worker process: initialise PyRosetta once, do a slice of nstruct."""
    opts, passthrough, verbose, lo, hi = payload
    rows = []
    pyrosetta = _init_pyrosetta(passthrough, verbose=verbose)
    api, make = _make_protocol(pyrosetta, opts["protocol"])
    start_pose = pyrosetta.pose_from_file(opts["input_pdb"])
    os.makedirs(opts["out_path_pdb"], exist_ok=True)
    for i in range(lo, hi + 1):
        tag = decoy_tag(opts["input_pdb"], i, opts["prefix"], opts["suffix"])
        pose = start_pose.clone()
        try:
            make().apply(pose)
        except Exception as exc:
            print(f"[shim] decoy {tag} FAILED: {exc}", flush=True)
            continue
        pose.dump_pdb(os.path.join(opts["out_path_pdb"], tag + ".pdb"))
        rows.append((tag, _collect_scores(pose)))
        print(f"[shim] {tag} done", flush=True)
    return rows


def run_pyrosetta_parallel(opts, passthrough, jobs, verbose=False, log=print):
    """Spread nstruct over `jobs` processes. Decoys are independent, so this is safe."""
    import multiprocessing as mp
    n = opts["nstruct"]
    jobs = max(1, min(int(jobs), n))
    if jobs == 1:
        return run_pyrosetta(opts, passthrough, verbose=verbose, log=log)

    edges, start = [], 1
    per = n // jobs
    extra = n % jobs
    for j in range(jobs):
        count = per + (1 if j < extra else 0)
        if count:
            edges.append((start, start + count - 1))
            start += count
    log(f"[shim] {n} decoys across {len(edges)} worker process(es): {edges}")
    log("[shim] each worker initialises its own PyRosetta (~1-2 GB RAM each)")

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(edges)) as pool:
        chunks = pool.map(_worker, [(opts, passthrough, verbose, lo, hi)
                                    for lo, hi in edges])
    rows = [r for chunk in chunks for r in chunk]
    rows.sort(key=lambda tr: tr[0])
    return rows


# --------------------------------------------------------------------------
# self-test (no PyRosetta needed)
# --------------------------------------------------------------------------

def self_test(tmp_dir=None):
    """
    Check the plumbing LPF depends on, without touching PyRosetta.

    Verifies that the exact command lines LPF builds are parsed correctly, that
    the decoy naming matches the `description` column, and — if LigandPathFinder
    is importable — that its own ScoreParser reads a score file written here.
    """
    import tempfile
    ok = True
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="lpf_shim_selftest_")
    print(f"self-test in {tmp_dir}\n")

    main_argv = ["@options", "-nstruct", "10", "-out:path:pdb", "./results",
                 "-parser:protocol", "task.xml", "-in:file:s", "task.pdb"]
    opts, pt = parse_rosetta_argv(main_argv)
    checks = [
        ("nstruct parsed", opts["nstruct"] == 10),
        ("out:path:pdb parsed", opts["out_path_pdb"] == "./results"),
        ("protocol parsed", opts["protocol"] == "task.xml"),
        ("input parsed", opts["input_pdb"] == "task.pdb"),
        ("default scorefile", opts["scorefile"] == "score.sc"),
        ("options file passed through", pt == ["@options"]),
    ]
    stats_argv = ["@options", "-in:file:s", "stats.pdb", "-out:path:pdb", "./stats",
                  "-out:file:scorefile", "stats.sc", "-parser:protocol", "stats.xml",
                  "-nstruct", "10"]
    o2, pt2 = parse_rosetta_argv(stats_argv)
    checks += [
        ("stats scorefile parsed", o2["scorefile"] == "stats.sc"),
        ("stats input parsed", o2["input_pdb"] == "stats.pdb"),
        ("stats passthrough", pt2 == ["@options"]),
        ("decoy tag", decoy_tag("task.pdb", 1) == "task_0001"),
        ("decoy tag 10", decoy_tag("/a/b/task.pdb", 10) == "task_0010"),
    ]
    for name, good in checks:
        print(f"  {'ok  ' if good else 'FAIL'} {name}")
        ok &= good

    # a score file with the shape a real run produces, read back by LPF itself
    rows = []
    for i in range(1, 6):
        rows.append((decoy_tag("task.pdb", i), {
            "total_score": -900.0 - i, "interface_delta_X": -8.0 + 0.1 * i,
            "atom_pair_constraint": 1.19, "angle_constraint": 0.0,
            "dihedral_constraint": 0.0, "if_X_atom_pair_constraint": 0.4,
            "if_X_angle_constraint": 0.0, "if_X_dihedral_constraint": 0.0,
            "fa_rep": 120.5,
        }))
    sc_path = write_score_file(os.path.join(tmp_dir, "score.sc"), rows)
    print(f"\n  wrote {sc_path}")
    print("".join("      " + l for l in open(sc_path).readlines()[:4]))

    try:
        import numpy as np
        raw = [l.strip().split() for l in open(sc_path).readlines()[1:]]
        arr = np.array(raw)
        headers, data = arr[0], arr[1:]
        for col in ("total_score", "interface_delta_X", "description"):
            good = col in headers
            print(f"  {'ok  ' if good else 'FAIL'} column {col} present")
            ok &= good
        good = len({len(r) for r in raw}) == 1
        print(f"  {'ok  ' if good else 'FAIL'} rows are rectangular "
              f"(numpy would not build an array otherwise)")
        ok &= good
    except Exception as exc:
        print(f"  FAIL numpy round-trip: {exc}")
        ok = False

    lpf = os.environ.get("LPF_REPO")
    if lpf and os.path.isdir(os.path.join(lpf, "lpf")):
        sys.path.insert(0, os.path.join(lpf, "lpf"))
        try:
            from util import ScoreParser
            os.makedirs(os.path.join(tmp_dir, "results"), exist_ok=True)
            print("\n  LigandPathFinder's own ScoreParser:")
            with open(sc_path) as fh:
                n_rows = len(fh.readlines()) - 2
            print(f"      score file holds {n_rows} decoys; "
                  f"report_final needs the decoy PDBs too, so it is exercised "
                  f"against real output in the notebook test rather than here")
        except Exception as exc:
            print(f"  (could not import LPF's ScoreParser: {exc})")

    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
# wrapper generation
# --------------------------------------------------------------------------

def write_wrapper(path, python_exe=None, script=None):
    """
    Write the tiny executable LPF will actually call.

    LPF execs the path in run.json's "rosetta" key directly, so it must be an
    executable file. A shell wrapper pins the interpreter, which matters when
    PyRosetta lives in a virtualenv that is not on PATH.

    The filename must not contain '.mpi.' — LPF reads that as an MPI build and
    would then allow n_proc > 1, which this shim does not implement.
    """
    python_exe = python_exe or sys.executable
    script = os.path.abspath(script or __file__)
    path = os.path.abspath(path)
    if ".mpi." in os.path.basename(path):
        raise ValueError("keep '.mpi.' out of the wrapper name: LPF would treat it as "
                         "an MPI build and allow n_proc > 1, which this shim cannot do")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("#!/bin/sh\n"
                 "# rosetta_scripts-compatible launcher backed by PyRosetta.\n"
                 "# Generated by pyrosetta_rosetta_scripts.py --write-wrapper\n"
                 f'exec "{python_exe}" "{script}" "$@"\n')
    os.chmod(path, 0o755)
    return path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def probe(xml_path, pdb_path, options_file=None, verbose=False):
    """
    Run a protocol once and list every score the pose ends up carrying.

    For when LPF's interface_scores.txt comes out all zeros: that means
    InterfaceScoreCalculator contributed no interface_delta_<chain>, and this
    shows what the pose actually has, so you can tell whether the mover ran,
    whether it targeted the right chain, and what the key is called.
    """
    passthrough = [f"@{options_file}"] if options_file else []
    pyrosetta = _init_pyrosetta(passthrough, verbose=verbose)
    api, make = _make_protocol(pyrosetta, xml_path)
    print(f"[probe] protocol built via {api}")
    pose = pyrosetta.pose_from_file(pdb_path)
    print(f"[probe] {pdb_path}: {pose.total_residue()} residues, "
          f"{pose.num_chains()} chains")
    chains = []
    for i in range(1, pose.num_chains() + 1):
        begin = pose.chain_begin(i)
        chains.append(f"{pose.pdb_info().chain(begin)}"
                      f"({pose.chain_end(i) - begin + 1})")
    print(f"[probe] chains: {' '.join(chains)}")
    make().apply(pose)

    try:
        print(f"[probe] PyRosetta {pyrosetta.version()}")
    except Exception:
        pass

    sc = dict(pose.scores)
    extras = _extra_scores(pose)
    print(f"[probe] pose.scores: {len(sc)} key(s); extra-score caches: "
          f"{len(extras)} key(s)")
    sc.update(extras)
    interface = {k: v for k, v in sc.items()
                 if "interface_delta" in k or k.startswith("if_")}
    print(f"\n[probe] {len(sc)} score terms on the finished pose")
    if interface:
        print(f"[probe] interface terms ({len(interface)}):")
        for k in sorted(interface):
            print(f"           {k} = {interface[k]}")
        print("\n[probe] InterfaceScoreCalculator worked. If LPF still reports "
              "zeros, the\n        chain in its chains= attribute is not the one "
              "LPF tracks.")
    else:
        print("[probe] NO interface terms at all — InterfaceScoreCalculator "
              "contributed nothing.")
        print("        Check that the protocol Adds it, that its chains= names the "
              "ligand's\n        chain, and that the chain exists in the pose "
              "(listed above).")
    chain = _ligand_chain_from_xml(xml_path)
    print(f"\n[probe] ligand chain from the XML: {chain!r}")
    if chain:
        try:
            sfxn, weights = _score_function_from_xml(pyrosetta, xml_path)
            total, per_term = _compute_interface_delta(pyrosetta, pose, chain, sfxn)
            print(f"[probe] computed directly (scorefxn '{weights}', constraints "
                  f"zeroed): interface_delta_{chain} = {total:+.3f} REU")
            top = sorted(per_term.items(), key=lambda kv: -abs(kv[1]))[:6]
            for k, v in top:
                print(f"           {k:24s} {v:+9.3f}")
            print("\n[probe] This is the fallback the shim now uses when the mover "
                  "stays silent,\n        so a sensible number here means your runs "
                  "will work.")
        except Exception as exc:
            print(f"[probe] the direct computation FAILED: {exc}")
            traceback.print_exc()

    print(f"\n[probe] all keys: {', '.join(sorted(sc))}")
    return sc


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--self-test" in argv:
        return 0 if self_test() else 1
    if "--probe" in argv:
        ap = argparse.ArgumentParser(prog="pyrosetta_rosetta_scripts.py --probe")
        ap.add_argument("--probe", nargs=2, metavar=("XML", "PDB"), required=True)
        ap.add_argument("--options", default=None,
                        help="the @options file from the same working directory")
        ap.add_argument("--verbose", action="store_true")
        args = ap.parse_args(argv)
        try:
            probe(args.probe[0], args.probe[1], args.options, args.verbose)
        except ImportError:
            print("error: PyRosetta is not importable in this interpreter",
                  file=sys.stderr)
            return 3
        except Exception:
            traceback.print_exc()
            return 4
        return 0
    if "--version" in argv:
        print(f"pyrosetta_rosetta_scripts {__version__}")
        return 0
    if "--write-wrapper" in argv:
        ap = argparse.ArgumentParser(prog="pyrosetta_rosetta_scripts.py")
        ap.add_argument("--write-wrapper", metavar="PATH", required=True)
        ap.add_argument("--python", default=sys.executable)
        args = ap.parse_args(argv)
        p = write_wrapper(args.write_wrapper, args.python)
        print(f"wrapper written: {p}")
        print(f'put this in run.json: "rosetta": "{p}"')
        return 0
    if not argv or argv[0] in ("-h", "--help", "-help"):
        print(__doc__)
        return 0

    verbose = os.environ.get("LPF_SHIM_VERBOSE", "") not in ("", "0", "false")
    jobs = int(os.environ.get("LPF_SHIM_JOBS", "1") or 1)

    opts, passthrough = parse_rosetta_argv(argv)
    for key, flag in (("protocol", "-parser:protocol"), ("input_pdb", "-in:file:s")):
        if not opts[key]:
            print(f"error: {flag} is required", file=sys.stderr)
            return 2
    for key, flag in (("protocol", "-parser:protocol"), ("input_pdb", "-in:file:s")):
        if not os.path.isfile(opts[key]):
            print(f"error: {flag} file not found: {opts[key]}", file=sys.stderr)
            return 2

    print(f"[shim] pyrosetta_rosetta_scripts {__version__} in {os.getcwd()}")
    print(f"[shim] nstruct={opts['nstruct']} jobs={jobs} "
          f"protocol={opts['protocol']} input={opts['input_pdb']}")

    try:
        runner = (run_pyrosetta_parallel if jobs > 1 else run_pyrosetta)
        rows = (runner(opts, passthrough, jobs, verbose=verbose) if jobs > 1
                else runner(opts, passthrough, verbose=verbose))
    except ImportError:
        print("error: PyRosetta is not importable in this interpreter.\n"
              "  pip install pyrosetta-installer && python -c "
              "'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'\n"
              "  then regenerate the wrapper with --write-wrapper so it points at "
              "the interpreter that has it.", file=sys.stderr)
        return 3
    except Exception:
        traceback.print_exc()
        return 4

    if not rows:
        print("error: every decoy failed; nothing to score", file=sys.stderr)
        return 5

    # interface_delta_X is the whole point of an LPF run: it is what
    # interface_scores.txt records. If no decoy produced one, the column would
    # still be written (LPF requires it) but filled with zeros, which looks like
    # a result rather than a missing one. Say so loudly instead.
    if not any("interface_delta_X" in sc for _, sc in rows):
        print("\n[shim] WARNING: no decoy reported interface_delta_X.\n"
              "[shim]   The column will be written as 0.0, so LPF's "
              "interface_scores.txt would be all zeros.\n"
              "[shim]   Usual cause: the protocol has no InterfaceScoreCalculator, or "
              "its chains= attribute\n"
              "[shim]   names a chain that is not the ligand's. Check the XML.",
              file=sys.stderr)

    score_dir = opts["out_path_score"] or "."
    try:
        out = write_score_file(os.path.join(score_dir, opts["scorefile"]), rows)
    except Exception:
        traceback.print_exc()
        return 6
    print(f"[shim] {len(rows)}/{opts['nstruct']} decoys -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
