#!/usr/bin/env python3
"""
lpf_run — command-line driver for Ligand PathFinder.

The same pipeline as the notebooks, for a terminal or a batch queue. It prepares
inputs for, launches, and analyses runs of the *unmodified* upstream code at
https://github.com/meilerlab/LigandPathFinder, for either a small molecule or a
peptide.

    Zlobin A, Eisenhuth P, Ferrario V, Meiler J. "Exploration of ligand
    (un)binding trajectories with Rosetta". bioRxiv 2026.
    doi:10.64898/2026.10.07.757317

Stages, which can be run in one go or separately:

    prep       build start.pdb, lig.params, lig_conformers.pdb, the guide
               points, the RosettaScripts protocols and the options file
    coarse     stage 1 — recursive bisection -> coarsepath.txt
    traj       stage 2 — the dense trajectory -> interface_scores.txt
    analyze    summary, energy profile, archive
    rescore    recompute interface energies without redoing the docking
    relocate   repair a project after you move its directory
    cluster    pool replicates, cluster the paths, average per cluster
    run        prep + coarse + traj + analyze
    doctor     check the environment and exit
    demo       the repository's BChE+paraoxon demo, small, to prove an install

`prep` writes project.json into the project directory; the later stages read it,
so each can run as its own cluster job.

Examples
--------
  # small molecule, bound pose in an SD file, everything in one go
  lpf_run.py run --target 1p0i.pdb --target-chains A \\
      --ligand pox.sdf --project runs/bche_pox

  # peptide taken out of the complex it is bound in
  lpf_run.py run --mode peptide --target 1cka.pdb --target-chains A \\
      --peptide-chain B --project runs/sh3_pep

  # prepare once, then submit the stages separately
  lpf_run.py prep  --target 1cka.pdb --target-chains A --mode peptide \\
      --peptide-chain B --project runs/sh3_pep
  lpf_run.py coarse --project runs/sh3_pep --replicates 10
  lpf_run.py traj   --project runs/sh3_pep --traj-nstruct 20

  # ligand files you already made with molfile_to_params
  lpf_run.py run --target prot.pdb --ligand-dir ligand/ \\
      --ligand-pose lig.pdb --project runs/mine

  # no Rosetta binary? use PyRosetta
  lpf_run.py run --rosetta-backend pyrosetta ...

  # check the set-up without spending any Rosetta time
  lpf_run.py prep ... --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import numpy as np
    import lpf_nb as L
except ImportError as exc:                                   # pragma: no cover
    sys.exit(f"lpf_run needs numpy, scipy and lpf_nb.py next to it: {exc}")

__version__ = "1.0"
MANIFEST = "project.json"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def banner(text):
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def die(msg, code=2):
    print(f"\nlpf_run: error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def find_repo(hint):
    """Locate the LigandPathFinder checkout, including next to this script."""
    return L.find_lpf_repo(hint,
                           os.path.join(HERE, "LigandPathFinder"),
                           os.path.join(os.path.dirname(HERE), "LigandPathFinder"))


def csv_list(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def xyz(s):
    v = [float(x) for x in csv_list(s)]
    if len(v) != 3:
        raise argparse.ArgumentTypeError(f"expected 'x,y,z', got {s!r}")
    return np.array(v)


def save_manifest(project_dir, data):
    """
    Write project.json atomically.

    Replicates launched concurrently (one process per run, which is how you use
    more than one core) each finish by updating the manifest. A plain open-and-
    write would let two of them interleave and leave truncated JSON that every
    later stage then fails to parse. Writing to a temporary file in the same
    directory and renaming is atomic on POSIX, so the worst case is simply that
    the last writer's `trajectory_run` wins — which is harmless.
    """
    path = os.path.join(project_dir, MANIFEST)
    data = dict(data)
    data["lpf_run_version"] = __version__
    data["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    fd, tmp = tempfile.mkstemp(dir=project_dir, prefix=".manifest-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    print(f"\nproject manifest -> {path}")
    return path


def load_manifest(project_dir):
    path = os.path.join(os.path.abspath(os.path.expanduser(project_dir)), MANIFEST)
    if not os.path.isfile(path):
        die(f"no {MANIFEST} in {project_dir}. Run `lpf_run.py prep` there first.")
    with open(path) as fh:
        man = json.load(fh)
    missing = [k for k in ("start_pdb", "ligand_dir", "files_dir", "mode")
               if k not in man]
    if missing:
        die(f"{path} is missing keys {missing}; it may be from another version")
    return man


def resolve_rosetta(args):
    """Settle on a backend. Returns (binary, database, is_mpi, backend_kind)."""
    info = L.resolve_rosetta_backend(
        backend=args.rosetta_backend,
        rosetta_hint=args.rosetta,
        python_exe=args.pyrosetta_python or None,
        shim_dir=os.path.join(args.project, "pyrosetta_backend"),
        install_if_missing=args.install_pyrosetta,
        shim_jobs=args.shim_jobs,
        database=args.rosetta_database,
    )
    return info["rosetta"], info["database"], info["is_mpi"], info["backend"]


# --------------------------------------------------------------------------
# stage: prep
# --------------------------------------------------------------------------

def stage_prep(args):
    project = os.path.abspath(os.path.expanduser(args.project))
    files_dir = os.path.join(project, "files")
    ligand_dir = os.path.join(project, "ligand")
    for d in (project, files_dir, ligand_dir):
        os.makedirs(d, exist_ok=True)
    args.project = project
    is_pep = args.mode == "peptide"

    banner("0 · environment")
    L.check_python_deps(optional=("matplotlib", "rdkit"))
    print()
    L.check_cli_tools(need_mpi=args.n_proc > 1)

    banner("1 · LigandPathFinder")
    repo = find_repo(args.lpf_repo)
    if not repo:
        if not args.clone:
            die("LigandPathFinder not found. Pass --lpf-repo /path/to/checkout, "
                "or --clone to fetch it.")
        repo = L.clone_lpf_repo(args.lpf_repo or os.path.join(project, "LigandPathFinder"))
    print("repo:", repo)
    xml_src = os.path.join(repo, "rosettaXML_files")

    banner("2 · Rosetta")
    rosetta, database, is_mpi, backend = resolve_rosetta(args)
    if args.n_proc > 1 and not is_mpi:
        extra = ("The PyRosetta backend has no MPI; use --shim-jobs instead."
                 if backend == "pyrosetta-shim" else "Use the .mpi. build.")
        die(f"--n-proc {args.n_proc} needs an '.mpi.' Rosetta build. {extra}")

    banner("3 · target protein")
    src = args.target
    if src and len(src) == 4 and not os.path.exists(src):
        src = L.fetch_pdb(src, project)
    if not src or not os.path.isfile(src):
        die(f"--target not found: {args.target!r}")
    print()
    L.pdb_summary(src, f"{os.path.basename(src)} as supplied:")
    keep = set(args.target_chains) if args.target_chains else None
    if keep and args.ligand_chain in keep:
        die(f"--ligand-chain {args.ligand_chain!r} is also a kept protein chain; "
            f"choose a free chain letter")
    print()
    protein_pdb = L.clean_protein(
        src, os.path.join(project, "protein.pdb"), keep_chains=keep,
        remove_waters=not args.keep_waters, keep_hetero=set(args.keep_hetero),
        strip_hydrogens=args.strip_hydrogens)

    banner("4 · " + ("peptide" if is_pep else "ligand"))
    atom_map = {}
    lig_pose = None
    if args.ligand_dir:
        srcdir = os.path.abspath(os.path.expanduser(args.ligand_dir))
        for f in ("lig.params", "lig_conformers.pdb"):
            p = os.path.join(srcdir, f)
            if not os.path.isfile(p):
                die(f"{f} not found in {srcdir}")
            dst = os.path.join(ligand_dir, f)
            # --ligand-dir may already BE the project's ligand/ directory, and on
            # macOS /tmp vs /private/tmp makes that non-obvious from the paths
            if os.path.exists(dst) and os.path.samefile(p, dst):
                continue
            shutil.copy(p, dst)
        lig_pose = (os.path.abspath(os.path.expanduser(args.ligand_pose))
                    if args.ligand_pose else None)
        print(f"using existing ligand files from {srcdir}")
    else:
        mol = _build_molecule(args, src, project, is_pep)
        lib = _build_conformers(args, mol, is_pep)
        nbr_index = None
        if is_pep or args.nbr_central:
            print()
            nbr_index = L.suggest_nbr_atom_index(lib)
        if args.conformer_report:
            print()
            L.conformer_library_report(lib, nbr_idx=(nbr_index or 1) - 1)
        sdf = L.write_conformer_sdf(
            lib, os.path.join(project, "conformers.sdf"),
            aromatic_bonds=not args.kekulize)
        print()
        mftp = L.ensure_molfile_to_params(
            os.path.join(project, "_molfile_to_params"),
            args.molfile_to_params, rosetta_bin=rosetta,
            allow_download=not args.no_download)
        print()
        res = L.run_molfile_to_params(
            sdf, ligand_dir, script=mftp, resname=args.resname,
            nbr_atom_index=nbr_index, recharge=args.recharge, verbose=args.verbose)
        lig_pose = res["lig_pdb"]
        if is_pep and getattr(args, "_peptide_pdb", None):
            print()
            atom_map = L.map_params_atoms_to_source(lig_pose, args._peptide_pdb)

    lig_params = os.path.join(ligand_dir, "lig.params")
    lig_confs = os.path.join(ligand_dir, "lig_conformers.pdb")

    banner("5 · tracked atom (NBR_ATOM)")
    print("current:", L.params_nbr_atom(lig_params),
          f"(NBR_RADIUS {L.params_nbr_radius(lig_params):.3f})")
    print()
    L.suggest_nbr_atom(conformers_pdb=lig_confs, params_path=lig_params, top=6)
    want = args.nbr_atom
    if args.nbr_atom_source:
        parts = args.nbr_atom_source.split()
        if len(parts) != 2:
            die('--nbr-atom-source needs "<atom> <resseq>", e.g. "CA 5"')
        hits = [p for p, (rn, rs, orig) in atom_map.items()
                if orig == parts[0] and rs == int(parts[1])]
        if not hits:
            die(f"no params atom maps to {parts[0]} of residue {parts[1]}"
                + (" (atom map is only built when the peptide is prepared here)"
                   if not atom_map else ""))
        want = hits[0]
        print(f"\n--nbr-atom-source {args.nbr_atom_source!r} -> {want}")
    if want:
        print()
        L.set_nbr_atom(lig_params, want, conformers_pdb=lig_confs,
                       margin=args.nbr_radius_margin)
    nbr_atom = L.params_nbr_atom(lig_params)
    print("\ntracking:", L.describe_atom(nbr_atom, atom_map) if atom_map else nbr_atom)

    banner("6 · start.pdb")
    if lig_pose is None:
        die("no ligand pose file to merge. Either let this script build the ligand "
            "(drop --ligand-dir), or pass --ligand-pose with a PDB of the ligand in "
            "its bound pose whose atom names match lig.params.")
    start_pdb = L.assemble_start_pdb(
        protein_pdb, lig_pose, os.path.join(project, "start.pdb"),
        params_path=lig_params, ligand_chain=args.ligand_chain,
        ligand_resnum=args.ligand_resnum,
        ligand_chain_in_source=args.ligand_pose_chain)
    header_pdb = L.write_header_pdb(
        os.path.join(project, "header.pdb"),
        open(args.header).read().splitlines() if args.header else [])
    print()
    L.pdb_summary(start_pdb, "start.pdb:")

    if args.minimize_start:
        banner("6b · relax the start pose into the Rosetta force field")
        print("tutorial section 1.3. Window 0 of the trajectory only settles the")
        print("interface region, so an unrelaxed complex shows up as a huge")
        print("window-0 energy.\n")
        # the options file has to exist before this runs, so write it early
        extra_now = [os.path.abspath(os.path.expanduser(q))
                     for q in args.extra_params]
        L.write_protocol_xmls(files_dir, xml_src, chain=args.ligand_chain,
                              tuning=L.LIGAND_XML_TUNING, files=("coarsepath.xml",),
                              verbose=False)
        L.write_options_file(
            os.path.join(files_dir, "options"),
            extra_res=(["lig.params", "VRT1.params"]
                       + [os.path.basename(q) for q in extra_now]),
            database=database, auto_setup_metals=args.auto_setup_metals)
        print()
        start_pdb = L.minimize_start_pose(
            start_pdb, os.path.join(project, "start_minimized.pdb"),
            rosetta=rosetta, files_dir=files_dir, ligand_dir=ligand_dir,
            ligand_chain=args.ligand_chain, nstruct=args.minimize_nstruct,
            extra_files=extra_now, minimize_backbone=not args.no_minimize_backbone,
            ca_restraint_sd=args.minimize_ca_sd, n_proc=args.n_proc,
            mpi_launcher="mpirun")

    span = L.ligand_span(lig_confs)
    print(f"\nspan over rotamers: {span:.1f} A")
    print()
    L.conformer_subsampling_preview(start_pdb, lig_confs, args.ligand_chain, lig_params)

    banner("7 · guide points")
    protein_coords = L.protein_heavy_coords(start_pdb, exclude_chain=args.ligand_chain)
    start_nbr = [a.xyz for a in L.read_pdb_atoms(start_pdb)
                 if a.chain == args.ligand_chain and a.name == nbr_atom][0]
    guide_path = _build_guide_path(args, protein_coords, start_nbr, span, project)

    banner("8 · protocols and options")
    grid = args.grid_width or L.auto_grid_width(span, box_size=args.transform_box)
    tuning = L.PEPTIDE_XML_TUNING if (is_pep and not args.no_tuning) else L.LIGAND_XML_TUNING
    if not args.no_tuning:
        tuning = dict(tuning)
        tuning["rules"] = list(tuning["rules"]) + [
            {"tag": "Transform", "name": n,
             "attrs": {"box_size": str(int(args.transform_box))}, "files": L._MAIN_LIKE}
            for n in ("transform", "transform_2")]
    print()
    L.write_protocol_xmls(files_dir, xml_src, chain=args.ligand_chain,
                          tuning=tuning, grid_width=grid, verbose=args.verbose)
    extra = [os.path.abspath(os.path.expanduser(p)) for p in args.extra_params]
    print()
    L.write_options_file(
        os.path.join(files_dir, "options"),
        extra_res=["lig.params", "VRT1.params"] + [os.path.basename(p) for p in extra],
        database=database, auto_setup_metals=args.auto_setup_metals)

    man = {
        "mode": args.mode, "project_dir": project, "lpf_repo": repo,
        "protein_pdb": protein_pdb, "start_pdb": start_pdb, "header_pdb": header_pdb,
        "ligand_dir": ligand_dir, "files_dir": files_dir,
        "ligand_chain": args.ligand_chain, "ligand_resnum": args.ligand_resnum,
        "resname": L.params_resname(lig_params), "nbr_atom": nbr_atom,
        "ligand_span": span, "guide_path": guide_path, "traj_path": guide_path,
        "rosetta": rosetta, "rosetta_database": database or "",
        "rosetta_backend": backend, "extra_files": extra,
        "grid_width": grid, "atom_map": {k: list(v) for k, v in atom_map.items()},
        "source_structure": os.path.abspath(src),
    }
    save_manifest(project, man)
    print("\nnext:  lpf_run.py coarse --project", project)
    return man


def _build_molecule(args, target_src, project, is_pep):
    """Load or build the RDKit molecule for the ligand / peptide."""
    if args.ligand:
        path = os.path.abspath(os.path.expanduser(args.ligand))
        if not os.path.isfile(path):
            die(f"--ligand not found: {path}")
        return L.mol_from_sdf(path)
    if args.ligand_smiles:
        return L.mol_from_smiles_3d(args.ligand_smiles, n_confs=1, seed=args.seed)
    if not is_pep:
        die("give one of --ligand FILE.sdf, --ligand-smiles SMILES, or --ligand-dir DIR")

    if args.peptide_pdb:
        pep = os.path.abspath(os.path.expanduser(args.peptide_pdb))
        if not os.path.isfile(pep):
            die(f"--peptide-pdb not found: {pep}")
    elif args.peptide_chain:
        pep = L.extract_chain(target_src, args.peptide_chain,
                              os.path.join(project, "peptide_raw.pdb"))
    else:
        die("for --mode peptide give --peptide-chain (to pull it out of --target) "
            "or --peptide-pdb, or supply --ligand/--ligand-dir yourself")

    print()
    rep = L.peptide_completeness_report(pep)
    if (rep["needs_oxt"] or rep["needs_o"]) and not args.no_repair_terminus:
        print()
        pep = L.complete_c_terminus(pep, os.path.join(project, "peptide.pdb"))
        rep = L.peptide_completeness_report(pep, verbose=False)
    if not rep["ok"]:
        die("the peptide is not chemically complete (see above). Rebuild the missing "
            "atoms upstream — PyMOL, pdbfixer, or a Rosetta FastRelax with the side "
            "chain repacked. Converting it as-is would give it the wrong chemistry.")
    args._peptide_pdb = pep
    print()
    return L.peptide_mol_from_pdb(
        pep, sequence=args.peptide_sequence or None,
        physiological_protonation=not args.no_protonation)


def _build_conformers(args, mol, is_pep):
    print()
    if is_pep:
        return L.generate_peptide_conformers(
            mol, n_local=args.n_local, n_medium=args.n_medium, n_global=args.n_global,
            sigma_local_deg=args.sigma_local, sigma_medium_deg=args.sigma_medium,
            prune_rmsd=args.prune_rmsd, minimize=not args.no_minimize, seed=args.seed)
    return L.generate_conformers(
        mol, n_confs=args.n_conformers, prune_rmsd=args.prune_rmsd,
        minimize=not args.no_minimize, seed=args.seed)


def _build_guide_path(args, protein_coords, start_nbr, span, project):
    if args.guide_path:
        gp = os.path.abspath(os.path.expanduser(args.guide_path))
        if not os.path.isfile(gp):
            die(f"--guide-path not found: {gp}")
        pts = L.read_guide_points(gp)
        print(f"using your guide path: {len(pts)} points")
        print()
        pts, was_rev = L.check_path_direction(pts, start_nbr,
                                              auto_reverse=args.reverse_path)
        if was_rev:
            if not args.reverse_path:
                die("the supplied path runs the wrong way (see above). Pass "
                    "--reverse-path to flip it, or export it site-to-outside.")
            gp = L.write_path_points(os.path.join(project, "path_reversed"), pts)
        print()
        L.path_clearance_report(pts, protein_coords, ligand_radius=span / 2)
        if len(pts) < 4:
            print("\n[warn] fewer than 4 points: usable for stage 1, but stage 2 fits "
                  "a cubic spline and needs at least 4")
        return gp

    min_dist = args.min_exit_distance or max(15.0, span / 2 + 10.0)
    if not args.min_exit_distance:
        print(f"exit distance (auto): {min_dist:.0f} A "
              f"= span {span:.1f}/2 + 10, floor 15\n")

    if args.exit_point is not None:
        end = args.exit_point
    else:
        rays = L.exit_direction_scan(
            protein_coords, start_nbr, n_dirs=args.scan_directions,
            hint=args.hint_vector, hint_cone_deg=(args.hint_cone if args.hint_vector
                                                 is not None else None),
            max_dist=args.scan_max_dist, top=args.list_rays or 8)
        if args.list_rays:
            print("\n--list-rays: stopping here. Re-run with --ray-index N to pick one.")
            raise SystemExit(0)
        if args.ray_index >= len(rays):
            die(f"--ray-index {args.ray_index} but only {len(rays)} rays were scanned")
        ray = rays[args.ray_index]
        print(f"\nchosen ray #{args.ray_index}: {ray}")
        if ray.bottleneck < span / 4:
            print(f"[warn] bottleneck {ray.bottleneck:.1f} A is tight for a molecule "
                  f"spanning {span:.1f} A — the protein will have to open up")
        print()
        end = L.exit_point_along(protein_coords, start_nbr, ray.direction,
                                 clearance=args.exit_clearance, min_dist=min_dist,
                                 ligand_span=span)
    print()
    return L.write_two_point_path(os.path.join(project, "path"), start_nbr, end)


# --------------------------------------------------------------------------
# stages: coarse, traj, analyze
# --------------------------------------------------------------------------

def opt(args, name, default=None):
    """
    Read an option that only some subcommands define.

    `coarse` has no --traj-step and `traj` has no --replicates, so the shared
    helpers below must tolerate their absence rather than assume one namespace.
    """
    return getattr(args, name, default)


def _params_for(mode, stage, args, n_ligand_atoms=None):
    """Start from the chosen preset, apply the paper's rules, then the overrides."""
    if stage == "coarse":
        base = dict(L.PEPTIDE_COARSE_PARAMS if mode == "peptide"
                    else L.LIGAND_COARSE_PARAMS)
        step = opt(args, "coarse_step")
        if step is not None:
            base["step"] = step
        jitter = opt(args, "jitter_radius")
        if jitter is not None:
            base["endpoint_jitter_radius"] = jitter
        if opt(args, "replicates", 1) > 1:
            base["endpoint_jitter"] = True
    else:
        preset = opt(args, "preset", "auto")
        if preset == "benchmark":
            base = dict(L.BENCHMARK_TRAJ_PARAMS)
            print("using the paper's benchmark trajectory settings "
                  "(SI, Geometry of Tunnels)")
        else:
            base = dict(L.PEPTIDE_TRAJ_PARAMS if mode == "peptide"
                        else L.LIGAND_TRAJ_PARAMS)
        for key, name in (("step", "traj_step"), ("smooth", "traj_smooth"),
                          ("outlier_rmsd", "outlier_rmsd"),
                          ("end_padding", "end_padding")):
            val = opt(args, name)
            if val is not None:
                base[key] = val
        base["run_stats"] = bool(opt(args, "run_stats", False))
        base["noise"] = bool(opt(args, "noise", False))
    thr = opt(args, "conformer_threshold")
    if thr is not None:
        base["conformer_subsampling"] = True
        base["conformer_rmsd_threshold"] = thr
    mem = opt(args, "memory_mode")
    if mem is not None:
        base["memory_mode"] = mem
    if opt(args, "dock_zero", False):
        base["dock_zero"] = True
    if n_ligand_atoms and not opt(args, "no_large_ligand_rule", False):
        base = L.apply_large_ligand_rule(base, n_ligand_atoms)
    for kv in opt(args, "param", []):
        if "=" not in kv:
            die(f"--param needs KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        if k not in base:
            print(f"[warn] --param {k} is not a {stage} key the defaults carry; "
                  f"setting it anyway")
        cur = base.get(k)
        if isinstance(cur, bool):
            base[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            base[k] = int(v)
        elif isinstance(cur, float):
            base[k] = float(v)
        else:
            try:
                base[k] = float(v) if "." in v else int(v)
            except ValueError:
                base[k] = v
    return base


def _config_common(man, args):
    return dict(
        files_dir=man["files_dir"], ligand_dir=man["ligand_dir"],
        start_pdb=man["start_pdb"], header_pdb=man.get("header_pdb") or None,
        ligand_chain=man["ligand_chain"], n_proc=opt(args, "n_proc", 1),
        extra_files=man.get("extra_files", []),
        rosetta_database=man.get("rosetta_database") or None,
        archive=not opt(args, "no_archive", False), verbose=True)


def stage_coarse(args):
    man = load_manifest(args.project)
    project = man["project_dir"]
    n_atoms = len(L.params_atom_names(os.path.join(man["ligand_dir"], "lig.params")))
    params = _params_for(man["mode"], "coarse", args, n_ligand_atoms=n_atoms)

    banner("stage 1 · coarse path")
    print("parameters:")
    for k, v in sorted(params.items()):
        print(f"   {k:34s} {v}")

    gp = L.read_guide_points(man["guide_path"])
    if len(gp) > 2:
        print(f"\n[note] {man['guide_path']} already has {len(gp)} points. The coarse "
              f"stage reads only the first and last; if this came from CAVER or "
              f"AQUA-DUCT, skip to `traj`.")
    nbr_xyz = [a.xyz for a in L.read_pdb_atoms(man["start_pdb"])
               if a.chain == man["ligand_chain"] and a.name == man["nbr_atom"]][0]
    print()
    if not L.check_path_anchor(gp, nbr_xyz, dock_zero=params.get("dock_zero", False)):
        die("the guide path and the start pose disagree (see above). Add "
            "--dock-zero, or fix the path / start pose.")
    seg = float(np.linalg.norm(gp[-1] - gp[0]))
    n_nodes = max(1, int(seg / params["step"]))
    print(f"\nguide segment {seg:.1f} A -> roughly {n_nodes} nodes per replicate")
    L.estimate_cost(n_nodes, args.coarse_nstruct, args.sec_per_decoy,
                    args.n_proc, label="node")
    if args.replicates > 1:
        print(f"x {args.replicates} replicates")

    runs, paths = [], []
    for rep in range(args.replicates):
        run_dir = os.path.join(project, f"coarse_{rep:02d}")
        L.prepare_run_dir(run_dir, clean=not args.keep_previous)
        cfg = L.build_run_config("coarse", man["rosetta"], guide_path=man["guide_path"],
                                 output_path=run_dir, params=params,
                                 nstruct=args.coarse_nstruct, **_config_common(man, args))
        cfg_path = L.write_run_config(os.path.join(run_dir, "run.json"), cfg)
        errs, _ = L.preflight_config(cfg_path, "coarse")
        if errs:
            die(f"preflight found {len(errs)} problem(s) — fix them and re-run")
        if args.dry_run:
            print("--dry-run: stopping before Rosetta")
            continue
        banner(f"replicate {rep + 1}/{args.replicates}")
        L.run_lpf_stage("coarse", man["lpf_repo"], cfg_path, run_dir,
                        overrides=args.set, log_path=os.path.join(run_dir, "coarse.log"))
        runs.append(run_dir)
        paths.append(os.path.join(run_dir, "coarsepath.txt"))

    if args.dry_run:
        return man
    banner("coarse results")
    for rd in runs:
        L.summarize_run(rd, stage="coarse")
        print()
    traj_path = paths[0]
    if len(paths) > 1:
        traj_path, _k, _D = L.coarse_path_medoid(paths)
    print("\npath going forward:", traj_path)

    pts = np.loadtxt(traj_path)
    print()
    L.path_clearance_report(
        pts, L.protein_heavy_coords(man["start_pdb"], exclude_chain=man["ligand_chain"]),
        ligand_radius=man["ligand_span"] / 2)
    if len(pts) < 4:
        die(f"the coarse path has only {len(pts)} points; stage 2 fits a cubic spline "
            f"and needs at least 4. Lower --coarse-step and re-run.")

    man["traj_path"] = traj_path
    man["coarse_runs"] = runs
    save_manifest(project, man)
    print("\nnext:  lpf_run.py traj --project", project)
    return man


def stage_traj(args):
    man = load_manifest(args.project)
    project = man["project_dir"]
    n_atoms = len(L.params_atom_names(os.path.join(man["ligand_dir"], "lig.params")))
    params = _params_for(man["mode"], "traj", args, n_ligand_atoms=n_atoms)
    traj_path = args.traj_path or man.get("traj_path") or man["guide_path"]

    banner("stage 2 · trajectory")
    print("guide path:", traj_path)
    print("parameters:")
    for k, v in sorted(params.items()):
        print(f"   {k:34s} {v}")

    pts = L.read_guide_points(traj_path)
    if len(pts) < 4:
        die(f"{traj_path} has {len(pts)} points; the trajectory spline needs at "
            f"least 4. Run `coarse` first, or supply a denser --guide-path.")
    nbr_xyz = [a.xyz for a in L.read_pdb_atoms(man["start_pdb"])
               if a.chain == man["ligand_chain"] and a.name == man["nbr_atom"]][0]
    print()
    if not L.check_path_anchor(pts, nbr_xyz, dock_zero=params.get("dock_zero", False)):
        die("the guide path and the start pose disagree (see above). Add "
            "--dock-zero, or fix the path / start pose.")
    print()
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    n_windows = max(1, int(length / params["step"]))
    print(f"\n{len(pts)} points, {length:.1f} A -> roughly {n_windows} windows "
          f"at step={params['step']} A")
    L.estimate_cost(n_windows, args.traj_nstruct, args.sec_per_decoy, args.n_proc)
    if params["run_stats"]:
        print("plus the statistics pass:")
        L.estimate_cost(n_windows, args.nstruct_stats, args.sec_per_decoy, args.n_proc)

    n_rep = max(1, int(opt(args, "traj_replicates", 1)))
    names = ([args.run_name] if n_rep == 1
             else [f"repl-{i:02d}" for i in range(n_rep)])

    # One guide path for every replicate, or one per replicate taken from the
    # coarse runs. The upstream tutorial does the former: its sbatch array copies
    # the same run.json into each replicate directory, so they share
    # input.guide_path. See the note in the README for what each one measures.
    paths = [traj_path] * len(names)
    if opt(args, "path_per_replicate", False):
        coarse = [os.path.join(d, "coarsepath.txt")
                  for d in man.get("coarse_runs", [])]
        coarse = [c for c in coarse if os.path.isfile(c)]
        if not coarse:
            die("--path-per-replicate needs coarse runs in the project. Run "
                "`coarse --replicates N` first; the manifest records them.")
        if len(coarse) < len(names):
            die(f"--path-per-replicate: {len(names)} replicates but only "
                f"{len(coarse)} coarse path(s). Run more coarse replicates, or "
                f"lower --traj-replicates.")
        paths = coarse[:len(names)]
        print(f"\none guide path per replicate, from {len(coarse)} coarse run(s):")
        for n, pth in zip(names, paths):
            print(f"   {n} <- {os.path.relpath(pth, project)}")
        print("This samples different ROUTES. For error bars on a single route, "
              "drop the flag\nand every replicate shares the medoid path, which "
              "is what the paper's sbatch\narray does.")

    if n_rep > 1:
        print(f"\n{n_rep} replicates, run serially: {names[0]} ... {names[-1]}")
        print("Each is an independent trajectory from the same inputs; pool them "
              "afterwards with\n`lpf_run.py cluster` or "
              "`L.replicate_convergence`. On a cluster, prefer the sbatch array "
              "in the README\nso they run in parallel.")
        L.estimate_cost(n_windows * n_rep, args.traj_nstruct, args.sec_per_decoy,
                        args.n_proc, label="window total")

    last = None
    for idx, (rep_name, rep_path) in enumerate(zip(names, paths), start=1):
        run_dir = os.path.join(project, rep_name)
        if n_rep > 1:
            banner(f"replicate {idx}/{n_rep}: {rep_name}")
        last = _run_one_trajectory(args, man, params, rep_path, run_dir)
    man["trajectory_run"] = last
    if n_rep > 1:
        man["trajectory_replicates"] = [os.path.join(project, n) for n in names]
    save_manifest(project, man)
    print("\nnext:  lpf_run.py analyze --project", project)
    if n_rep > 1:
        print("       lpf_run.py cluster --project", project)
    return man


def _run_one_trajectory(args, man, params, traj_path, run_dir):
    """One trajectory run. Resumes from its own checkpoint unless --fresh."""
    L.prepare_run_dir(run_dir, clean=args.fresh and args.clean_on_fresh)
    cfg = L.build_run_config("trajectory", man["rosetta"], guide_path=traj_path,
                             output_path=run_dir, params=params,
                             nstruct=args.traj_nstruct, nstruct_stats=args.nstruct_stats,
                             checkpoint_interval=args.checkpoint_interval,
                             use_tmp=args.use_tmp, tmp_path=args.tmp_path or "",
                             **_config_common(man, args))
    cfg_path = L.write_run_config(os.path.join(run_dir, "run.json"), cfg)
    errs, _ = L.preflight_config(cfg_path, "trajectory")
    if errs:
        die(f"preflight found {len(errs)} problem(s) — fix them and re-run")
    if args.dry_run:
        print("--dry-run: stopping before Rosetta")
        return run_dir

    L.run_lpf_stage("trajectory", man["lpf_repo"], cfg_path, run_dir,
                    overrides=args.set, fresh=args.fresh,
                    log_path=os.path.join(run_dir, "trajectory.log"))
    return run_dir


def stage_analyze(args):
    man = load_manifest(args.project)
    run_dir = args.run_dir or man.get("trajectory_run") \
        or os.path.join(man["project_dir"], opt(args, "run_name", "trajectory"))
    if not os.path.isdir(run_dir):
        die(f"no run directory at {run_dir}. Run `traj` first, or pass --run-dir.")

    banner("results")
    L.summarize_run(run_dir, stage="trajectory")
    d = L.load_run_scores(run_dir)
    y = d["interface_scores"]
    if y is None or not len(y):
        die(f"no interface_scores.txt in {run_dir} — the run did not finish")

    print()
    print(f"window 0 (bound)   {y[0]:+.2f} REU")
    print(f"last window (free) {y[-1]:+.2f} REU")
    print(f"deepest minimum    {y.min():+.2f} REU at window {int(y.argmin())}")
    print(f"released along it  {y[-1] - y.min():+.2f} REU")
    st = L.load_stats(run_dir)
    if st is not None:
        print(f"statistics pass    mean {np.round(st['interface_mean'], 2)}")
        print(f"                   sd   {np.round(st['interface_std'], 2)}")
    print("\nA single replicate is noisy — read the shape, not the digits, and pool "
          "replicates before quoting numbers.")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ax = L.plot_energy_profile(run_dir, x="distance", show_stats=True,
                                       smooth_window=args.smooth_window)
            out = args.plot if isinstance(args.plot, str) else \
                os.path.join(run_dir, "energy_profile.png")
            ax.figure.savefig(out, dpi=160)
            print(f"\nenergy profile -> {out}")
        except ImportError:
            print("\n[warn] matplotlib not installed; skipping the plot")

    if args.csv:
        nt = d["nbr_trajectory"]
        dist = L.cumulative_path_length(nt) if nt is not None and len(nt) == len(y) \
            else np.arange(len(y), dtype=float)
        out = args.csv if isinstance(args.csv, str) else \
            os.path.join(run_dir, "profile.csv")
        with open(out, "w") as fh:
            fh.write("window,distance_A,interface_delta_X,total_score\n")
            tot = d["total_scores"]
            for i, (x, v) in enumerate(zip(dist, y)):
                t = tot[i] if tot is not None and i < len(tot) else float("nan")
                fh.write(f"{i},{x:.3f},{v:.4f},{t:.4f}\n")
        print(f"table -> {out}")

    if args.zip:
        print()
        L.zip_results(run_dir, include_windows=args.zip_windows)
    return man


def stage_rescore(args):
    """Recompute interface energies for a finished run, without redoing docking."""
    man = load_manifest(args.project)
    run_dir = args.run_dir or man.get("trajectory_run") \
        or os.path.join(man["project_dir"], "trajectory")
    if not os.path.isdir(run_dir):
        die(f"no run directory at {run_dir}; pass --run-dir")

    banner("rescore: one scoring pass per window, no docking")
    print("For a run whose interface_scores.txt came out all zeros. Each window's")
    print("best.pdb is rescored once, so this costs minutes rather than hours.\n")

    files_dir = man["files_dir"]
    proto = args.protocol or os.path.join(files_dir, "zero_score.xml")
    if not os.path.isfile(proto):
        src = os.path.join(man["lpf_repo"], "rosettaXML_files", "zero_score.xml")
        if os.path.isfile(src):
            L.write_protocol_xmls(files_dir, os.path.dirname(src),
                                  chain=man["ligand_chain"],
                                  files=("zero_score.xml",), verbose=False)
            print(f"copied zero_score.xml into {files_dir} "
                  f"(chain {man['ligand_chain']})\n")
        else:
            die(f"no scoring protocol: {proto} is absent and {src} was not found")

    rosetta = args.rosetta or man["rosetta"]
    arr = L.rescore_windows(
        run_dir, rosetta=rosetta, files_dir=files_dir,
        ligand_dir=man["ligand_dir"], protocol=proto,
        ligand_chain=man["ligand_chain"], extra_files=man.get("extra_files", []),
        out_name=args.out_name, keep_work=args.keep_work, install=args.install)
    return arr


def stage_relocate(args):
    """Repair a project whose directory has been moved."""
    banner("relocate: repair the absolute paths in a moved project")
    remap = []
    for spec in args.remap:
        if "=" not in spec:
            die(f"--remap needs OLD=NEW, got {spec!r}")
        a, b = spec.split("=", 1)
        remap.append((a, b))
    res = L.relocate_project(args.project, remap=remap, old_root=args.old_root,
                             dry_run=args.dry_run)
    if res["stale"] and not args.dry_run:
        die(f"{len(res['stale'])} file(s) still reference the old path — "
            f"inspect them before using this project")
    return res


def stage_cluster(args):
    """Cluster pooled replicates the way the paper does (SI, Clustering)."""
    runs = []
    for pat in args.runs:
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        runs.extend(h for h in hits if os.path.isdir(h))
    if not runs and args.project:
        man = load_manifest(args.project)
        runs = sorted(glob.glob(os.path.join(man["project_dir"], "repl-*"))) \
            or sorted(glob.glob(os.path.join(man["project_dir"], "trajectory*")))
    if len(runs) < 2:
        die("need at least two finished trajectory run directories. Pass them as "
            "arguments or a glob, e.g. `lpf_run.py cluster 'runs/repl-*'`.")

    banner(f"clustering {len(runs)} replicate(s)")
    print("Method: frame-by-frame ligand coordinate RMSD with linearly decreasing")
    print("frame weights, missing frames penalised, affinity propagation on the")
    print("negative distance matrix, profiles averaged per cluster, small clusters")
    print("discarded (SI, 'Clustering of sampled paths').\n")

    man = None
    try:
        man = load_manifest(args.project) if args.project else None
    except SystemExit:
        man = None
    chain = args.ligand_chain or (man["ligand_chain"] if man else "X")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
    res = L.cluster_trajectories(
        runs, ligand_chain=chain, missing_penalty=args.missing_penalty,
        min_cluster_size=args.min_cluster_size, step=args.frame_step,
        plot=bool(args.plot))

    if args.convergence:
        banner("how many replicates do you need?")
        try:
            L.replicate_convergence(runs, step=args.frame_step, plot=bool(args.plot))
        except ValueError as exc:
            print("convergence:", exc)

    print()
    for c in res["kept"]:
        m = res["means"][c]
        print(f"  cluster {c}: n={m['n']:3d}  exemplar {m['exemplar']}  "
              f"bound {m['mean'][0]:+.2f} REU  final {m['mean'][-1]:+.2f} REU")
        if args.verbose:
            print(f"      members: {', '.join(m['runs'])}")

    out_dir = os.path.abspath(args.out or ".")
    os.makedirs(out_dir, exist_ok=True)
    if args.plot:
        import matplotlib.pyplot as plt
        png = args.plot if isinstance(args.plot, str) else \
            os.path.join(out_dir, "clusters.png")
        plt.savefig(png, dpi=160)
        print(f"\nplot -> {png}")
    if args.csv:
        csv = args.csv if isinstance(args.csv, str) else \
            os.path.join(out_dir, "clusters.csv")
        with open(csv, "w") as fh:
            fh.write("cluster,n,frame,distance_A,interface_mean,interface_sem\n")
            for c in res["kept"]:
                m = res["means"][c]
                for f, (mu, se) in enumerate(zip(m["mean"], m["sem"])):
                    fh.write(f"{c},{m['n']},{f},{f * res['step']:.3f},"
                             f"{mu:.4f},{se:.4f}\n")
        print(f"table -> {csv}")
    np.savetxt(os.path.join(out_dir, "path_distances.txt"), res["distances"],
               fmt="%.4f")
    print(f"distance matrix -> {os.path.join(out_dir, 'path_distances.txt')}")
    return res


def stage_doctor(args):
    banner("environment")
    try:
        L.check_python_deps()
    except ImportError as exc:
        print(exc)
    print()
    try:
        L.check_cli_tools(need_mpi=False)
    except FileNotFoundError as exc:
        print(exc)
    print()
    repo = find_repo(args.lpf_repo)
    print("LigandPathFinder:", repo or "NOT FOUND (pass --lpf-repo or --clone)")
    print()
    mftp = L.find_molfile_to_params(getattr(args, "molfile_to_params", None))
    print("molfile_to_params.py:", mftp or
          "not found locally (it can be downloaded on demand)")
    print()
    have, where = L.pyrosetta_available(args.pyrosetta_python or sys.executable)
    print("PyRosetta:", where if have else "not importable")
    print()
    cands = L.find_rosetta_scripts(args.rosetta)
    if cands:
        print("rosetta_scripts candidates:")
        for c in cands[:5]:
            print("   ", c, f"[{L.detect_rosetta_backend(c)}]")
        print()
        L.validate_rosetta(cands[0], args.rosetta_database)
    else:
        print("rosetta_scripts: none found")
        if have:
            print("  ...but PyRosetta is importable, so the default "
                  "--rosetta-backend auto will use it through the shim. "
                  "Nothing to do.")
        else:
            print("  and PyRosetta is not importable either, so there is no backend "
                  "yet. Either:")
            print("    * get a binary: "
                  "https://downloads.rosettacommons.org/downloads/academic/")
            print("    * or install PyRosetta: pip install pyrosetta-installer, then")
            print("      python -c 'import pyrosetta_installer; "
                  "pyrosetta_installer.install_pyrosetta()'")
    return {}


def stage_demo(args):
    """The repository's own BChE+paraoxon demo, small, to prove an install."""
    repo = find_repo(args.lpf_repo)
    if not repo:
        die("need the LigandPathFinder checkout for the demo: --lpf-repo /path, "
            "or --clone")
    demo = os.path.join(repo, "demo/trajectory")
    project = os.path.abspath(os.path.expanduser(args.project))
    files_dir = os.path.join(project, "files")
    os.makedirs(project, exist_ok=True)

    banner("demo · butyrylcholinesterase + paraoxon")
    print(f"inputs from {demo}")
    if os.path.isdir(os.path.join(project, "ligand")):
        shutil.rmtree(os.path.join(project, "ligand"))
    shutil.copytree(os.path.join(demo, "ligand"), os.path.join(project, "ligand"))
    for f in ("start.pdb", "header.pdb"):
        shutil.copy(os.path.join(demo, f), project)
    pts = np.loadtxt(os.path.join(demo, "path"))[:args.demo_points]
    L.write_path_points(os.path.join(project, "path"), pts)

    banner("Rosetta")
    args.project = project
    rosetta, database, _mpi, backend = resolve_rosetta(args)

    banner("protocols")
    L.write_protocol_xmls(files_dir, os.path.join(repo, "rosettaXML_files"),
                          chain="X", tuning=L.LIGAND_XML_TUNING, verbose=False)
    L.write_options_file(os.path.join(files_dir, "options"), database=database)

    man = {
        "mode": "ligand", "project_dir": project, "lpf_repo": repo,
        "protein_pdb": None, "start_pdb": os.path.join(project, "start.pdb"),
        "header_pdb": os.path.join(project, "header.pdb"),
        "ligand_dir": os.path.join(project, "ligand"), "files_dir": files_dir,
        "ligand_chain": "X", "ligand_resnum": 1, "resname": "lig",
        "nbr_atom": "O1", "ligand_span": L.ligand_span(
            os.path.join(project, "ligand/lig_conformers.pdb")),
        "guide_path": os.path.join(project, "path"),
        "traj_path": os.path.join(project, "path"),
        "rosetta": rosetta, "rosetta_database": database or "",
        "rosetta_backend": backend, "extra_files": [], "atom_map": {},
    }
    save_manifest(project, man)

    args.traj_path = None
    args.run_name = "run"
    stage_traj(args)
    if not args.dry_run:
        args.run_dir = None
        stage_analyze(args)
        print("\nIf the profile starts near -8 REU and climbs, your install is sound: "
              "that is the published value for this complex at the catalytic site.")
    return man


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def _p(**kw):
    return argparse.ArgumentParser(add_help=False, **kw)


def build_parser():
    common = _p()
    g = common.add_argument_group("project")
    g.add_argument("--project", "-o", required=True, metavar="DIR",
                   help="project directory: holds the prepared inputs and every run")
    g.add_argument("--n-proc", type=int, default=1, metavar="N",
                   help="MPI ranks per Rosetta call; >1 needs an .mpi. build and "
                        "N <= nstruct (default: 1)")
    g.add_argument("--no-archive", action="store_true",
                   help="do not zip each finished node/window directory")
    g.add_argument("--dry-run", action="store_true",
                   help="prepare and preflight, but never call Rosetta")
    g.add_argument("--sec-per-decoy", type=float, default=L.SEC_PER_DECOY_BENCHMARK,
                   metavar="S",
                   help="only for the runtime estimate. The default is implied by the "
                        "paper's benchmark (2.7 min/frame at nstruct 10, one core, "
                        "136 systems); its spread is 0.9-8.2 min/frame, so "
                        "recalibrate from your own first window "
                        f"(default: {int(L.SEC_PER_DECOY_BENCHMARK)})")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="passed straight to LPF as --set (repeatable), e.g. "
                        "--set output.use_tmp=true")
    g.add_argument("--verbose", "-v", action="store_true",
                   help="more detail from the preparation steps")

    env = _p()
    g = env.add_argument_group("environment")
    g.add_argument("--lpf-repo", metavar="DIR",
                   help="LigandPathFinder checkout (searched for if omitted)")
    g.add_argument("--clone", action="store_true",
                   help="clone LigandPathFinder if it is not found")

    ros = _p()
    g = ros.add_argument_group("Rosetta")
    g.add_argument("--rosetta", metavar="PATH",
                   help="rosetta_scripts binary (searched for if omitted)")
    g.add_argument("--rosetta-database", metavar="DIR",
                   help="Rosetta database; found next to the binary if omitted. "
                        "Written into the options file as -in:path:database")
    g.add_argument("--rosetta-backend", choices=("auto", "binary", "pyrosetta"),
                   default="auto",
                   help="'auto' prefers a compiled rosetta_scripts binary and falls "
                        "back to an importable PyRosetta — so activating a PyRosetta "
                        "environment is enough. 'pyrosetta' forces the shim, "
                        "'binary' forces a compiled build (default: auto)")
    g.add_argument("--pyrosetta-python", metavar="PATH",
                   help="interpreter that has PyRosetta (default: this one)")
    g.add_argument("--install-pyrosetta", action="store_true",
                   help="pip-install PyRosetta if absent (~1.5 GB download)")
    g.add_argument("--shim-jobs", type=int, default=1, metavar="N",
                   help="PyRosetta backend: worker processes per Rosetta call; each "
                        "needs 1-2 GB RAM (default: 1)")

    prep = _p()
    g = prep.add_argument_group("target protein")
    g.add_argument("--target", "-t", required=True, metavar="PDB|ID",
                   help="receptor structure: a PDB file, or a 4-character PDB ID to "
                        "download. For a peptide, give the complex it is bound in")
    g.add_argument("--target-chains", type=csv_list, default=None, metavar="A,B",
                   help="protein chains to keep (default: all)")
    g.add_argument("--keep-hetero", type=csv_list, default=[], metavar="HEM,ZN",
                   help="hetero residues to keep; each needs its own .params via "
                        "--extra-params")
    g.add_argument("--keep-waters", action="store_true", help="keep water molecules")
    g.add_argument("--strip-hydrogens", action="store_true",
                   help="drop hydrogens and let Rosetta rebuild them")
    g.add_argument("--extra-params", type=csv_list, default=[], metavar="F.params",
                   help="cofactor params files, added to extra_files and to "
                        "-extra_res_fa")
    g.add_argument("--auto-setup-metals", action="store_true",
                   help="add -in:auto_setup_metals to the options file")
    g.add_argument("--header", metavar="FILE",
                   help="lines prepended to every task.pdb (metal constraints, "
                        "membrane spans, symmetry)")

    g = prep.add_argument_group("ligand / peptide")
    g.add_argument("--mode", choices=("ligand", "peptide"), default="ligand",
                   help="selects the parameter defaults and protocol tuning "
                        "(default: ligand)")
    g.add_argument("--ligand", "-l", metavar="FILE.sdf",
                   help="SD/MOL file whose FIRST entry is the bound pose")
    g.add_argument("--ligand-smiles", metavar="SMILES",
                   help="build from SMILES — no bound pose, so for testing only")
    g.add_argument("--ligand-dir", metavar="DIR",
                   help="directory that already holds lig.params + "
                        "lig_conformers.pdb")
    g.add_argument("--ligand-pose", metavar="PDB",
                   help="with --ligand-dir: the ligand in its bound pose, atom names "
                        "matching lig.params")
    g.add_argument("--ligand-pose-chain", metavar="CH",
                   help="if --ligand-pose is a whole complex, which chain in it is "
                        "the ligand")
    g.add_argument("--peptide-chain", metavar="CH",
                   help="--mode peptide: chain of the bound peptide in --target")
    g.add_argument("--peptide-pdb", metavar="PDB",
                   help="--mode peptide: peptide-only PDB, in the receptor's frame")
    g.add_argument("--peptide-sequence", metavar="SEQ",
                   help="override the one-letter sequence read from the PDB")
    g.add_argument("--no-repair-terminus", action="store_true",
                   help="do not build a missing terminal O/OXT")
    g.add_argument("--no-protonation", action="store_true",
                   help="keep the neutral template instead of pH-7 charge states")
    g.add_argument("--ligand-chain", default="X", metavar="CH",
                   help="chain the ligand gets in start.pdb; must be free "
                        "(default: X)")
    g.add_argument("--ligand-resnum", type=int, default=1, metavar="N",
                   help="residue number of the ligand in its chain (default: 1)")
    g.add_argument("--resname", default=None, metavar="ABC",
                   help="3-character PDB residue name (default: lig, or pep for "
                        "--mode peptide)")

    g = prep.add_argument_group("conformer library")
    g.add_argument("--n-conformers", type=int, default=200, metavar="N",
                   help="ligand mode: ETKDG conformers to embed (default: 200)")
    g.add_argument("--n-local", type=int, default=120, metavar="N",
                   help="peptide mode: small perturbations of a few torsions")
    g.add_argument("--n-medium", type=int, default=120, metavar="N",
                   help="peptide mode: larger perturbations of more torsions")
    g.add_argument("--n-global", type=int, default=60, metavar="N",
                   help="peptide mode: free ETKDG embeddings")
    g.add_argument("--sigma-local", type=float, default=12.0, metavar="DEG")
    g.add_argument("--sigma-medium", type=float, default=45.0, metavar="DEG")
    g.add_argument("--prune-rmsd", type=float, default=None, metavar="A",
                   help="minimum RMSD between kept conformers "
                        "(default: 0.5 ligand, 0.6 peptide)")
    g.add_argument("--no-minimize", action="store_true",
                   help="skip the short MMFF relaxation of each conformer")
    g.add_argument("--kekulize", action="store_true",
                   help="write a Kekule SD file; molfile_to_params expects explicit "
                        "aromatic bonds, so leave this off")
    g.add_argument("--recharge", default=None, metavar="CHG",
                   help="force the net charge instead of taking it from the SD "
                        "file's formal charges")
    g.add_argument("--conformer-report", action="store_true",
                   help="report the library's RMSD ladder and whether it is connected")
    g.add_argument("--molfile-to-params", metavar="PATH",
                   help="Rosetta's molfile_to_params.py (searched for, then "
                        "downloaded, if omitted)")
    g.add_argument("--no-download", action="store_true",
                   help="never download molfile_to_params.py")
    g.add_argument("--seed", type=int, default=0xF00D, metavar="N")

    g = prep.add_argument_group("tracked atom (NBR_ATOM)")
    g.add_argument("--nbr-atom", metavar="NAME",
                   help="params atom name to track, e.g. C17")
    g.add_argument("--nbr-atom-source", metavar='"CA 5"',
                   help="name it in the original peptide instead, e.g. \"CA 5\"")
    g.add_argument("--nbr-central", action="store_true",
                   help="ligand mode: pick the most central atom rather than keeping "
                        "molfile_to_params' choice (always done for peptides)")
    g.add_argument("--nbr-radius-margin", type=float, default=0.5, metavar="A")

    g = prep.add_argument_group("guide points")
    g.add_argument("--guide-path", metavar="FILE",
                   help="your own path (xyz text or PDB), e.g. from CAVER or "
                        "AQUA-DUCT. With 4+ points you can skip stage 1. It must "
                        "run from the binding site outwards")
    g.add_argument("--reverse-path", action="store_true",
                   help="flip --guide-path, for a tunnel exported outside-in")
    g.add_argument("--exit-point", type=xyz, default=None, metavar="X,Y,Z",
                   help="exit point, instead of scanning for one")
    g.add_argument("--ray-index", type=int, default=0, metavar="N",
                   help="which ranked exit ray to take (default: 0, the widest)")
    g.add_argument("--list-rays", type=int, nargs="?", const=12, default=0,
                   metavar="N", help="print the top N exit rays and stop")
    g.add_argument("--hint-vector", type=xyz, default=None, metavar="X,Y,Z",
                   help="restrict the scan to a cone about this direction")
    g.add_argument("--hint-cone", type=float, default=60.0, metavar="DEG")
    g.add_argument("--scan-directions", type=int, default=512, metavar="N")
    g.add_argument("--scan-max-dist", type=float, default=50.0, metavar="A")
    g.add_argument("--exit-clearance", type=float, default=8.0, metavar="A",
                   help="how far from the protein the endpoint must sit (default: 8)")
    g.add_argument("--min-exit-distance", type=float, default=None, metavar="A",
                   help="path length floor (default: from the molecule's span)")

    g = prep.add_argument_group("start-pose minimisation (tutorial 1.3)")
    g.add_argument("--minimize-start", action="store_true",
                   help="relax the assembled complex into the Rosetta force field "
                        "before running LPF. Do this unless your start pose is "
                        "already minimised — LPF assumes it is")
    g.add_argument("--minimize-nstruct", type=int, default=5, metavar="N",
                   help="independent minimisation attempts; the best by "
                        "constraint-free total_score is kept (default: 5)")
    g.add_argument("--minimize-ca-sd", type=float, default=0.5, metavar="A",
                   help="C-alpha coordinate restraint sd; 0 reproduces the "
                        "tutorial's unrestrained `minimize` run (default: 0.5)")
    g.add_argument("--no-minimize-backbone", action="store_true",
                   help="side chains and rigid-body only")

    g = prep.add_argument_group("protocols")
    g.add_argument("--grid-width", type=float, default=None, metavar="A",
                   help="SCORINGGRIDS width (default: sized from the molecule)")
    g.add_argument("--transform-box", type=float, default=None, metavar="A",
                   help="Transform translation box (default: 5 ligand, 8 peptide)")
    g.add_argument("--no-tuning", action="store_true",
                   help="copy the upstream protocols verbatim, only retargeting the "
                        "chain")

    pars = _p()
    g = pars.add_argument_group("shared stage parameters")
    g.add_argument("--conformer-threshold", type=float, default=None, metavar="A",
                   help="conformer_rmsd_threshold; also switches subsampling on")
    g.add_argument("--memory-mode", dest="memory_mode", action="store_true",
                   default=None, help="carry the whole relaxed pose forward, so "
                                      "induced protein motion accumulates")
    g.add_argument("--no-memory-mode", dest="memory_mode", action="store_false",
                   help="transplant only the ligand coordinates each step")
    g.add_argument("--dock-zero", action="store_true",
                   help="dock onto the path's first point before starting")
    g.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="override any parameters.* key (repeatable), e.g. "
                        "--param rmsd_constraint_sd=1.5")
    g.add_argument("--preset", choices=("auto", "benchmark"), default="auto",
                   help="'auto' uses the demo-derived defaults for the chosen "
                        "--mode; 'benchmark' uses the settings from the paper's "
                        "136-system benchmark, which suit an external CAVER path "
                        "(default: auto)")
    g.add_argument("--no-large-ligand-rule", action="store_true",
                   help="do not apply the paper's >50-atom settings "
                        "(conformer_rmsd_threshold and outlier_rmsd both 5.0)")

    co = _p()
    g = co.add_argument_group("stage 1 · coarse")
    g.add_argument("--coarse-step", type=float, default=None, metavar="A",
                   help="node spacing (default: 2.0 ligand, 3.0 peptide)")
    g.add_argument("--coarse-nstruct", type=int, default=10, metavar="N",
                   help="decoys per node (default: 10)")
    g.add_argument("--replicates", "-r", type=int, default=1, metavar="N",
                   help="independent coarse runs; the medoid path goes forward "
                        "(default: 1)")
    g.add_argument("--jitter-radius", type=float, default=None, metavar="A",
                   help="endpoint jitter radius (default: 5)")
    g.add_argument("--keep-previous", action="store_true",
                   help="do not wipe an existing coarse directory (the coarse stage "
                        "cannot resume, so re-runs normally need a clean one)")

    tr = _p()
    g = tr.add_argument_group("stage 2 · trajectory")
    g.add_argument("--traj-step", type=float, default=None, metavar="A",
                   help="window spacing (default: 0.5 ligand, 1.0 peptide)")
    g.add_argument("--traj-smooth", type=float, default=None, metavar="S",
                   help="spline smoothing (default: 0.5 ligand, 1.0 peptide)")
    g.add_argument("--traj-nstruct", type=int, default=10, metavar="N",
                   help="decoys per window (default: 10)")
    g.add_argument("--traj-path", metavar="FILE",
                   help="guide path to use (default: the coarse medoid)")
    g.add_argument("--run-stats", action="store_true",
                   help="second scoring pass per window, i.e. error bars")
    g.add_argument("--nstruct-stats", type=int, default=10, metavar="N")
    g.add_argument("--noise", action="store_true",
                   help="jitter the projected previous point, to escape local traps")
    g.add_argument("--outlier-rmsd", type=float, default=None, metavar="A")
    g.add_argument("--end-padding", type=float, default=None, metavar="A",
                   help="relax the in-plane restraint over the last windows")
    g.add_argument("--checkpoint-interval", type=int, default=10, metavar="N",
                   help="0 disables; the run resumes from checkpoint.pkl (default: 10)")
    g.add_argument("--use-tmp", action="store_true",
                   help="run each window on a RAM disc and copy results back")
    g.add_argument("--tmp-path", metavar="DIR")
    g.add_argument("--fresh", action="store_true",
                   help="ignore an existing checkpoint and start over")
    g.add_argument("--clean-on-fresh", action="store_true",
                   help="with --fresh, also delete the old run directory")
    g.add_argument("--run-name", default="trajectory", metavar="NAME",
                   help="trajectory run subdirectory (default: trajectory)")
    g.add_argument("--path-per-replicate", action="store_true",
                   help="give each replicate its own coarse path (from the "
                        "coarse_NN runs) instead of all sharing the medoid. "
                        "Samples different exit ROUTES rather than putting error "
                        "bars on one route")
    g.add_argument("--traj-replicates", "-R", type=int, default=1, metavar="N",
                   help="run N independent trajectories serially, into repl-00.. "
                        "repl-NN, then pool them with `cluster`. On a cluster use "
                        "the sbatch array instead, which runs them in parallel "
                        "(default: 1)")

    an = _p()
    g = an.add_argument_group("analysis")
    g.add_argument("--run-dir", metavar="DIR",
                   help="trajectory run to analyse (default: the one just run)")
    g.add_argument("--plot", nargs="?", const=True, default=False, metavar="PNG",
                   help="save the energy profile (default name inside the run dir)")
    g.add_argument("--smooth-window", type=int, default=0, metavar="N",
                   help="also draw a running mean over N windows")
    g.add_argument("--csv", nargs="?", const=True, default=False, metavar="CSV",
                   help="write window, distance, interface and total score as CSV")
    g.add_argument("--zip", action="store_true", help="archive the analysis files")
    g.add_argument("--zip-windows", action="store_true",
                   help="include every window directory in the archive (large)")

    ap = argparse.ArgumentParser(
        prog="lpf_run.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("Examples")[0].strip(),
        epilog="Examples" + __doc__.split("Examples")[1])
    ap.add_argument("--version", action="version",
                    version=f"lpf_run {__version__}")
    sub = ap.add_subparsers(dest="stage", metavar="STAGE")

    sub.add_parser("prep", parents=[common, env, ros, prep],
                   help="build every input file, then stop",
                   description="Build start.pdb, the ligand files, the guide points, "
                               "the protocols and the options file, and write "
                               "project.json.")
    sub.add_parser("coarse", parents=[common, pars, co],
                   help="stage 1 — coarse path by bisection",
                   description="Needs a project prepared by `prep`.")
    sub.add_parser("traj", parents=[common, pars, tr],
                   help="stage 2 — the dense trajectory",
                   description="Needs a project prepared by `prep` (and normally a "
                               "coarse path). Resumes from checkpoint.pkl.")
    sub.add_parser("analyze", parents=[common, an],
                   help="summarise a finished trajectory")
    r = sub.add_parser("run", parents=[common, env, ros, prep, pars, co, tr, an],
                       help="prep + coarse + traj + analyze",
                       description="The whole pipeline in one go.")
    r.add_argument("--skip-coarse", action="store_true",
                   help="go straight to stage 2 (needs --guide-path with 4+ points)")
    cl = sub.add_parser("cluster", parents=[],
                        help="cluster pooled replicates (the paper's method)",
                        description="Cluster finished trajectory runs by path "
                                    "similarity and average their energy profiles "
                                    "per cluster, as the paper does for an "
                                    "ensemble of replicates.")
    cl.add_argument("runs", nargs="*", default=[], metavar="RUN_DIR",
                    help="finished trajectory run directories, or a glob")
    cl.add_argument("--project", "-o", metavar="DIR",
                    help="take repl-*/ or trajectory*/ from this project instead")
    cl.add_argument("--ligand-chain", metavar="CH",
                    help="default: from the project manifest, else X")
    cl.add_argument("--min-cluster-size", type=int, default=10, metavar="N",
                    help="discard clusters with this many members or fewer "
                         "(default: 10, the paper's value)")
    cl.add_argument("--missing-penalty", type=float, default=0.5, metavar="A",
                    help="distance charged per missing frame (default: 0.5, the "
                         "paper's value)")
    cl.add_argument("--frame-step", type=float, default=None, metavar="A",
                    help="per-frame spacing for the x-axis (default: from run.json)")
    cl.add_argument("--plot", nargs="?", const=True, default=True, metavar="PNG")
    cl.add_argument("--csv", nargs="?", const=True, default=False, metavar="CSV")
    cl.add_argument("--out", metavar="DIR", help="where to write outputs")
    cl.add_argument("--convergence", action="store_true",
                    help="also measure how much the pooled mean still moves as "
                         "replicates are added, to judge whether you have enough")
    cl.add_argument("--verbose", "-v", action="store_true")

    rs = sub.add_parser("rescore", parents=[],
                        help="recompute interface energies for a finished run",
                        description="Rescore each window's selected pose with a "
                                    "scoring-only protocol. Use this when "
                                    "interface_scores.txt came out all zeros: it "
                                    "costs one Rosetta call per window instead of "
                                    "nstruct docking runs.")
    rs.add_argument("--project", "-o", required=True, metavar="DIR")
    rs.add_argument("--run-dir", metavar="DIR",
                    help="default: the trajectory run in the manifest")
    rs.add_argument("--protocol", metavar="XML",
                    help="default: zero_score.xml from the project's files/")
    rs.add_argument("--rosetta", metavar="PATH",
                    help="default: the binary recorded in the manifest")
    rs.add_argument("--out-name", default="interface_scores_rescored.txt",
                    metavar="NAME",
                    help="written next to the original, which is left untouched")
    rs.add_argument("--install", action="store_true",
                    help="replace interface_scores.txt with the rescored values "
                         "(the original is backed up first), so analyze and "
                         "cluster use them")
    rs.add_argument("--keep-work", action="store_true",
                    help="keep the per-window working directories")

    rl = sub.add_parser("relocate", parents=[],
                        help="repair a project directory that has been moved",
                        description="Every path this tooling writes is absolute, "
                                    "so moving a project leaves stale paths in "
                                    "project.json, each run.json, the checkpoints "
                                    "and the PyRosetta launcher. This rewrites "
                                    "them to the directory's new location.")
    rl.add_argument("--project", "-o", required=True, metavar="DIR",
                    help="the project in its NEW location")
    rl.add_argument("--old-root", metavar="PATH",
                    help="the old project path; read from project.json otherwise")
    rl.add_argument("--remap", action="append", default=[], metavar="OLD=NEW",
                    help="extra path mapping for files that lived outside the "
                         "project, e.g. the --target structure (repeatable)")
    rl.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")

    sub.add_parser("doctor", parents=[env, ros],
                   help="check the environment and exit")
    d = sub.add_parser("demo", parents=[common, env, ros, pars, tr, an],
                       help="run the repository's BChE+paraoxon demo")
    d.add_argument("--demo-points", type=int, default=6, metavar="N",
                   help="how many of the demo's guide points to use (default: 6, "
                        "a short run)")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.stage:
        ap.print_help()
        return 1

    # mode-dependent defaults that argparse cannot express
    if getattr(args, "mode", None) == "peptide":
        if args.resname is None:
            args.resname = "pep"
        if args.prune_rmsd is None:
            args.prune_rmsd = 0.6
        if args.transform_box is None:
            args.transform_box = 8.0
    elif hasattr(args, "mode"):
        if args.resname is None:
            args.resname = "lig"
        if args.prune_rmsd is None:
            args.prune_rmsd = 0.5
        if args.transform_box is None:
            args.transform_box = 5.0
    if hasattr(args, "resname") and args.resname and len(args.resname) > 3:
        die(f"--resname must be at most 3 characters, got {args.resname!r}")

    t0 = time.time()
    try:
        if args.stage == "prep":
            stage_prep(args)
        elif args.stage == "coarse":
            stage_coarse(args)
        elif args.stage == "traj":
            stage_traj(args)
        elif args.stage == "analyze":
            stage_analyze(args)
        elif args.stage == "relocate":
            stage_relocate(args)
        elif args.stage == "rescore":
            stage_rescore(args)
        elif args.stage == "cluster":
            stage_cluster(args)
        elif args.stage == "doctor":
            stage_doctor(args)
        elif args.stage == "demo":
            stage_demo(args)
        elif args.stage == "run":
            man = stage_prep(args)
            pts = L.read_guide_points(man["guide_path"])
            if args.skip_coarse or len(pts) >= 4:
                if not args.skip_coarse:
                    print(f"\n[note] the guide path already has {len(pts)} points, so "
                          f"stage 1 is unnecessary — going straight to stage 2. "
                          f"Pass --replicates to force a coarse run.")
                if args.replicates > 1 and not args.skip_coarse:
                    stage_coarse(args)
            else:
                stage_coarse(args)
            if not args.dry_run:
                stage_traj(args)
                args.run_dir = None
                stage_analyze(args)
    except KeyboardInterrupt:
        print("\ninterrupted. The trajectory stage resumes from its checkpoint: "
              "re-run the same command.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        print(f"\nlpf_run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"\ndone in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
