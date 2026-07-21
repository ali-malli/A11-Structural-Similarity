#!/usr/bin/env python3
"""
Build candidate A11 surface sites after DSSP.

Inputs:
  cleaned PDB files (heteroatoms removed, all desired chains retained)
  DSSP files already generated from those PDBs

Outputs:
  candidate_sites.pkl : full Python object for downstream scoring
  candidate_sites_atoms.csv : atom-level table of all expanded candidate sites
  site_summary.csv : one row per candidate site
"""

import argparse
import os
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

BACKBONE = {"N", "CA", "C", "O"}
BETA_CODES_DEFAULT = {"E", "B"}  # E=extended beta strand, B=isolated beta bridge


@dataclass(frozen=True)
class AtomRecord:
    pdb_id: str
    atom_uid: int
    chain_id: str
    resseq: int
    icode: str
    resname: str
    atom_name: str
    element: str
    x: float
    y: float
    z: float
    sasa: float
    ss: str
    is_beta: bool
    is_surface: bool


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb_dir", required=True)
    p.add_argument("--dssp_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pdb_glob", default="*.pdb")
    p.add_argument("--dssp_ext", default=".dssp")
    p.add_argument("--probe_radius", type=float, default=1.4)
    p.add_argument("--cluster_cutoff", type=float, default=4.0)
    p.add_argument("--min_cluster_atoms", type=int, default=4)
    p.add_argument("--beta_codes", default="E,B", help="DSSP codes treated as beta, comma-separated")
    p.add_argument("--backbone_only", action="store_true", help="Cluster only exposed beta backbone atoms N,CA,C,O")
    return p.parse_args()


def parse_dssp_file(path):
    """Parse DSSP fixed-width output into {(chain, resseq, icode): ss_code}.

    Robust enough for standard mkdssp/dssp text output. DSSP starts residue table
    after a line containing '#  RESIDUE'. Columns: resseq [5:10], icode [10], chain [11], ss [16].
    """
    ss_map = {}
    started = False
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if not started:
                if line.lstrip().startswith("#") and "RESIDUE" in line:
                    started = True
                continue
            if len(line) < 17:
                continue
            # Skip chain breaks marked by !
            if line[13:14] == "!":
                continue
            resseq_txt = line[5:10].strip()
            if not re.match(r"^-?\d+$", resseq_txt):
                continue
            resseq = int(resseq_txt)
            icode = line[10].strip() or ""
            chain = line[11].strip() or " "
            ss = line[16].strip() or "C"
            ss_map[(chain, resseq, icode)] = ss
    return ss_map


def load_atoms_with_sasa(pdb_path, dssp_path, pdb_id, beta_codes, probe_radius):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(pdb_path))

    # Atom-level SASA with 1.4 A probe
    sr = ShrakeRupley(probe_radius=probe_radius, n_points=960)
    sr.compute(structure, level="A")

    ss_map = parse_dssp_file(dssp_path)
    records = []
    uid = 0
    # Use first model only, standard for crystallographic PDBs
    model = next(structure.get_models())
    for chain in model:
        chain_id = chain.id
        for residue in chain:
            hetflag, resseq, icode = residue.id
            # You said heteroatoms are removed, but keep this safeguard.
            if hetflag.strip():
                continue
            icode = icode.strip() or ""
            ss = ss_map.get((chain_id, resseq, icode), "C")
            is_beta = ss in beta_codes
            resname = residue.get_resname().strip()
            for atom in residue:
                # Ignore alternate locations except blank/A
                altloc = atom.get_altloc()
                if altloc not in (" ", "A"):
                    continue
                coord = atom.coord.astype(float)
                atom_name = atom.get_name().strip()
                element = (atom.element or atom_name[0]).strip().upper()
                sasa = float(getattr(atom, "sasa", 0.0) or 0.0)
                records.append(AtomRecord(
                    pdb_id=pdb_id,
                    atom_uid=uid,
                    chain_id=chain_id,
                    resseq=resseq,
                    icode=icode,
                    resname=resname,
                    atom_name=atom_name,
                    element=element,
                    x=coord[0], y=coord[1], z=coord[2],
                    sasa=sasa,
                    ss=ss,
                    is_beta=is_beta,
                    is_surface=sasa > 0.0,
                ))
                uid += 1
    return records


def connected_components_with_cutoff(coords, cutoff):
    if len(coords) == 0:
        return np.array([], dtype=int), 0
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
    n = len(coords)
    if len(pairs) == 0:
        graph = csr_matrix((n, n), dtype=np.int8)
    else:
        row = np.concatenate([pairs[:, 0], pairs[:, 1]])
        col = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(len(row), dtype=np.int8)
        graph = csr_matrix((data, (row, col)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    return labels, n_comp


def build_sites_for_pdb(records, cluster_cutoff, min_cluster_atoms, backbone_only):
    df = pd.DataFrame([asdict(r) for r in records])
    all_coords = df[["x", "y", "z"]].to_numpy(float)

    surface_mask = df["is_surface"].to_numpy(bool)
    beta_mask = df["is_beta"].to_numpy(bool)
    if backbone_only:
        beta_mask &= df["atom_name"].isin(BACKBONE).to_numpy(bool)

    seed_mask = surface_mask & beta_mask
    seed_indices = np.flatnonzero(seed_mask)
    seed_coords = all_coords[seed_indices]

    labels, n_comp = connected_components_with_cutoff(seed_coords, cluster_cutoff)
    surface_indices = np.flatnonzero(surface_mask)
    surface_coords = all_coords[surface_indices]
    surface_tree = cKDTree(surface_coords) if len(surface_coords) else None

    sites = []
    site_atom_rows = []
    local_site_id = 0
    for comp_id in range(n_comp):
        comp_seed_local = np.flatnonzero(labels == comp_id)
        if len(comp_seed_local) < min_cluster_atoms:
            continue
        comp_seed_global = seed_indices[comp_seed_local]
        comp_coords = all_coords[comp_seed_global]

        expanded_surface_local = set()
        if surface_tree is not None:
            hits = surface_tree.query_ball_point(comp_coords, r=cluster_cutoff)
            for h in hits:
                expanded_surface_local.update(h)
        expanded_global = sorted(surface_indices[list(expanded_surface_local)].tolist())

        site_id = f"{df.iloc[0]['pdb_id']}_site{local_site_id:04d}"
        local_site_id += 1
        sites.append({
            "site_id": site_id,
            "pdb_id": df.iloc[0]["pdb_id"],
            "n_seed_atoms": int(len(comp_seed_global)),
            "n_expanded_atoms": int(len(expanded_global)),
            "seed_atom_uids": df.iloc[comp_seed_global]["atom_uid"].astype(int).tolist(),
            "expanded_atom_uids": df.iloc[expanded_global]["atom_uid"].astype(int).tolist(),
        })
        for idx in expanded_global:
            row = df.iloc[idx].to_dict()
            row["site_id"] = site_id
            row["is_seed_atom"] = int(idx in set(comp_seed_global.tolist()))
            site_atom_rows.append(row)
    return df, sites, pd.DataFrame(site_atom_rows)


def main():
    args = parse_args()
    pdb_dir = Path(args.pdb_dir)
    dssp_dir = Path(args.dssp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    beta_codes = set(x.strip() for x in args.beta_codes.split(",") if x.strip())

    all_atom_tables = {}
    all_sites = []
    all_site_atom_tables = []

    for pdb_path in sorted(pdb_dir.glob(args.pdb_glob)):
        pdb_id = pdb_path.stem.lower()
        dssp_path = dssp_dir / f"{pdb_path.stem}{args.dssp_ext}"
        if not dssp_path.exists():
            raise FileNotFoundError(f"Missing DSSP for {pdb_path.name}: expected {dssp_path}")
        print(f"Building sites for {pdb_id}")
        records = load_atoms_with_sasa(pdb_path, dssp_path, pdb_id, beta_codes, args.probe_radius)
        atom_df, sites, site_atom_df = build_sites_for_pdb(
            records, args.cluster_cutoff, args.min_cluster_atoms, args.backbone_only
        )
        all_atom_tables[pdb_id] = atom_df
        all_sites.extend(sites)
        if len(site_atom_df):
            all_site_atom_tables.append(site_atom_df)
        print(f"  candidate sites: {len(sites)}")

    site_summary = pd.DataFrame([{k: v for k, v in s.items() if not k.endswith("uids")} for s in all_sites])
    site_atoms = pd.concat(all_site_atom_tables, ignore_index=True) if all_site_atom_tables else pd.DataFrame()

    with open(out_dir / "candidate_sites.pkl", "wb") as fh:
        pickle.dump({"atom_tables": all_atom_tables, "sites": all_sites, "params": vars(args)}, fh)
    site_summary.to_csv(out_dir / "site_summary.csv", index=False)
    site_atoms.to_csv(out_dir / "candidate_sites_atoms.csv", index=False)

    print("\nDone.")
    print(f"Total candidate sites: {len(all_sites)}")
    print(f"Wrote: {out_dir / 'candidate_sites.pkl'}")


if __name__ == "__main__":
    main()
