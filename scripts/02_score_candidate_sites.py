#!/usr/bin/env python3
"""
Score candidate A11 sites using six-dimensional PATTY-style vectors,
local physicochemical environments, Tanimoto coefficients, and a patch-alignment strategies.

Output:
  site_similarity_matrix.csv
  site_pair_scores.csv

This implements:
  C_i = sum_j v_j * (1 - d_ij / dc), for d_ij <= dc
where v_j is a six-dimensional atom-type vector.
"""

import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance_matrix
from scipy.optimize import linear_sum_assignment


# Six A11 atom classes: cation, anion, donor, acceptor, hydrophobic, none
CAT, ANI, DON, ACC, HYD, NON = range(6)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sites_pkl", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dc", type=float, default=3.2)
    p.add_argument("--aligner", choices=["none", "icp", "triplet"], default="triplet")
    p.add_argument("--match_dist", type=float, default=1.5)
    p.add_argument("--triplet_tol", type=float, default=0.75)
    p.add_argument("--max_triplets", type=int, default=4000, help="Random cap per site for triplet aligner; 0=no cap")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def patty6_vector(resname, atom_name, element):
    """Protein-specific implementation of A11's six PATTY classes.

    For donor+acceptor atoms, return 0.5 donor + 0.5 acceptor.
    Charged atoms are typed as cation/anion rather than additionally donor/acceptor,
    matching the idea of one broad physicochemical class per atom except polar.
    """
    resname = resname.upper().strip()
    atom_name = atom_name.upper().strip()
    element = element.upper().strip()
    v = np.zeros(6, dtype=float)

    # Backbone atoms
    if atom_name == "N":
        v[DON] = 1.0; return v
    if atom_name == "O" or atom_name == "OXT":
        v[ACC] = 1.0; return v

    # Charged sidechain atoms
    if resname in {"ASP", "ASH"} and atom_name in {"OD1", "OD2"}:
        v[ANI] = 1.0; return v
    if resname in {"GLU", "GLH"} and atom_name in {"OE1", "OE2"}:
        v[ANI] = 1.0; return v
    if resname == "LYS" and atom_name == "NZ":
        v[CAT] = 1.0; return v
    if resname == "ARG" and atom_name in {"NE", "NH1", "NH2"}:
        v[CAT] = 1.0; return v

    # Histidine: ambiguous without hydrogens. Treat ring nitrogens as polar.
    if resname in {"HIS", "HID", "HIE", "HIP"} and atom_name in {"ND1", "NE2"}:
        v[DON] = 0.5; v[ACC] = 0.5; return v

    # Sidechain donor/acceptor/polar atoms
    if resname in {"SER", "THR"} and atom_name in {"OG", "OG1"}:
        v[DON] = 0.5; v[ACC] = 0.5; return v
    if resname == "TYR" and atom_name == "OH":
        v[DON] = 0.5; v[ACC] = 0.5; return v
    if resname == "CYS" and atom_name == "SG":
        v[DON] = 0.5; v[ACC] = 0.5; return v
    if resname == "ASN" and atom_name == "OD1":
        v[ACC] = 1.0; return v
    if resname == "ASN" and atom_name == "ND2":
        v[DON] = 1.0; return v
    if resname == "GLN" and atom_name == "OE1":
        v[ACC] = 1.0; return v
    if resname == "GLN" and atom_name == "NE2":
        v[DON] = 1.0; return v
    if resname == "TRP" and atom_name == "NE1":
        v[DON] = 1.0; return v
    if resname == "MET" and atom_name == "SD":
        v[HYD] = 1.0; return v

    # Generic elements after residue-specific rules
    if element == "O":
        v[ACC] = 1.0; return v
    if element == "N":
        v[DON] = 1.0; return v
    if element in {"C", "S", "SE", "F", "CL", "BR", "I"}:
        v[HYD] = 1.0; return v

    v[NON] = 1.0
    return v


def primary_type(v):
    return int(np.argmax(v))


def compatible(v1, v2):
    # Same class, or shared donor/acceptor fractional contribution.
    return float(np.dot(v1, v2)) > 0.0


def tanimoto(c1, c2):
    dot = float(np.dot(c1, c2))
    denom = float(np.dot(c1, c1) + np.dot(c2, c2) - dot)
    if denom <= 0:
        return 0.0
    return dot / denom


def kabsch(P, Q):
    """Return R,t that maps P onto Q: P @ R + t."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Q.mean(axis=0) - P.mean(axis=0) @ R
    return R, t


def transform(coords, R, t):
    return coords @ R + t


def assign_pairs(coords_a, coords_b, vecs_a, vecs_b, match_dist):
    """Greedy-ish optimal assignment within compatible atom types."""
    if len(coords_a) == 0 or len(coords_b) == 0:
        return []
    D = distance_matrix(coords_a, coords_b)
    big = 1e6
    cost = D.copy()
    for i in range(len(coords_a)):
        for j in range(len(coords_b)):
            if D[i, j] > match_dist or not compatible(vecs_a[i], vecs_b[j]):
                cost[i, j] = big
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < big]
    return pairs


def score_aligned(coords_a, coords_b, env_a, env_b, vecs_a, vecs_b, match_dist):
    pairs = assign_pairs(coords_a, coords_b, vecs_a, vecs_b, match_dist)
    if not pairs:
        return 0.0, 0
    s = sum(tanimoto(env_a[i], env_b[j]) for i, j in pairs)
    return s / min(len(coords_a), len(coords_b)), len(pairs)


def env_vectors(site_coords, all_surface_coords, all_surface_vecs, dc):
    tree = cKDTree(all_surface_coords)
    env = np.zeros((len(site_coords), 6), dtype=float)
    for i, xyz in enumerate(site_coords):
        hits = tree.query_ball_point(xyz, r=dc)
        for j in hits:
            d = np.linalg.norm(xyz - all_surface_coords[j])
            if d <= dc:
                env[i] += all_surface_vecs[j] * (1.0 - d / dc)
    return env


def prepare_sites(data, dc):
    atom_tables = data["atom_tables"]
    sites_raw = data["sites"]

    prepared = []
    # Precompute atom vectors and all-surface arrays per PDB
    pdb_pre = {}
    for pdb_id, df in atom_tables.items():
        df = df.copy()
        vecs = np.vstack([patty6_vector(r.resname, r.atom_name, r.element) for r in df.itertuples()])
        df["primary_type"] = [primary_type(v) for v in vecs]
        surface_mask = df["is_surface"].to_numpy(bool)
        pdb_pre[pdb_id] = {
            "df": df,
            "vecs": vecs,
            "surface_coords": df.loc[surface_mask, ["x", "y", "z"]].to_numpy(float),
            "surface_vecs": vecs[surface_mask],
        }

    for s in sites_raw:
        pdb_id = s["pdb_id"]
        pre = pdb_pre[pdb_id]
        df = pre["df"]
        uid_to_idx = {int(uid): i for i, uid in enumerate(df["atom_uid"].astype(int).tolist())}
        idxs = [uid_to_idx[int(uid)] for uid in s["expanded_atom_uids"]]
        coords = df.iloc[idxs][["x", "y", "z"]].to_numpy(float)
        vecs = pre["vecs"][idxs]
        env = env_vectors(coords, pre["surface_coords"], pre["surface_vecs"], dc)
        atom_rows = df.iloc[idxs].copy()
        prepared.append({
            "site_id": s["site_id"],
            "pdb_id": pdb_id,
            "coords": coords,
            "vecs": vecs,
            "env": env,
            "atom_df": atom_rows,
        })
    return prepared


def center_aligner(site_a, site_b, match_dist):
    A = site_a["coords"] - site_a["coords"].mean(axis=0)
    B = site_b["coords"] - site_b["coords"].mean(axis=0)
    return score_aligned(A, B, site_a["env"], site_b["env"], site_a["vecs"], site_b["vecs"], match_dist)


def icp_aligner(site_a, site_b, match_dist, n_iter=20):
    A0 = site_a["coords"]
    B = site_b["coords"]
    A = A0 - A0.mean(axis=0) + B.mean(axis=0)
    R_total = np.eye(3)
    t_total = B.mean(axis=0) - A0.mean(axis=0)
    best = score_aligned(A, B, site_a["env"], site_b["env"], site_a["vecs"], site_b["vecs"], match_dist)
    for _ in range(n_iter):
        pairs = assign_pairs(A, B, site_a["vecs"], site_b["vecs"], match_dist * 2.0)
        if len(pairs) < 3:
            break
        P = A0[[i for i, _ in pairs]]
        Q = B[[j for _, j in pairs]]
        R, t = kabsch(P, Q)
        A = transform(A0, R, t)
        sc = score_aligned(A, B, site_a["env"], site_b["env"], site_a["vecs"], site_b["vecs"], match_dist)
        if sc[0] > best[0]:
            best = sc
            R_total, t_total = R, t
    return best


def triplets_for_site(coords, vecs, max_triplets, rng):
    n = len(coords)
    trips = []
    for tri in itertools.combinations(range(n), 3):
        vtypes = [primary_type(vecs[i]) for i in tri]
        d = sorted([np.linalg.norm(coords[tri[0]] - coords[tri[1]]),
                    np.linalg.norm(coords[tri[0]] - coords[tri[2]]),
                    np.linalg.norm(coords[tri[1]] - coords[tri[2]])])
        trips.append((tri, tuple(vtypes), np.array(d)))
    if max_triplets and len(trips) > max_triplets:
        idx = rng.choice(len(trips), size=max_triplets, replace=False)
        trips = [trips[i] for i in idx]
    return trips


def triplet_aligner(site_a, site_b, match_dist, triplet_tol, max_triplets, rng):
    A0 = site_a["coords"]
    B = site_b["coords"]
    if len(A0) < 3 or len(B) < 3:
        return center_aligner(site_a, site_b, match_dist)
    trips_a = triplets_for_site(A0, site_a["vecs"], max_triplets, rng)
    trips_b = triplets_for_site(B, site_b["vecs"], max_triplets, rng)
    best = (0.0, 0)
    for ta, types_a, dist_a in trips_a:
        sorted_types_a = sorted(types_a)
        for tb, types_b, dist_b in trips_b:
            if sorted(types_b) != sorted_types_a:
                continue
            if np.max(np.abs(dist_a - dist_b)) > triplet_tol:
                continue
            for perm in itertools.permutations(tb):
                ok = all(compatible(site_a["vecs"][ia], site_b["vecs"][ib]) for ia, ib in zip(ta, perm))
                if not ok:
                    continue
                R, t = kabsch(A0[list(ta)], B[list(perm)])
                A = transform(A0, R, t)
                sc = score_aligned(A, B, site_a["env"], site_b["env"], site_a["vecs"], site_b["vecs"], match_dist)
                if sc[0] > best[0]:
                    best = sc
    return best


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(args.sites_pkl, "rb") as fh:
        data = pickle.load(fh)
    sites = prepare_sites(data, args.dc)
    n = len(sites)
    print(f"Scoring {n} candidate sites with aligner={args.aligner}")

    S = np.eye(n, dtype=float)
    pair_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if args.aligner == "none":
                score, nmatch = center_aligner(sites[i], sites[j], args.match_dist)
            elif args.aligner == "icp":
                score, nmatch = icp_aligner(sites[i], sites[j], args.match_dist)
            else:
                score, nmatch = triplet_aligner(sites[i], sites[j], args.match_dist, args.triplet_tol, args.max_triplets, rng)
            S[i, j] = S[j, i] = score
            pair_rows.append({
                "site_i": sites[i]["site_id"], "pdb_i": sites[i]["pdb_id"],
                "site_j": sites[j]["site_id"], "pdb_j": sites[j]["pdb_id"],
                "score": score, "n_matched_atoms": nmatch,
            })
        print(f"  done {i+1}/{n}")

    site_ids = [s["site_id"] for s in sites]
    sim_df = pd.DataFrame(S, index=site_ids, columns=site_ids)
    sim_df.to_csv(out_dir / "site_similarity_matrix.csv")
    pd.DataFrame(pair_rows).to_csv(out_dir / "site_pair_scores.csv", index=False)
  
    meta = pd.DataFrame([{
        "site_id": s["site_id"], "pdb_id": s["pdb_id"], "n_atoms": len(s["coords"])
    } for s in sites])
    meta.to_csv(out_dir / "scored_site_metadata.csv", index=False)
    with open(out_dir / "prepared_sites.pkl", "wb") as fh:
        pickle.dump(sites, fh)
    print("Done.")


if __name__ == "__main__":
    main()
