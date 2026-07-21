#!/usr/bin/env python3
"""
Cluster high-similarity sites and write Figure-6-like PyMOL selections.

Because the A11 paper does not report the exact cutoff/linkage used to get
23 high-similarity sites, this script sweeps score thresholds and chooses the
threshold that gives a selected set closest to --target_n.

Selected high-similarity sites are nodes in graph components that:
  - have edges score >= threshold
  - contain sites from at least --min_pdbs structures
  - have at least --min_component_size nodes

Then one display site per PDB is chosen as the site with the highest average
similarity to the selected high-similarity set, excluding same-PDB sites by default.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--similarity_csv", required=True)
    p.add_argument("--prepared_sites_pkl", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--target_n", type=int, default=23)
    p.add_argument("--min_pdbs", type=int, default=3)
    p.add_argument("--min_component_size", type=int, default=2)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--exclude_same_pdb", action="store_true")
    return p.parse_args()


def high_similarity_nodes(S, pdb_ids, threshold, min_pdbs, min_component_size):
    n = S.shape[0]
    edges = np.argwhere(np.triu(S >= threshold, k=1))
    if len(edges) == 0:
        return set(), []
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.int8)
    graph = csr_matrix((data, (row, col)), shape=(n, n))
    ncomp, labels = connected_components(graph, directed=False)
    selected = set()
    comps = []
    for c in range(ncomp):
        members = np.flatnonzero(labels == c).tolist()
        if len(members) < min_component_size:
            continue
        distinct_pdbs = {pdb_ids[i] for i in members}
        if len(distinct_pdbs) >= min_pdbs:
            selected.update(members)
            comps.append(members)
    return selected, comps


def choose_threshold(S, pdb_ids, target_n, min_pdbs, min_component_size):
    vals = S[np.triu_indices_from(S, k=1)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        raise ValueError("No positive pairwise similarities found.")
    # Sweep unique-ish thresholds from high to low.
    qs = np.linspace(0.99, 0.50, 200)
    thresholds = sorted(set(np.quantile(vals, qs)), reverse=True)
    best = None
    for th in thresholds:
        selected, comps = high_similarity_nodes(S, pdb_ids, th, min_pdbs, min_component_size)
        diff = abs(len(selected) - target_n)
        # Prefer exact or closer; if tied prefer higher threshold.
        candidate = (diff, -th, th, selected, comps)
        if best is None or candidate < best:
            best = candidate
    return best[2], best[3], best[4]


def residue_selection(atom_df):
    residues = []
    for r in atom_df.itertuples():
        resi = str(int(r.resseq)) + (str(r.icode).strip() if str(r.icode).strip() not in {"", "nan"} else "")
        residues.append((str(r.chain_id), resi))
    # Unique, stable
    seen = set(); out = []
    for item in residues:
        if item not in seen:
            seen.add(item); out.append(item)
    parts = []
    for chain, resi in out:
        parts.append(f"(chain {chain} and resi {resi})")
    return " or ".join(parts), out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_df = pd.read_csv(args.similarity_csv, index_col=0)
    site_ids = sim_df.index.tolist()
    S = sim_df.to_numpy(float)

    with open(args.prepared_sites_pkl, "rb") as fh:
        sites = pickle.load(fh)
    site_by_id = {s["site_id"]: s for s in sites}
    pdb_ids = [site_by_id[sid]["pdb_id"] for sid in site_ids]

    if args.threshold is None:
        threshold, selected_idx, comps = choose_threshold(
            S, pdb_ids, args.target_n, args.min_pdbs, args.min_component_size
        )
    else:
        threshold = args.threshold
        selected_idx, comps = high_similarity_nodes(S, pdb_ids, threshold, args.min_pdbs, args.min_component_size)

    selected_idx = sorted(selected_idx)
    selected_site_ids = [site_ids[i] for i in selected_idx]
    print(f"Threshold: {threshold:.6f}")
    print(f"High-similarity sites selected: {len(selected_site_ids)}")

    pd.DataFrame({
        "site_id": selected_site_ids,
        "pdb_id": [pdb_ids[i] for i in selected_idx],
    }).to_csv(out_dir / "high_similarity_sites.csv", index=False)

    # Pick one site per PDB, by average similarity to the high-similarity set.
    display_rows = []
    selected_set = set(selected_idx)
    for pdb in sorted(set(pdb_ids)):
        candidates = [i for i, p in enumerate(pdb_ids) if p == pdb]
        best = None
        for i in candidates:
            refs = list(selected_set)
            if args.exclude_same_pdb:
                refs = [j for j in refs if pdb_ids[j] != pdb]
            if not refs:
                avg = 0.0
            else:
                avg = float(np.mean([S[i, j] for j in refs if j != i]))
            cand = (avg, i)
            if best is None or cand > best:
                best = cand
        display_rows.append({"pdb_id": pdb, "site_id": site_ids[best[1]], "avg_similarity_to_high_set": best[0]})
    display_df = pd.DataFrame(display_rows)
    display_df.to_csv(out_dir / "figure6_display_sites.csv", index=False)

    # Write per-PDB PyMOL scripts and one combined script.
    combined = []
    for row in display_df.itertuples():
        site = site_by_id[row.site_id]
        atom_df = site["atom_df"]
        sel, residues = residue_selection(atom_df)
        pml = []
        pml.append(f"fetch {row.pdb_id}, async=0")
        pml.append("remove solvent")
        pml.append("hide everything")
        pml.append("show cartoon")
        pml.append("color lightblue, all")
        pml.append(f"select a11_site_{row.pdb_id}, {sel}")
        pml.append(f"color red, a11_site_{row.pdb_id}")
        pml.append(f"show sticks, a11_site_{row.pdb_id}")
        pml.append("orient")
        pml.append(f"png {row.pdb_id}_figure6_like.png, dpi=300, ray=1")
        pml_text = "\n".join(pml) + "\n"
        (out_dir / f"{row.pdb_id}_figure6_like.pml").write_text(pml_text)
        combined.extend(pml)
        combined.append("delete all")

        pd.DataFrame(residues, columns=["chain", "resi"]).to_csv(out_dir / f"{row.pdb_id}_selected_residues.csv", index=False)

    (out_dir / "all_figure6_like.pml").write_text("\n".join(combined) + "\n")
    print(f"Wrote PyMOL scripts to {out_dir}")


if __name__ == "__main__":
    main()
