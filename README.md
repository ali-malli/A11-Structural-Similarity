# A11-Structural-Similarity
Used to evaluate whether A11-positive enzymes share common structural features

## Overview
Detect and compare solvent-exposed β-sheet surface patches in A11-positive β-glucosidases.

## Input
- PDB files
- DSSP files

## Method summary
1. Identify surface-exposed B/E DSSP atoms with SASA > 0 Å².
2. Cluster β-sheet surface atoms using 4 Å single-linkage.
3. Assign six-dimensional physicochemical atom vectors (PATTY).
4. Compute local environment vectors within 3.2 Å.
5. Align candidate sites using ICP.
6. Score site similarity using Tanimoto-based similarity.
7. Identify recurrent high-similarity β-sheet motifs.

## Reproducing the analysis
```bash
conda env create -f environment.yml
conda activate bgl-a11

python scripts/01_build_candidate_sites.py \
  --pdb_dir data/cleaned_pdbs \
  --dssp_dir data/dssp \
  --out_dir results/01_sites \
  --beta_codes B,E

python scripts/02_score_candidate_sites.py \
  --sites_pkl results/01_sites/candidate_sites.pkl \
  --out_dir results/02_scores_icp \
  --aligner icp \
  --score_atoms seed \
  --exclude_same_pdb

python scripts/03_cluster_and_write_pymol.py \
  --similarity_csv results/02_scores_icp/site_similarity_matrix.csv \
  --prepared_sites_pkl results/02_scores_icp/prepared_sites.pkl \
  --out_dir results/03_selected_sites \
  --pdb_dir data/cleaned_pdbs \
  --beta_codes B,E
```
## Outputs
describe each CSV/PML file.

## Citation
1. Yoshiike, Y.; Minai, R.; Matsuo, Y.; Chen, Y. R.; Kimura, T.; Takashima, A. Amyloid
oligomer conformation in a group of natively folded proteins. PLoS One 2008, 3 (9),
e3235. https://www.ncbi.nlm.nih.gov/pubmed/18800165
2. Gorelov, S.; Titov, A.; Tolicheva, O.; Konevega, A.; Shvetsov, A. DSSP in
GROMACS: Tool for Defining Secondary Structures of Proteins in Trajectories. J Chem
Inf Model 2024, 64 (9), 3593-3598. https://www.ncbi.nlm.nih.gov/pubmed/38655711
