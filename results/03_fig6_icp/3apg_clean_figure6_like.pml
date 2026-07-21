fetch 3apg_clean, async=0
remove solvent
hide everything
show cartoon
color lightblue, all
select a11_site_3apg_clean, (chain C and resi 209) or (chain C and resi 264) or (chain C and resi 265) or (chain C and resi 266) or (chain C and resi 305)
color red, a11_site_3apg_clean
show sticks, a11_site_3apg_clean
orient
png 3apg_clean_figure6_like.png, dpi=300, ray=1
