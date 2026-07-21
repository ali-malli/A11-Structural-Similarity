fetch 6t13_monomer_clean, async=0
remove solvent
hide everything
show cartoon
color lightblue, all
select a11_site_6t13_monomer_clean, (chain A and resi 233) or (chain A and resi 280) or (chain A and resi 281) or (chain A and resi 282)
color red, a11_site_6t13_monomer_clean
show sticks, a11_site_6t13_monomer_clean
orient
png 6t13_monomer_clean_figure6_like.png, dpi=300, ray=1
