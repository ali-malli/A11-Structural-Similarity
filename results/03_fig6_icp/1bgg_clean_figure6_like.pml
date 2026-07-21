fetch 1bgg_clean, async=0
remove solvent
hide everything
show cartoon
color lightblue, all
select a11_site_1bgg_clean, (chain B and resi 225) or (chain B and resi 298) or (chain B and resi 299) or (chain B and resi 300)
color red, a11_site_1bgg_clean
show sticks, a11_site_1bgg_clean
orient
png 1bgg_clean_figure6_like.png, dpi=300, ray=1
