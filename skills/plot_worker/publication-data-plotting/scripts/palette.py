"""Customer-validated NPG categorical colors. No layout or chart template.

Use only the colors needed, with stable category mappings across panels.
Sequential, diverging and modality-specific maps remain appropriate where
categorical colors would misrepresent the data. The skill owns that policy.
"""

# Derived from Yuan1z0825/nature-skills (Apache-2.0); see ../LICENSE.nature-skills.
# Includes CatMaster's complete-metadata and claim-relative evidence fixes.
PALETTE = {
    "main_red":        "#E64B35",  # RGB 230, 75, 53
    "cyan_blue":       "#4DBBD5",  # RGB 77, 187, 213
    "teal":            "#00A087",  # RGB 0, 160, 135
    "deep_blue":       "#3C5488",  # RGB 60, 84, 136
    "coral_red":       "#F39B7F",  # RGB 243, 155, 127
    "muted_light_blue":"#8491B4",  # RGB 132, 145, 180
    "mint_green":      "#91D1C2",  # RGB 145, 209, 194
    "deep_red":        "#DC0000",  # RGB 220, 0, 0
    "brown":           "#7E6148",  # RGB 126, 97, 72
    "sand":            "#B09C85",  # RGB 176, 156, 133
}

DEFAULT_COLORS = [
    PALETTE["main_red"],
    PALETTE["cyan_blue"],
    PALETTE["coral_red"],
    PALETTE["muted_light_blue"],
    PALETTE["teal"],
    PALETTE["deep_blue"],
    PALETTE["mint_green"],
    PALETTE["deep_red"],
    PALETTE["brown"],
    PALETTE["sand"],
]
