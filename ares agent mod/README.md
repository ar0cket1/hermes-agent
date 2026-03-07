# Ares Agent Mod

This folder is the drop-in payload for the Ares visual mod.

What it contains:
- `mod.py`: Ares palette, branding copy, prompt/spinner assets, and masthead data.
- `pixel_art_large-2.png`, `pixel_art_large.png`, `pixel_art_small.png`: banner art sources.
- `spartan_emblem_pixel_art_transparent.json`: fallback emblem art.
- `ascii-art.png`, `ares_template.png`: extra source assets kept with the mod.

How it is used:
- Hermes reads this folder from the repo root.
- The main CLI keeps only thin loader/render hooks.
- If this folder is present, the Ares payload is available to the CLI skin.
