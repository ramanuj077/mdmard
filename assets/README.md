# assets/
Static assets for the MDMARD simulation renderer.

## Subdirectories
- images/   -- Background images, map tiles, explosion sprites
- icons/    -- Aircraft icons (blue/red force), waypoint markers
- fonts/    -- Fonts for tactical HUD (recommend: JetBrains Mono, Orbitron)

## Usage
Loaded by rendering/renderer.py at startup.
All assets are optional -- renderer falls back to shapes if not found.