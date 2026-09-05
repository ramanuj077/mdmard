# data/
Scenario data for the simulation. Loaded at env reset or curriculum stage init.

## Subdirectories
- terrain/          -- Heightmap files (.npy) for terrain-following (Phase 2)
- wind_profiles/    -- Wind field definitions (.yaml) for disturbance modelling
- scenarios/        -- Mission scenario configs (.yaml) defining start/goal/threats
- obstacles/        -- Obstacle layout files (.json) for curriculum stages

## File formats
YAML  : human-readable scenario/wind configs
JSON  : obstacle layouts (list of {x, y, radius, type})
NPY   : numpy arrays for terrain heightmaps