# models/
Saved model weights and checkpoints. Never commit large weight files to git.
Add models/ to .gitignore and use DVC or cloud storage for weights.

## Subdirectories
- checkpoints/  -- Periodic saves during training (every N steps)
- best/         -- Best model by eval reward (overwritten each eval cycle)

## File format
SB3 saves as .zip files containing policy weights + optimizer state.