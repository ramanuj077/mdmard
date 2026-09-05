# logs/
All runtime logs are written here. Never commit this directory.

## Subdirectories
- tensorboard/  -- TensorBoard event files (one subdir per experiment)
- training/     -- Text logs from training runs (timestamped .log files)

## Viewing TensorBoard
    tensorboard --logdir logs/tensorboard/