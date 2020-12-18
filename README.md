# Data-Utils for Competition

This repository is contained useful feature and preprocess modules

For my note, 

## Env

- WSL2

- CUDA 11.0

## Ready for Using

Build Docker Container

```
# Build Container
docker build -t my_env .
```

Run Container

```
# Run Container
docker run -it --rm --gpus all -v $(pwd):/workspace my_env bash
# Install Library on RapidsAI
sh setup.sh
```
