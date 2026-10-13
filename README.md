# Classification Error

AppMax method for classification models

## Setup

```sh
# download the repository
git clone https://github.com/vitkolos/classification-error.git
cd classification-error

# install dependencies (using one of the following commands)
uv sync             # if uv is available
pip install -e .    # otherwise
```

## Contents

- `src/classerr` – implementation of the AppMax method
- `tests` covering most of the `classerr` implementation

## Executable modules

These modules can be run using `uv run -m` or `python -m`.

- `classerr` – the main module
- `classerr.visualization` – produces charts and tables
- `classerr.trainable` – trains a model
