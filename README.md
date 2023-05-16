[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg "Apache 2.0 License")](http://www.apache.org/licenses/LICENSE-2.0.html)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![imports - isort](https://img.shields.io/badge/imports-isort-ef8336.svg)](https://github.com/pycqa/isort)
[![code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

# Basics on Quality Assurance in Python

The basics towards better ML software with Python and MLOps, by means of development best
practices. A work in progress with a continuous improvement mindset.

There are some basic system-wide prerequisites such as `python`, `venv`, `pip`,
and `git`. Next, we will install `pipx` at user level and use this tool
to install `pipenv` isolated from the general environment. Finally, `pyenv`
is installed to assure that any Python version requested is available and
easily switched to (independently of the system Python, it uses a collection
of scripts).

**NOTICE:** You can use other python virtualenv managers, like `conda`, but we recommend you use `pipenv`. For more information on how it works, consult its [usage](https://github.com/pypa/pipenv#usage).

![Pipenv and pre-commit flow diagram](docs/pipenv-pre-commit.jpg)
<figcaption><code>pipenv</code> and <code>pre-commit</code> flow diagram.</figcaption>

**NOTICE:** Using UNIX shell commands in a Debian GNU/Linux Bash shell.
Adapt accordingly your Operating System.

## Content

- [Basics on Quality Assurance in Python](#basics-on-quality-assurance-in-python)
  - [Content](#content)
  - [Prerequisites](#prerequisites)
  - [Quality Assurance](#quality-assurance)
    - [Code Formatting](#code-formatting)
    - [Code Style Enforcement](#code-style-enforcement)
    - [Type Checking](#type-checking)
    - [Security](#security)
    - [Testing](#testing)
    - [Git Hooks](#git-hooks)
    - [Wrap-up](#wrap-up)
  - [MLOps](#mlops)
    - [Prerequisites](#prerequisites-1)
    - [Create `.env` file](#create-env-file)
    - [Change `model.py` and `train.py` scripts](#change-modelpy-and-trainpy-scripts)
    - [MLproject file](#mlproject-file)

## Prerequisites

**[Linux](docs/README-Linux.md)**

**[macOS](docs/README-macOS.md)**

**[Windows](docs/README-Windows.md)**

## Quality Assurance

**NOTICE:** Make sure you've completed the [Prerequisites](#prerequisites) for
your operating system case!

If you want to setup quality assurance libraries and configure pre-commit, in your project, follow the steps below. If you don't need quality assurance in your project, skip this section.

### Code Formatting

`isort`, `black`

```shell
pipenv install isort black --dev
```

**NOTICE:** black and isort may have conflicts, since they both enforce styles
in the code (https://pycqa.github.io/isort/docs/configuration/black_compatibility.html).
To ensure isort follows the same style as black, add a line in the
configuration file as showed below:

`pyproject.toml`

```toml
[tool.isort]
profile = "black"
```

```shell
pipenv run isort .
pipenv run black .
```

### Code Style Enforcement

`flake8` + `pyproject.toml` support = `flake8p`

```shell
pipenv install Flake8-pyproject --dev
```

`pyproject.toml`

```toml
[tool.flake8]
max-line-length = 120
ignore = ["E203", "E266", "E501", "W503"]
max-complexity = 18
select = ["B", "C", "E", "F", "W", "T4"]
```

```shell
pipenv run flake8p .
```

### Type Checking

`mypy`

```shell
pipenv install mypy --dev
```

`pyproject.toml`

```toml
[tool.mypy]
files = "."
ignore_missing_imports = true
```

```shell
pipenv run mypy .
```

### Security

`bandit`, `pipenv check`

```shell
pipenv install bandit[toml] --dev
```

`pyproject.toml`

```toml
[tool.bandit]
assert_used.skips = "*/tests/*"
```

```shell
pipenv run bandit -c pyproject.toml -r .
pipenv check
```

### Testing

`pytest`, `pytest-cov`

```shell
pipenv install pytest pytest-cov --dev
```

`pyproject.toml`

```toml
[tool.pytest.ini_options]
addopts = "--cov --cov-fail-under=100"

[tool.coverage.run]
source = ["."]

[tool.coverage.report]
show_missing = true
omit = ["*/tests/*"]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:"
]
```

```shell
pipenv run pytest
```

### Git Hooks

`pre-commit`

Putting it all together, i.e., automating while distinguishing Git `commit`
fast-checking requirement from the Git `push` more time-consuming possible
actions such as `pytest` (including coverage) and `pipenv check`.

```shell
pipenv install pre-commit --dev
```

`.pre-commit-config.yaml`

**NOTICE:** The `pipenv check` and the `pytest` (including coverage) are
configured to run only on Git `push`!

```yaml
repos:
  - repo: local
    hooks:

      ### CODE FORMATTING

      - id: isort
        name: isort
        stages: [ commit ]
        language: system
        entry: pipenv run isort .
        types: [ python ]

      - id: black
        name: black
        stages: [ commit ]
        language: system
        entry: pipenv run black .
        types: [ python ]

      ### CODE STYLE ENFORCEMENT

      - id: flake8
        name: flake8
        stages: [ commit ]
        language: system
        entry: pipenv run flake8p .
        types: [ python ]

      ### TYPE CHECKING

      - id: mypy
        name: mypy
        stages: [ commit ]
        language: system
        entry: pipenv run mypy .
        types: [ python ]
        pass_filenames: false

      ### SECURITY

      - id: bandit
        name: bandit
        stages: [ commit ]
        language: system
        entry: pipenv run bandit -c pyproject.toml -r .
        types: [ python ]

      - id: check
        name: check
        stages: [ push ]
        language: system
        entry: pipenv check
        types: [ python ]

      ### TESTING

      - id: pytest
        name: pytest
        stages: [ push ]
        language: system
        entry: pipenv run pytest
        types: [ python ]
        pass_filenames: false
```

```shell
pipenv run pre-commit install -t pre-commit
pipenv run pre-commit install -t pre-push
```

It may be run without the Git `commit` hook triggering. This presents a 
useful way of testing the `.pre-commit-config.yaml` calls and also the
configuration in `pyproject.toml`:

```shell
pipenv run pre-commit run --all-files --hook-stage commit
pipenv run pre-commit run --all-files --hook-stage push
```

### Wrap-up

All the prerequisites must be accomplished (by following
the above instructions or by means of a previous project installation).
The project files for quality assurance must be in place by means of creating a new repository from this template (click the `Use this template` button).

```shell
pipenv install --dev
git init
pipenv run pre-commit install -t pre-commit
pipenv run pre-commit install -t pre-push
```

Later, you may add your local Git-based repository to a remote, such as,
[GitLab](https://gitlab.com).

```shell
git remote add origin <URL>
```

## MLOps

### Prerequisites

This section will show you how to setup a MLOps framework to easily run and track all your experiments. This is done with `Azure`, `Databricks` and `MLflow`. First, make sure you have the following prerequisites:

- Access to an Azure Databricks Workspace
- `databricks-cli` installed and configured (follow this [guide](https://docs.databricks.com/dev-tools/cli/index.html))
- `mlflow` installed (for info, check their [docs](https://mlflow.org/docs/latest/index.html))

### Create `.env` file

Be sure to create a `.env` file on your local project which will create the directory paths and other variables in your system. 

```shell
ROOT_DIR = "/path/to/your/project-dir"
RAW_DATA_DIR = "${ROOT_DIR}/data/raw"
PROCESSED_DATA_DIR = "${ROOT_DIR}/data/processed"
RESULTS_DIR = "${ROOT_DIR}/results"
MODELS_DIR = "${ROOT_DIR}/models"

LOGS_DIR = "${ROOT_DIR}/logs"
LOGGER_LEVEL = "INFO"

EXP_NAME = "/Users/<your-databricks-user-email>/<name-of-the-experiment>"

DATABRICKS_HOST = "<your-databricks-workspace-url>"
DATABRICKS_TOKEN = "<your-databricks-access-token>"

GIT_USER = "<your-github-username>"
GIT_TOKEN = "<your-github-access-token>"
GIT_URI = "github.com/AxiansML/<your-github-repo-name>.git"
```

### Change `model.py` and `train.py` scripts

The **[model.py](src/model.py)** script is where the model and data modules are defined. If you´re using `pytorch` in your project, check out how to do this using `pytorch_lightning`, by following the links below. Otherwise, adapt this code to your ML library (`tensorflow` or other).
- [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html)
- [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html)

The **[train.py](src/train.py)** script is where you load your model and dataset, input the hyperparameters, train the model and log the results to `mlflow`.

### MLproject file

The **[MLproject](MLproject)** file is where you define the entry points of your `mlflow` project. In this example, you only have the **[train](src/train.py)** entry point but you can create more if you wish, by creating additional scripts (*e.g.,* `predict.py`) and defining the input parameters and shell command, in the **[MLproject](MLproject)** file.