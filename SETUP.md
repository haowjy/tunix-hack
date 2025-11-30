# Setup Guide with uv

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management and virtual environment handling.

## Prerequisites

Install `uv` if you haven't already:

```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

## Setup Local Environment

### 1. Create Virtual Environment

`uv` will automatically create a `.venv` directory in your project:

```bash
uv venv
```

This creates a virtual environment at `.venv/` using the Python version specified in `.python-version` (or the default if not specified).

**Note**: With `uv`, you don't need to activate the virtual environment! `uv` automatically detects and uses `.venv` in the current directory or parent directories.

### 2. Sync Dependencies

Install all dependencies from `pyproject.toml`:

```bash
uv sync
```

This will:
- Create the virtual environment if it doesn't exist
- Install all dependencies listed in `pyproject.toml`
- Install optional dev dependencies (jupyter, wandb, etc.)

To install only main dependencies (without dev tools):

```bash
uv sync --no-dev
```

### 3. Run Commands in the Environment

You can run Python scripts directly with `uv run`:

```bash
# Run a script
uv run python scripts/step0_env_sanity.py

# Run Python interactively
uv run python

# Run any command
uv run pytest
```

### 4. Add New Dependencies

To add a new dependency:

```bash
uv add package-name
```

This automatically updates `pyproject.toml` and installs the package.

To add a dev dependency:

```bash
uv add --dev package-name
```

Or manually edit `pyproject.toml` and run `uv sync`.

## Virtual Environment Details

- **Location**: `.venv/` (in project root)
- **Activation**: Not required! `uv` automatically uses `.venv` when present
- **Manual activation** (if needed): `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)

## Workflow

1. **First time setup**:
   ```bash
   uv venv
   uv sync
   ```

2. **Daily development**:
   ```bash
   # Just run commands - uv handles the environment automatically
   uv run python scripts/step0_env_sanity.py
   ```

3. **Adding dependencies**:
   ```bash
   uv add new-package
   ```

4. **Updating dependencies**:
   ```bash
   uv sync  # Re-syncs based on pyproject.toml
   ```

## Benefits of uv

- **10-100x faster** than pip
- **Automatic virtual environment detection** - no need to activate
- **Better dependency resolution** - faster and more reliable
- **Drop-in replacement** for pip commands

## Troubleshooting

If `uv` doesn't find your virtual environment:

1. Make sure `.venv/` exists in the project root
2. Or set `VIRTUAL_ENV` environment variable: `export VIRTUAL_ENV=.venv`
3. Or explicitly specify: `uv run --python .venv/bin/python script.py`

