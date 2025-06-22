# Installation Guide

**IMPORTANT: To copy the raw markdown for this document, please look for a "Copy Code" or "View Raw" button/option associated with this entire block on the right side of your screen. This will give you the literal text, including all markdown formatting like `#` and ````. Then paste it directly into your `installation.md` file.**

This guide will help you set up your environment to run the **WindGym** simulation environment and its agents.

## 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

* **Git**: For cloning the WindGym and its dependency repositories.
   * [Download Git](https://git-scm.com/downloads)
* **Pixi**: A modern package manager for Python and other languages. WindGym uses Pixi to manage its Python dependencies in an isolated environment.
   * [Install Pixi](https://pixi.sh/latest/)
* **Python 3.7 to 3.11**: Pixi will help manage this, but ensure you don't have conflicting global Python versions if possible.

## 2. Clone the Repositories

First, clone the necessary repositories. Navigate to your desired working directory and run the following commands:

```bash
git clone https://gitlab.windenergy.dtu.dk/sys/windgym/WindGym.git
git clone https://github.com/kilojoules/WindGym-Zoo.git
```

This will create two new directories: `WindGym` (which contains the main WindGym Python package) and `WindGym-Zoo` (a dependency used by some WindGym scripts). Your working directory structure will look similar to this:

```
your-working-directory/
├── WindGym/
│   ├── WindGym/ (The core WindGym Python package)
│   ├── pyproject.toml
│   ├── docusaurus-site/ (Docusaurus documentation project)
│   └── ...
└── WindGym-Zoo/
    └── ...
```

## 3. Set Up the WindGym Python Environment (using Pixi)

Navigate into the `WindGym` directory. This directory contains the `pyproject.toml` file, which defines all the necessary Python dependencies for WindGym.

```bash
cd WindGym
```

Now, use `pixi` to create and activate an isolated Python environment and install all dependencies:

```bash
pixi install
```

* `pixi install`: This command reads your `pyproject.toml` file, resolves all specified Python dependencies (including `WindGym` itself in editable mode, `dynamiks` from GitLab, `gymnasium`, `stable_baselines3`, `pytest`, `nbconvert`, `xarray`, etc.), and creates a dedicated virtual environment. This process may take a few minutes on the first run, depending on your internet connection and system resources.

After `pixi install` completes, you can activate the environment and run Python commands:

```bash
pixi shell
```

* `pixi shell`: This command activates the `WindGym` Python environment within your current terminal session. You should see `(WindGym)` (or a similar prefix) in your terminal prompt, indicating that the environment is active and all WindGym's Python dependencies are available.

**Note:** You will need to run `pixi shell` whenever you open a new terminal session and want to execute WindGym code. Alternatively, you can directly run tasks defined in `pyproject.toml` using `pixi run <task_name>` (e.g., `pixi run test` to run tests).

## 4. Verify Your Installation

Once the environment is set up, you can quickly verify the installation by running a simple WindGym script or one of your tests. From within the `WindGym` directory (after running `pixi shell`), try importing `WindFarmEnv`:

```python
python -c "from WindGym.Wind_Farm_Env import WindFarmEnv; print('WindGym installed successfully!')"
```

If you see "WindGym installed successfully!", your Python environment is ready to use for running simulations and developing agents!

## **Optional: For Developers & Documentation Contributors**

If you intend to contribute to the WindGym documentation or want to build the Docusaurus site locally, you will need Node.js and npm to set up the `docusaurus-site` project. Please refer to the `WindGym/docusaurus-site/README.md` file within the cloned `WindGym` repository for specific instructions on setting up the Node.js environment and building the documentation site.
