# Running TRACE on Windows

This guide covers Windows-specific setup steps. Follow the [main README](../README.md) for general context, but use the commands here instead where they differ.

---

## Prerequisites

- **Git for Windows** — download from [git-scm.com](https://git-scm.com/). During installation, choose "Git from the command line and also from 3rd-party software" so that `sh.exe` is available on your PATH (see [step 5](#5-fix-the-path-for-sh) below).
- **R** — download from [cran.r-project.org](https://cran.r-project.org/). During installation, tick the option to add R to your PATH.

---

## Setup steps

All commands are run from **PowerShell** unless noted otherwise. Open it by pressing `Win + R`, typing `powershell`, and pressing Enter.

### 1. Clone the repository

```powershell
git clone --recurse-submodules https://github.com/M-DICE/trace.git
cd trace
```

### 2. Install uv

First allow PowerShell to run downloaded scripts (required once per machine):

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then install uv:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell after this step so that the `uv` command is available.

### 3. Install dependencies

```powershell
uv sync
```

> You may see a warning: `Failed to hardlink files; falling back to full copy.` This is harmless and can be ignored.

### 4. Generate the critical value table

```powershell
uv run Rscript scripts/export_crit_val_table.R
```

If this fails with `Rscript not found`, R is not on your PATH. Re-run the R installer and tick "Add R to PATH", or add `C:\Program Files\R\R-x.x.x\bin` manually (see [adding to PATH](#adding-a-directory-to-path) below).

### 5. Fix the PATH for `sh` (may be required)

`uv run trace-sim` shells out to `sh.exe`, which is not available by default on Windows. Git for Windows ships it, but its `bin` directory is not always on the system PATH.

To add it:

1. Press `Win + R`, type `SystemPropertiesAdvanced`, press Enter.
2. Click **Environment Variables…**
3. Under **System variables**, select `Path` and click **Edit**.
4. Click **New** and add: `C:\Program Files\Git\bin` (adjust if Git is installed elsewhere).
5. Click OK on all dialogs, then **restart PowerShell**.

To verify: `sh --version`

### 6. Install Rtools (may be required)

Some R packages and build steps require Rtools. Install the version that matches your R version from [cran.r-project.org/bin/windows/Rtools](https://cran.r-project.org/bin/windows/Rtools/).

After installing, add the Rtools `usr\bin` directory to your **User variables** PATH (same process as above). The path follows the pattern `C:\RtoolsNN\usr\bin`, where `NN` matches your R version — for example:

| R version | Rtools path |
|-----------|-------------|
| 4.3.x | `C:\Rtools43\usr\bin` |
| 4.4.x | `C:\Rtools44\usr\bin` |
| 4.5.x | `C:\Rtools45\usr\bin` |

To verify: `make --version`

### 7. Generate pre-computed results (optional)

```powershell
uv run trace-sim all --quick
```

### 8. Start the app

```powershell
uv run streamlit run Home.py
```

Then open **http://localhost:8501** in your browser.

---

## Adding a directory to PATH

1. Press `Win + R`, type `SystemPropertiesAdvanced`, press Enter.
2. Click **Environment Variables…**
3. To affect all users: edit `Path` under **System variables**.  
   To affect only your account: edit `Path` under **User variables**.
4. Click **Edit → New**, paste the directory path, and click OK.
5. Restart any open terminals for the change to take effect.
