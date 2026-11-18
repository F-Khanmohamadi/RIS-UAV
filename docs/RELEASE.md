# Publishing this project on GitHub

The repository is ready to push as it is. Generated data (`work/`, `outputs/`) and the virtual environment (`.venv/`)
are excluded by `.gitignore`. The largest committed files are the checkpoints (3.8 MB each), well below GitHub's 100 MB
limit, so Git LFS is not needed.

Before the first push:

1. Choose a licence and add a `LICENSE` file (README Section 10).
2. Add the citation of the published article to README Section 10 (optionally a `CITATION.cff`).
3. Run `python -m risuav test --fast` and `python -m risuav quick`.

Push (after creating an empty repository on GitHub, without README or licence):

```powershell
git init -b main
git add .
git commit -m "RIS-UAV implementation, published checkpoints and results"
git remote add origin https://github.com/<account>/RIS-UAV.git
git push -u origin main
```

`git status` before the commit should list no files under `work/`, `outputs/` or `.venv/`.

A clone of the repository is the same project as the ZIP: open the cloned folder in VS Code and follow README Section 2.
