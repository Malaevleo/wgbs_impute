# Push the changes to GitHub

Run these commands from the `wgbs_impute` repository, not its parent analysis directory. The existing `origin` should point to your GitHub repository; verify it before pushing.

```bash
git status --short
git remote -v
git switch -c codex/production-hardening
python -m pip install '.[ml,dev]'
ruff check .
python -m unittest discover -s tests -v
python -m build
docker build -t wgbs-impute:0.2.0 .
docker run --rm --network none --read-only --tmpfs /tmp:rw,nosuid,size=256m \
  wgbs-impute:0.2.0 --demo --out /tmp/demo
git diff --check
git diff
```

Review every changed/untracked file, then stage the release files explicitly:

```bash
git add .dockerignore .gitignore .github/workflows/ci.yml Dockerfile \
  README.md RELEASE_NOTES.md PUBLISHING.md pyproject.toml MANIFEST.in \
  requirements.txt requirements-docker.lock \
  __init__.py __main__.py blup.py cli.py docker_entry.py impute.py io.py \
  optional.py panel.py thinning.py validation.py \
  tests/test_impute.py tests/test_production.py
git add -u DOCKER_PROMPT.md
git diff --cached --stat
git commit -m "Harden wgbs_impute packaging, data validation, and Docker workflow"
git push -u origin codex/production-hardening
```

Any pre-existing change to `tests/__init__.py` remains unstaged by these commands; review it separately and include it only if intended. Generated data, environments, cache directories, and build outputs are ignored.

Open a pull request from `codex/production-hardening` into the repository's default branch using GitHub. Wait for the new CI checks, review the diff, and merge. With GitHub CLI installed and authenticated, `gh pr create --fill` can create the pull request after the push.

If HTTPS authentication is requested, use your configured GitHub credential manager, `gh auth login`, or a GitHub token; GitHub account passwords do not authenticate Git pushes. No changes are pushed automatically by the local hardening work.
