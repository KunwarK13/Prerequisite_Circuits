# Development and releases

Use short feature branches and pull requests for public software work. Keep
unpublished hypotheses, exploratory runs, and private data in a separate private
research repository. GitHub visibility applies to the repository, not individual
branches. See [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow).

## Move research into the package

1. Develop and validate the research in the private workspace. Preserve the raw
   records, controls, and execution provenance.
2. Start a feature branch from the public repository's current `main`. Bring over
   the reviewed implementation and relevant tests, with small reference examples.
   Do not merge the private repository's complete history into the public branch.
3. Open a pull request, run CI, and review the API, documentation, evidence scope,
   and dependencies. Large records belong in versioned assets.
4. Merge when the change is ready. Update the version in `pyproject.toml`, build
   the distributions, and publish a new tag/release with concise release notes.

Normal bug fixes and public improvements can happen directly through public
feature branches. A permanent public `develop` branch is unnecessary for this
project. The private workspace remains active; it is not a read-only archive. Keep its
`software` branch aligned with public `main` as a starting point for new private
research branches. Copy or cherry-pick reviewed changes into a clean public
feature branch when they are ready.

## Validate a change

```bash
python -m pip install -e '.[dev,pythia,browser]'
python -m playwright install chromium
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m pytest
ruff check src tests examples analysis experiments
python -m analysis.reproduce verify
python -m build
```

Browser tests use Playwright's Chromium. To use an installed Chrome instead, set
`PREREQ_TEST_BROWSER` to its executable. Historical parity and paper-record tests
skip explicitly when their optional assets are absent. CI tests base installs on
three Python versions, PyTorch on CPU, and the interactive report in Chromium.

## Publish a version

Keep released evidence assets and old tags unchanged. A software fix gets a new
package version; a scientific correction also needs a clear correction note and
a versioned record. API compatibility and demonstrated scientific scope are
separate: a new adapter alone does not establish a new prerequisite mechanism.

The GitHub release includes installable wheels and source distributions. PyPI
publication is a separate step. For the first PyPI release, sign in to your PyPI
account, verify its email address, and enable the required two-factor authentication.
Then add a GitHub publisher under
[Account settings → Publishing](https://pypi.org/manage/account/publishing/):

| Field | Value |
|---|---|
| Project | `prerequisite-circuits` |
| GitHub owner | `KunwarK13` |
| Repository | `Prerequisite_Circuits` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

This is a [pending trusted publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/):
the first successful upload creates the PyPI project. It does not require an API
token in GitHub. The existing `pypi` GitHub environment and publishing workflow
are already configured.

After setting up the publisher, launch the workflow on the tested release tag:

```bash
gh workflow run publish.yml --repo KunwarK13/Prerequisite_Circuits --ref v0.1.0
```

The `v0.1.0` tag already exists. For later versions, create a new matching tag
after CI passes; keep old tags and assets unchanged. Running this workflow on
`main` fails its version check. The workflow tests and builds the package before
uploading through PyPI's trusted publishing service.

After the workflow succeeds, check the PyPI project page and install from PyPI
in a fresh environment:

```bash
python -m venv /tmp/prerequisite-pypi-check
/tmp/prerequisite-pypi-check/bin/python -m pip install prerequisite-circuits==0.1.0
/tmp/prerequisite-pypi-check/bin/python -m pip check
/tmp/prerequisite-pypi-check/bin/prereq --version
```

On Windows, use the environment's `Scripts` directory instead of `bin`. Confirm
that the installed CLI generates a report before announcing the release. The
Python package name uses a hyphen; the GitHub repository retains its underscore.

Future discovery methods and non-induction task adapters belong in this same
package, accompanied by suitable fresh validation and versioned examples.
