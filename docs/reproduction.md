# Reproduction

There are three levels: inspect saved evidence, audit/rebuild it, and run a new
training replication. These are different operations. New runs never replace the
reference records.

## 1. Inspect the evidence without training

```bash
python -m pip install .
python -m analysis.reproduce verify
prereq report results/reference/410m_circuit_tape38.json --html outputs/410m.html
```

The checked-in trajectories are small and the reporting path has no PyTorch
dependency. Their metadata maps directly to original records. The report threshold
is a simple accuracy criterion; it does not replace the paper's key-validation
criteria.

## 2. Rebuild figures and tables on CPU

```bash
python -m pip install '.[plot]'
python -m experiments.artifacts paper-inputs
python -m analysis.reproduce figures
python -m analysis.reproduce tables
```

The input download is about 27 MiB (about 276 MiB extracted). Each source file is
verified before analysis. Outputs go to `outputs/figures/` and `outputs/tables/`.
No GPU, model weights, or corpus downloads are needed. The command includes the
controlled figures, Pythia figures, larger-model figure, and later appendix
tables. It does not overwrite the manuscript's figures or PDF.

### Build the current paper

The complete Overleaf source is in `paper/`, including the figures and generated
table rows. With a standard TeX Live installation (including `texlive-latex-extra`
and `texlive-fonts-recommended` on Debian/Ubuntu), run:

```bash
python -m analysis.reproduce paper
```

The PDF is written to `outputs/paper/main.pdf`; the checked-in `paper/main.pdf`
remains the release reference. No data download or Python plotting dependencies
are needed. The bibliography is embedded in `refs.tex`, so BibTeX is not required.
In Overleaf, upload the contents of `paper/` and select `main.tex` with pdfLaTeX.

## 3. Audit the historical controlled/160M release

```bash
python -m experiments.artifacts reference-source
python -m experiments.artifacts controlled-checkpoints
python -m experiments.prepare controlled outputs/reference --checkpoints
```

The checkpoint download is about 435 MiB. The prepared workspace preserves original
paths because source seals refer to them. The historical source is deliberately
unaltered. Its original CPU audit can be run from the extracted reference release:

```bash
python artifacts/reference-source/reproduce.py audit
```

`prepare` imports the matching checkpoints into that local release first. Install
`experiments/environments/analysis.txt` in a separate environment for the historical
audit. The external-corpus integration check is skipped unless corpus data are
provided; this is reported explicitly. Large pretrained endpoint weights are not
included. The saved-prediction audits must not be described as new weight audits.

The earlier `reproduce.py paper` command rebuilds the **September 20 reference
manuscript**, not the current preprint. Use `python -m analysis.reproduce paper`
for the current source and the figure/table commands above for the later analyses.

## 4. Run a new controlled replication

Use the prepared `outputs/reference/curriculum_order` directory. The original
source and its protocols are in the reference asset; `activation_rescue/run.py`
and `stable_input_reversible/run.py` implement removal and supply. The data
construction instructions, corpus hashes, and exact original commands are in
`artifacts/reference-source/docs/data_and_models.md` and its reproduction guide.

A historical workspace contains recorded outcomes for comparison. Before a new
execution, work in a new copy and remove only the selected study's generated
outputs; keep its starting checkpoints, protocol, source, and manifest. New output
is a replication, not the original sealed execution. The archived runners select
CUDA for training. Preparing a workspace never launches them.

For new code rather than exact historical execution, start with `prereq demo` or
`examples/monitor_training.py` and the reusable API.

## 5. Prepare a larger-model replication

```bash
python -m experiments.prepare pythia outputs/pythia --hours 72
cd outputs/pythia
export PYTHIA_SCALING_CONFIG="$PWD/replication.json"
```

Use the separate recorded Pythia environment. The new config copies the original
campaign's scientific settings but assigns a fresh operational deadline. It is a
new execution configuration, not an alteration of the original protocol.

Place the verified enwik8 corpus at `data/local/enwik8`. Then these commands
explicitly download inputs/models and run GPU selection and validation:

```bash
python -m extensions.pythia_scaling.prepare data
python -m extensions.pythia_scaling.prepare model --size 160m --revision step1000
python -m extensions.pythia_scaling.select --size 160m
python -m extensions.pythia_scaling.preflight
python -m extensions.pythia_scaling.preflight --intervention-diagnostics
python -m extensions.pythia_scaling.campaign --sizes 160m --cohort replication --intervention-diagnostics
```

This is the initial 160M campaign. Larger-model and long-horizon studies used
different pinned checkpoints, learning rates, head selections, and configurations.
Their exact settings and execution bindings are in the paper-input bundle, and
their runners are in the scaling-source asset. Do not obtain a claimed exact
replication by merely substituting a model size in the example above.

For models requiring two GPUs, the archived `fsdp_train.py`, `fsdp_core.py`, and
`fsdp_checkpoint.py` provide the tested full-parameter backend. Its distributed
preflight and matching input/selection records are required. The lightweight
package runner is intentionally not a replacement for that backend.

## Release validation

`results/validation.json` records the checks actually performed for this release.
No new GPU experiment is required to replot the evidence. Numerical parity and
short CPU tests validate the extracted package semantics; they do not constitute
fresh large-model replications. Hardware/time claims are measured, not inferred
from model parameter counts.
