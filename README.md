# Code and data for "Reported POPE gains are mostly template artifacts"

This release accompanies the manuscript:

> Junfei Huang. Reported POPE gains are mostly template artifacts: an audit of
> object-hallucination mitigation and detection. 2026.

It contains the analysis code, per-item model outputs, result summaries, and
figure-generation scripts used to audit object-hallucination evaluation on
POPE.

## What is included

- `scripts/`: evaluation, per-item decomposition, McNemar tests, bootstrap
  intervals, CHAIR analysis, PAS detection reproduction, label-noise analysis,
  consistency checks, and figure generation.
- `outputs/`: saved model outputs and result summaries, including the full
  5000-image PAS detection records.
- `figures/`: the two figures in TIFF (300 dpi) and PNG format.
- `RESULTS.md`: experiment log with file pointers for every reported number.
- `坑.md`: reproducibility pitfalls encountered during the runs.

## Data provenance

- POPE questions and images: the public POPE release (`lmms-lab/POPE`).
- COCO val2014 images and instance annotations: the public COCO release.
- Model checkpoints: Hugging Face releases of LLaVA-1.5-7B, LLaVA-NeXT-7B,
  LLaVA-OneVision-7B, and Qwen2.5-VL-7B-Instruct. Model weights are not
  redistributed here.

## Environment

The runs used Python 3.12, PyTorch 2.5.1 with CUDA 12.1, transformers 5.9.0,
bitsandbytes 0.50.2, and a single 12 GB GPU. See `RESULTS.md` for run-specific
settings.

## Reproduce the checks

```bash
python scripts/audit_paper_consistency.py
python scripts/make_figures.py
```

The first command recomputes the manuscript's headline numbers from
`outputs/*.json`. The second regenerates the figures.

## Licensing

- Code in `scripts/`: MIT License, see `LICENSE`.
- Derived result files and figures: Creative Commons Attribution 4.0
  International, see `LICENSE-DATA`.

## Contact

Junfei Huang, School of Computer Science, Beijing University of Posts and
Telecommunications, Beijing, China. E-mail: huangjunfei@bupt.edu.cn.
