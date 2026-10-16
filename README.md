# Multi-Noise2Noise

Code of *Multi-Noise2Noise: Physics-Aware Self-Supervised Denoising for Cryo-Electron
Microscopy* (Grzesiuk, Karanowski, Kuczbański, Rzepiela, Frey, Zięba; Bioinformatics,
submitted). How to cite: [`CITATION.cff`](CITATION.cff). License: [MIT](LICENSE).

Multi-Noise2Noise trains a U-Net on dose-fractionated particle movies: the average of all but
one frame is the input and the held-out frame the target, optionally with CTF- and
dose-weighted Fourier losses. The repository contains training, denoising, 3D reconstruction
of the denoised particles, the evaluation behind every reported number
(`source/evaluation`, `configs_eval/paper.yaml`) and the motion-correction and extraction
pipelines used to prepare the particle movies.

Data, partitions, masks, reference maps, checkpoints and reconstructions are in the
[`cryo_datasets`](https://github.com/cryo-EM-team/cryo_datasets) repository (DVC, see its
README for `dvc pull` and Google Drive access).

## Installation

Tested on Linux with Python 3.13 (ETH Euler; training and reconstruction on one RTX 4090).

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt   # exact environment of the manuscript
pip install --no-deps -e .        # this package (entry points need the repository checkout)
```

`pip install -e ".[scunet,test,render]"` instead resolves current versions of the same
dependencies. The `scunet` extra (`einops`, `timm`) is only needed for `model=scunet`, which
the paper does not use; `render` (`pyvista`) for `source.evaluation.render_locres` (Figs. 7
and 8).

Machine-specific paths go into `configs_train/local/default.yaml` (git-ignored):

```yaml
storage_dir: /path/to/storage   # logs, checkpoints and reconstructions are written here
user_name: your-name            # sub-directory of the logs
```

Data are read from `paths.data_dir` (default `${storage_dir}/data`); point it to the
`cryo_datasets` checkout with `paths.data_dir=/path/to/cryo_datasets` or a symlink.
Training logs to Weights & Biases by default; add `logger.wandb.offline=True` to log locally.
Keep the W&B logger (online or offline) for training: the validation loss is computed only
when a logger other than `CSVLogger` is attached. With `logger=csv` or without a logger,
`val/loss` is logged as `inf` and early stopping ends training as soon as `min_epochs` is
reached. `debug=limit` disables the logger and callbacks and runs on the CPU; it is meant for
smoke tests only.

## Smoke test

```bash
pip install pytest
python -m pytest tests -p no:cacheprovider
```

`tests/test_train_smoke.py` trains one epoch of `experiment=10948/multinoise2noise/default`
with `debug=limit` on a synthetic HDF5 (400 particles, 4 frames, 32 × 32 px), checks the
two half-map reconstructions of the test stage and denoises the dataset with the
checkpoint (about 4 min on a CPU). `tests/test_cli_help.py` runs `--help` of every entry
point.

## Method variants

Configs are `configs_train/experiment/<dataset>/<family>/<variant>.yaml`; checkpoints are
`cryo_datasets/checkpoints/<dataset>/<checkpoint>`.

| Method in the paper | Experiment | `task_name` | Checkpoint |
|---|---|---|---|
| Topaz-Denoise | `<ds>/multinoise2noise/base` | `noise2noise` | `base.ckpt` |
| Topaz-Denoise + CTF | `<ds>/multinoise2noise_ctf/base` | `noise2noise_ctf` | `base_ctf.ckpt` |
| Multi-Noise2Noise | `<ds>/multinoise2noise/default` | `multi_noise2noise` | `default.ckpt` |
| Multi-Noise2Noise + CTF | `<ds>/multinoise2noise_ctf/default` | `multi_noise2noise_ctf` | `default_ctf.ckpt` |
| Multi-Noise2Noise + DW (experimental data) | `<ds>/multinoise2noise/dose_weighted` | `multi_noise2noise_dw` | `dose_weighted.ckpt` |
| Multi-Noise2Noise + CTF + DW (experimental data) | `<ds>/multinoise2noise_ctf/dose_weighted` | `multi_noise2noise_ctf_dw` | `dose_weighted_ctf.ckpt` |
| Fig. 2, `N` frames averaged into the input (simulated data) | `<ds>_sim/multinoise2noise/split_<N>` | `split_<N>` | – |
| Raw (not denoised) reconstructions | `<ds>/multinoise2noise/raw` | `raw` | – |

Other configs are not used in the paper:

- `dose_weighted_legacy`: loss configuration the DW checkpoints were trained with
  (`dose_weighted{,_ctf}_legacy.ckpt`); `dose_weighted{,_ctf}.ckpt` hold the same weights for
  the manuscript loss (weights unchanged, not retrained).
- `split_<N>` of the experimental datasets, `raw_full`, `dose_weighted_both`,
  `dose_weighted_regularized`, `*_patch`, `scunet`, `base_scunet`, and the `dose_weighted`
  configs of the simulated datasets.

## Training

The paper's partition (25 % train, 50 % validation, 25 % test) is the `split` column of
`cryo_datasets/<ds>/training/<ds>_split.star`:

```bash
python -m source.train experiment=10948/multinoise2noise/default \
    datamodule.split_mode=column \
    datamodule.particles_star_path=/path/to/cryo_datasets/10948/training/10948_split.star
```

Validation particles drive only early stopping and the LR schedule; with `split_mode=column`
they never enter a reconstruction. After training, the best checkpoint reconstructs the two
half-maps from the two `rlnRandomSubset` halves of the test set (`trainer.test`), written to
`${storage_dir}/reconstruction/<ds>/<task_name>/denoised_mean/`. Without the overrides, the
configs use the earlier setting (`<ds>_subset.star`, `split_mode=random_subset`, training on
its `rlnRandomSubset == 1` half, validation on the other half).

## Reconstructing the evaluation subsets

`Refine_4/<r>` of every method is the reconstruction of the test subset
`<ds>/subsets/denoised_4_<r>.star` (r = 1–5) with the trained model, one half-map per
`rlnRandomSubset` half:

```bash
python -m source.train experiment=10948/multinoise2noise/default train=false \
    ckpt_path=/path/to/cryo_datasets/checkpoints/10948/default.ckpt \
    datamodule.particles_star_path=/path/to/cryo_datasets/10948/subsets/denoised_4_1.star \
    datamodule.weight_doses=False task_name=multi_noise2noise/Refine_4/1
```

Keep the default `split_mode=random_subset` here: the `split` column of these STAR files marks
every particle `train`, which `split_mode=column` rejects. With `random_subset`, a STAR file
without a `split` column gets one written back in place (`recalculate_splits` forces this);
pass a copy if the file must stay unchanged. For `denoised_4_1.star` of 10948 this command reproduces
`results/10948/multi_noise2noise/Refine_4/1/denoised_mean/half_{1,2}_reconstruction.mrc`
(voxel correlation 1.0, maximum relative difference 6e-8).

## Denoising

```bash
python -m source.denoise experiment=10948/multinoise2noise/default \
    "ckpt_path='/path/to/model.ckpt'" \
    datamodule.particles_star_path=/path/to/particles.star
```

Denoised particles are written as one `.mrcs` stack per micrograph to
`${storage_dir}/reconstruction/<ds>/<task_name>/`, and `denoised.star` next to the input STAR
file. Quote checkpoint paths that contain `=` (Lightning's `epoch=…` names) as shown.

## Regenerating tables and figures

`configs_eval/paper.yaml` holds every evaluation parameter. Inputs are read from
`$CRYO_DATASETS_DIR` (the `cryo_datasets` checkout), outputs written to `$EVAL_OUTPUT_DIR`
(default `./eval_outputs`):

```bash
export CRYO_DATASETS_DIR=/path/to/cryo_datasets EVAL_OUTPUT_DIR=/path/to/eval_outputs
python -m source.evaluation.make_mask                              # tight masks from <ds>_model.mrc
python -m source.evaluation.evaluate_reconstruction                # model-to-map FSC, Table 4
python -m source.evaluation.evaluate_local_resolution locres.stage=items
python -m source.evaluation.evaluate_local_resolution locres.stage=aggregate
python -m source.evaluation.make_tables                            # tables/fsc_recon.{tex,csv}
python -m source.evaluation.make_figures                           # figures/*.{png,pdf}
python -m source.evaluation.render_locres                          # figures/locres_grid_*; needs .[render]
```

`evaluate_reconstruction` and `evaluate_local_resolution locres.stage=items` accept
`shard.index=<i> shard.count=<n>` for array jobs (then `evaluate_reconstruction
collect_only=true` merges the rows) and `select=[10025,10948]` to restrict datasets.

| Output | Paper |
|---|---|
| `figures/frame_sweep` | Fig. 2 |
| `figures/fsc_true_10474` | Fig. 5 |
| `figures/locres_histograms_3x3` | Fig. 6 |
| `figures/locres_grid_real` | Fig. 7 |
| `figures/locres_grid_sim` | Fig. 8 |
| `figures/fsc_true_3x3` | Supp. Fig. S2 |
| `tables/fsc_recon.tex` | Table 4 |

## Repository layout

```
configs_train/              training, denoising and reconstruction (Hydra)
configs_eval/paper.yaml     evaluation parameters of the paper
configs_extract/            particle extraction (python -m source.extract)
configs_motion_correction/  motion correction (python -m source.correct_motion)
source/
  _noise2noise/             Lightning models of the Noise2Noise family
  datasets/                 datamodules, datasets, split_star (partition builder)
  evaluation/               masks, FSC, local resolution, tables, figures
  loss_functions/           CTF- and dose-weighted losses, FRC
  models/                   U-Net (and SCUNet, not used in the paper)
  reconstruction/           back-projection, CTF, dose weighting, postprocessing
tests/
```

Reconstruction supports the D7 and O symmetries (`source/reconstruction/symmetry_files/`).
