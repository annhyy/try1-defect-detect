# Next X-SDD DNM experiments

This directory is independent of the original `alfoil_dnm` implementation and
keeps all new model code and output entry points together. All experiments use
the fixed `datasets/xsdd_yolo11_classification` split unless `--data` is
provided.

## Entries

```powershell
# Clean four-synapse ablations: only V2 branch_features changes from 8 to 4.
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_next\train_v2a_f4.py
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_next\train_v2b_f4.py

# Tuned V1: independent branch projections, LayerNorm without a sigmoid,
# log-domain four-way products, signed branch strengths, and class bias.
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_next\train_v1_tuned.py

# Optional paired weighted-loss control. This does not overwrite the old control.
D:\Anaconda_envs\envs\pytorch\python.exe .\alfoil_dnm_next\train_conv_control_weighted.py
```

The default output directories are:

| entry | output |
| --- | --- |
| V2a-F4 | `runs1/controlled/xsdd_dnm_v2a_f4_cls` |
| V2b-F4 | `runs1/controlled/xsdd_dnm_v2b_f4_cls` |
| V1-Tuned | `runs1/controlled/xsdd_dnm_v1_tuned_cls` |
| weighted Conv-Control | `runs1/controlled/xsdd_conv_control_weighted_cls` |

Every epoch logs `pred_count=[...]` for both train and validation and stores
the same values in `metrics.csv`. Each run writes `best_accuracy.pt`,
`best_macro_f1.pt`, `best.pt` (an accuracy-selected alias), and `last.pt`.
The final `test_metrics.json` evaluates the accuracy-selected checkpoint;
Macro-F1 selection remains available through its separately saved checkpoint.

V2a-F4 and V2b-F4 retain the old 100-epoch, single `1e-3` learning-rate
protocol and cosine minimum of zero. V1-Tuned defaults to 150 epochs, backbone
and head learning rates `1e-3` and `3e-3`, and a cosine floor of 1%. Use
`--class-weighting balanced` for the optional inverse-frequency loss run; the
paired Conv-Control entry uses the same class weights by default.
