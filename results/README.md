# Results

`paper/` contains the reported run records, numerical outputs and scripts for rebuilding them without training.

```bash
cd results/paper && bash rebuild.sh
```

The command unpacks the archives into `paper/work/`, recomputes the statistics and compares them
with `paper/tables/`. It needs Python 3.10 or later with NumPy and SciPy. `PYTHON` selects the interpreter.

## Record archives (`paper/records/`)

The run archives contain configurations, splits, leaderboards, decisions, metrics and timings.
The main archive also contains selections and signal arrays. `paired_units.tar.gz` stores compact
ranking-split arrays for Figure 3. Paths in JSON records use `<repo>`, `<data>` and `<host>` placeholders.

| Archive | Cells | Paper content |
|---|---:|---|
| `main_v2_3.tar.gz` | 72 | Main and OmniSelect-NC runs for the eleven benchmark tasks and auxiliary C-10-CLIP |
| `main_adapt_v2_3.tar.gz` | 18 | EL2N, GraNd and CCS on forecasting and text |
| `scale_up.tar.gz` | 15 | IN-100 and Text-100k, including NC and text adaptations |
| `robustness_v2_3f.tar.gz` | 153 | Conditions of Figure 4(e)(f), with nine main cells reused from the main archive |
| `ablation_v2_3.tar.gz` | 45 | Three signal removals on five tasks, Figure 4(d) displays four of them |
| `validation_size_v2_3.tar.gz` | 9 | CIFAR-100 at 200, 400 and 1600 validation records, the 800 point reuses the main archive |
| `paired_units.tar.gz` | 69 | Figure 3, main and robustness ranking splits |

Each task or condition contains the reported multi-seed runs listed in `paper/seeds_reported.json`.
The signal-removal comparisons use the same runs as their main-table controls. The record packages
retain the original metric values.

## Tables (`paper/tables/`)

| File | Content |
|---|---|
| `paper_main_stats.json`, `paper_nocoop_stats.json` | Per-seed test utilities, means, sample standard deviations, paired intervals, elections and adoptions on the benchmark tasks |
| `paper_main_agg.json`, `paper_nocoop_agg.json` | Cross-task comparisons, mean gains, Friedman ranks, Holm-corrected Wilcoxon tests and cost |
| `table_method_ranks.json` | Mean ranks across the eleven benchmark tasks |
| `scale_up_stats.json` | Multi-seed statistics for IN-100 and Text-100k |
| `main_table_body.tex` | The 13-column body of Table 3 |
| `paper_numbers.json`, `adoption_numbers.json` | Aggregate result, adoption and cost statistics |
| `paired_diagnostics.json` | Figure 3, 23 groups, 1452 pairs and 2904000 resamples |
| `robust_cells_view.json`, `robustness_summary.json` | The 162 robustness cells and their summaries |
| `validation_noise_plot_data.json` | Figure 4(f), 36 validation-noise points |
| `abl_v23_summary.json` | Multi-seed signal-removal results used in Figure 4(d) |
| `validation_size.json` | Figure 4(g), twelve validation-size points |
| `figure6_validation_size_and_signal_ablation.json` | Existing combined export of Figure 4(d)(g), retaining its original filename |
| `admission_shares.json`, `diagnostics.json` | Admission shares and strategy diagnostics exported from the full records |

`rebuild.sh` compares 16 rebuilt outputs. It also compares the signal-removal results and the
plotted validation-size values in the combined export. The standalone admission and strategy
diagnostic exports retain their existing values.
