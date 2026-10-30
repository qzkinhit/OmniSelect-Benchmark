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
| `main_v2_3.tar.gz` | 72 | Main and OmniSelect w/o Coop runs for the eleven benchmark tasks and auxiliary C-10-CLIP |
| `main_adapt_v2_3.tar.gz` | 18 | EL2N, GraNd and CCS on forecasting and text |
| `scale_up.tar.gz` | 15 | IN-100 and Text-100k, including OmniSelect w/o Coop and text adaptations |
| `robustness_v2_3f.tar.gz` | 117 | Conditions of Figures 3 and 4(e)(f), with nine main cells reused from the main archive |
| `ablation_v2_3.tar.gz` | 36 | Three signal removals on the four tasks of Figure 4(d) |
| `validation_size_v2_3.tar.gz` | 9 | CIFAR-100 at 200, 400 and 1600 validation records, the 800 point reuses the main archive |
| `paired_units.tar.gz` | 69 | Figure 3, main and robustness ranking splits |
| `text_selections_v2_3.tar.gz` | 3 | Selections of the six Table 4 strategies in the Text main runs, unpacked into the main tree |

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
| `paired_diagnostics.json` | Figure 3, 23 groups, 1454 pairs and 2908000 resamples |
| `robust_cells_view.json`, `robustness_summary.json` | The 126 robustness cells and their summaries |
| `validation_noise_plot_data.json` | Figure 4(f), 36 validation-noise points |
| `abl_v23_summary.json` | Multi-seed signal-removal results used in Figure 4(d) |
| `validation_size.json` | Figure 4(g), twelve validation-size points |
| `table4_admission.json` | Table 4, the share of each kind of dirty record that a strategy selects, its column minima and row means, and the best external baseline of each task |
| `adjudication_by_task.json` | Appendix table of paired verdicts and election on each task |
| `diagnostics.json` | Cleanliness AUCs, hindsight fusion means and cooperative-strategy means quoted in Sections 1 and 5 |

`rebuild.sh` compares 18 rebuilt outputs.
