"""Main-table body of the paper from paper_main_stats.json (OmniSelect, common fixed repeats) and paper_nocoop_stats.json
(the same portfolio without cooperative strategies, same seeds). Best mean per column is bold, second distinct
mean is italic. Competition ranks 1, 2, 3 use dark, medium and light blue; exact ties share a rank and color.
Ranking uses unrounded, direction-aware means of all 14 methods including NC; reference summaries are excluded.
Tasks without records print the pending marker. Output: main_table_body.tex, and with
--zh the Chinese mirror main_table_body_zh.tex (same cells, translated group and row labels)."""
import json, sys
from pathlib import Path
import os
os.chdir(Path.cwd())
CIRCLE = json.load(open("table_method_ranks.json"))["method_to_circle_rank"]
ZH = "--zh" in sys.argv
L = {"Signal baselines": "信号基线", "External baselines": "外部基线", "Our method": "本文方法", "Reference summary": "参考汇总",
     "w/o cooperation": "无协同", "(ours)": "（本文）", "Best external": "最强外部基线", "Best signal": "最强单信号"}
def t(x): return L[x] if ZH else x
M = json.load(open("paper_main_stats.json"))["tasks"]; N = json.load(open("paper_nocoop_stats.json"))["tasks"]
SCALE = json.load(open("scale_up_stats.json"))
M.update(SCALE["main"]); N.update(SCALE["nocoop"])
COLS = ["C-100", "C-100N", "C-10", "IN-100", "ETTh1", "ETTm1", "ETTh2", "CSTR", "Steam", "TEP21", "Elec", "Text", "Text-100k"]
LOWER = {"ETTh1", "ETTm1", "ETTh2", "CSTR", "Steam", "Text", "Text-100k"}
GROUPS = [("Signal baselines", [("random", "Random"), ("influence_only", "Infl-only"), ("coreset", "Coverage"), ("mmdataselect", "Fixed fusion"), ("auth_only", "Auth-only")]),
          ("External baselines", [("herding", "Herding"), ("el2n", "EL2N/adapt"), ("grand", "GraNd/adapt"), ("ccs", "CCS/adapt"), ("density", "Density"), ("quadmix_pub", "QuaDMix-pub"), ("dmf_pub", "DMF-pub")])]
TEXT_ALIAS = {"herding": "herding_text", "density": "density_text", "coreset": "coverage_text"}
def cell(T, name, col):
    if col not in T or "rows" not in T[col]: return None
    rows = T[col]["rows"]; key = name
    if col == "Text" and name not in rows and name in TEXT_ALIAS: key = TEXT_ALIAS[name]
    return rows.get(key)
def fmt(r, col):
    d = 3
    return f"{r['mean']:.{d}f}{{\\scriptsize$\\pm${r['sd']:.{d}f}}}" if r["sd"] == r["sd"] else f"{r['mean']:.{d}f}"
entries = {}   # (rowname, col) -> row dict
for _, members in GROUPS:
    for key, _ in members:
        for c in COLS: entries[(key, c)] = cell(M, key, c)
for c in COLS:
    entries[("omni", c)] = cell(M, "mmds_adapt", c)
    entries[("nocoop", c)] = cell(N, "mmds_adapt", c)
rank = {}
competition_rank = {}
RANK_COLORS = {1: "omnitablefirst", 2: "omnitablesecond", 3: "omnitablethird"}
for c in COLS:
    vals = [(r["mean"], k) for (k, cc), r in entries.items() if cc == c and r is not None]
    if not vals: continue
    vals.sort(reverse=c not in LOWER)
    uniq = sorted({v for v, _ in vals}, reverse=c not in LOWER)
    best = uniq[0]; second = uniq[1] if len(uniq) > 1 else None
    for v, k in vals:
        rank[(k, c)] = "best" if v == best else ("second" if second is not None and v == second else "")
        # Competition/min rank: 1 + the number of strictly better methods.
        # E.g. two tied first places are followed by rank 3, not rank 2.
        competition_rank[(k, c)] = 1 + sum(other < v if c in LOWER else other > v
                                          for other, _ in vals)
def show(k, c):
    r = entries.get((k, c))
    if r is None:
        return "--" if (c in M and "rows" in M[c]) else r"\rp"
    tag = rank.get((k, c), "")
    mean = f"{r['mean']:.3f}"
    if tag == "best": mean = r"\textbf{" + mean + "}"
    elif tag == "second": mean = r"\textit{" + mean + "}"
    color = RANK_COLORS.get(competition_rank.get((k, c)))
    prefix = r"\cellcolor{" + color + "}" if color else ""
    return prefix + r"\omnicell{" + mean + "}{" + f"{r['sd']:.3f}" + "}"
lines = []
for gname, members in GROUPS:
    for i, (key, label) in enumerate(members):
        name = rf"\Circled{{{CIRCLE[key]}}}\,{label}"
        if i == 0: name = r"\omnigroup{" + t(gname) + "}{" + name + "}"
        lines.append(name + " & " + " & ".join(show(key, c) for c in COLS) + r" \\")
    lines.append(r"\midrule")
name = rf"\Circled{{{CIRCLE['nocoop']}}}\,\sys{{}}-NC"
lines.append(r"\omnigroup{" + t("Our method") + "}{" + name + "} & " + " & ".join(show("nocoop", c) for c in COLS) + r" \\")
lines.append(rf"\Circled{{{CIRCLE['omni']}}}\,\sys{{}} {t('(ours)')} & " + " & ".join(show("omni", c) for c in COLS) + r" \\")
open("main_table_body_zh.tex" if ZH else "main_table_body.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[-4:]))
