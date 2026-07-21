"""Fidelity-curve checker: parse the CIFAR-10 clean pruning curve and verify it reproduces
the published Data-Diet (Paul 2021) + CCS (Zheng 2023) qualitative shape within tolerance:
  (1) at high pruning (keep<=0.2) EL2N and GraNd collapse well below random;
  (2) CCS is robust at high pruning (keep<=0.3): above both EL2N and GraNd there;
  (3) at low pruning (keep>=0.5) EL2N/GraNd recover to within ~0.05 of random.
Writes outputs/fidelity_curve/cifar10.json + a verdict; no marker unless all hold."""
import glob
import json
import os
import re
import sys

R = "/root/autodl-tmp/OmniSelect"
curve = {}
for lg in sorted(glob.glob(f"{R}/experiments/fidelity_curve_cifar10_keep*.log")):
    K = float(re.search(r"keep([0-9.]+)\.log", lg).group(1))
    txt = open(lg, errors="ignore").read()
    if not re.search(rf"KEEP={K} python_exit=0", txt):
        continue
    row = {}
    for m in re.finditer(r"^\s{2}(random|el2n|grand|ccs)\s+val=([0-9.]+) test=([0-9.]+)", txt, re.M):
        row[m.group(1)] = float(m.group(3))
    if set(row) == {"random", "el2n", "grand", "ccs"}:
        curve[K] = row

print("keep   random   el2n    grand   ccs")
for K in sorted(curve):
    r = curve[K]
    print(f"{K:<5}  {r['random']:.3f}   {r['el2n']:.3f}  {r['grand']:.3f}  {r['ccs']:.3f}")

fails = []
# (1) high-pruning collapse
for K in [k for k in curve if k <= 0.2]:
    r = curve[K]
    if not (r["el2n"] < r["random"] - 0.05 and r["grand"] < r["random"] - 0.05):
        fails.append(f"keep{K}: EL2N/GraNd not collapsed below random")
# (2) CCS robust in the HIGH-pruning regime only (keep<=0.3); at low pruning EL2N/GraNd
# are legitimately better (Data-Diet: keeping hard examples helps at moderate pruning),
# so CCS is not expected to dominate everywhere - the published claim is high-pruning robustness.
for K in [k for k in curve if k <= 0.3]:
    r = curve[K]
    if not (r["ccs"] >= r["el2n"] and r["ccs"] >= r["grand"]):
        fails.append(f"keep{K}: CCS not robust vs EL2N/GraNd at high pruning")
# (3) low-pruning recovery
for K in [k for k in curve if k >= 0.5]:
    r = curve[K]
    if not (r["el2n"] >= r["random"] - 0.05 and r["grand"] >= r["random"] - 0.05):
        fails.append(f"keep{K}: EL2N/GraNd did not recover near random")

d = os.path.join(R, "outputs/fidelity_curve")
os.makedirs(d, exist_ok=True)
verdict = "FIDELITY_CURVE_OK" if (not fails and len(curve) >= 4) else "FIDELITY_CURVE_INCOMPLETE"
json.dump({"curve": curve, "checks_failed": fails, "verdict": verdict,
           "reproduces": "Data-Diet(Paul21)+CCS(Zheng23) pruning-curve shape"},
          open(os.path.join(d, "cifar10.json"), "w"), indent=2)
print(f"\n{verdict}  keeps={sorted(curve)}  fails={fails}")
if verdict == "FIDELITY_CURVE_OK":
    open("/root/FIDELITY_CURVE_OK", "w").write(json.dumps({"keeps": sorted(curve)}))
sys.exit(0 if not fails and len(curve) >= 4 else 1)
