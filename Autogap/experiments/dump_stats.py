"""Dump v2 statistics from summary_{PARAM_SET}_v2.json."""
import json
import os
import sys

# Try both likely locations
candidates = [
    os.path.join(os.path.dirname(__file__), "..", "results"),
    os.path.join(os.path.dirname(__file__), "results"),
]

res_dir = None
for c in candidates:
    if os.path.exists(c):
        res_dir = c
        break

if res_dir is None:
    print("results/ not found")
    sys.exit(1)

# Find the v2 summary file
files = [f for f in os.listdir(res_dir) if f.startswith("summary_") and
         f.endswith("_v2.json")]
if not files:
    print(f"No summary_*_v2.json found in {res_dir}")
    sys.exit(1)

# Default to d10 unless arg given
param_set = sys.argv[1] if len(sys.argv) > 1 else "d10"
path = os.path.join(res_dir, f"summary_{param_set}_v2.json")
if not os.path.exists(path):
    print(f"{path} not found")
    sys.exit(1)

with open(path) as f:
    s = json.load(f)

print(f"File: {path}")
print(f"Parameter set: {s.get('param_set')}, d = {s.get('num_params')}")
print(f"Sobol n_samples: {s.get('sobol_n_samples')}")
print()

print("=== Methods ===")
for m, v in s["methods"].items():
    print(f"  {m:16s}  gap={v['gap_reduction_mean']:.4f} "
          f"± {v['gap_reduction_std']:.4f}   "
          f"align={v['alignment_mean']:.4f} "
          f"± {v['alignment_std']:.4f}")

print()
print("=== Ablations ===")
for k, v in s["ablations"].items():
    print(f"  {k:16s}  {v:.4f}")

print()
print("=== DR-isolation ===")
dr = s.get("dr_isolation", {})
for k in ["mean", "std", "median", "q25", "q75", "min", "max", "frac_negative"]:
    if k in dr:
        print(f"  {k:16s}  {dr[k]:.4f}")

st = s["statistics"]

print()
print("=== Simple vs X tests ===")
for name in ["vs_rs10", "vs_rs30", "vs_dr", "vs_autogap"]:
    if name not in st["simple_tests"]:
        continue
    t = st["simple_tests"][name]["paired_t"]
    w = st["simple_tests"][name]["wilcoxon"]
    print(f"  Simple {name}:")
    print(f"    paired t: p = {t['p']:.4e}, d = {t['cohens_d']:.3f}")
    print(f"    wilcoxon: stat = {w['stat']:.4f}, p = {w['p']:.4e}")

print()
print("=== Holm-Bonferroni (family of 4) ===")
if "holm_family" in st:
    for label, h in st["holm_family"].items():
        print(f"  {label:22s}  raw = {h['p_raw']:.4e}  "
              f"adj = {h['p_holm']:.4e}")

print()
print("=== Shapiro on paired differences ===")
for k in sorted(st.keys()):
    if k.startswith("shapiro_diff"):
        print(f"  {k:42s}  W = {st[k]['W']:.3f}, p = {st[k]['p']:.4f}")

print()
print("=== Omnibus (active methods only) ===")
if "kruskal_active" in st:
    print(f"  Kruskal:  H = {st['kruskal_active']['H']:.3f}, "
          f"p = {st['kruskal_active']['p']:.4e}")
if "levene_active" in st:
    print(f"  Levene:   stat = {st['levene_active']['stat']:.4f}, "
          f"p = {st['levene_active']['p']:.4e}")

print()
print("=== Correlation ===")
c = st["corr_gap_vs_al"]
print(f"  Pearson r = {c['r']:.4f}, p = {c['p']:.4e}, "
      f"CI = [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]")