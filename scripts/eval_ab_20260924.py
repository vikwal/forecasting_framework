import csv, glob, statistics as st, itertools, math
rows = {}
for f in sorted(glob.glob("data/test_results/altab_*_fold1.csv")):
    tag = f.split("/")[-1].replace("altab_", "").replace("_fold1.csv", "")
    r = list(csv.DictReader(open(f)))
    rows[tag] = {x["station_id"]: x for x in r}
    print("%-30s n=%3d" % (tag, len(r)))
print()
print("%-22s %8s %8s %8s %8s" % ("Arm / Normierung", "RMSE", "MAE", "R2", "Skill"))
for tag in sorted(rows):
    d = rows[tag]
    def m(c): return st.mean(float(x[c]) for x in d.values())
    print("%-22s %8.4f %8.4f %8.4f %8.4f" % (tag, m("rmse"), m("mae"), m("r2"), m("skill_nwp")))
print()
for arm in ("dcrnn", "dcrnn_idw_alt"):
    a, b = rows.get(arm + "_alt500"), rows.get(arm + "_alt3000")
    if not (a and b): continue
    common = sorted(set(a) & set(b))
    diff = [float(a[s]["rmse"]) - float(b[s]["rmse"]) for s in common]
    mean = st.mean(diff)
    sd = st.pstdev(diff)
    se = sd / math.sqrt(len(diff))
    better = sum(1 for x in diff if x < 0)
    print("%-16s /500 minus /3000 je Station: Mittel %+.4f  (SE %.4f)  besser an %d von %d"
          % (arm, mean, se, better, len(diff)))
    try:
        from scipy.stats import wilcoxon
        print("%-16s   Wilcoxon p = %.3f" % ("", wilcoxon(diff).pvalue))
    except Exception as e:
        print("%-16s   (kein scipy: %s)" % ("", type(e).__name__))
