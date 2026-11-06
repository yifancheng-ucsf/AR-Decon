#!/bin/bash
# compare_with_v09.sh -- whole-pipeline check: current AR-Decon (ER-Decon II built from source)
# against AR-Decon 0.9 (bundled Priism binary) on the tutorial data.
#
#   tests/compare_with_v09.sh AR-Decon_Tutorial OLD_ENV NEW_ENV [OUTDIR] [ITER]
#
# OLD_ENV / NEW_ENV: conda environments with AR-Decon 0.9 and the current version installed
# (e.g. ardecon and ardecon1).  Both runs use the same random cone
# directions: 0.9 by seeding Python's random module at start-up
# (sitecustomize), the current version with --seed.  Then every output of the two pipelines
# is compared: the directional FSC, the OTF, the masked map and the
# deconvolved map (which may differ in the last digit of a few voxels, as
# the old program does from run to run).
set -euo pipefail
T=$(cd "$1" && pwd); OLD=$2; NEW=$3; OUT=${4:-compare_v09}; ITER=${5:-50}; SEED=7
mkdir -p "$OUT/site"; cd "$OUT"
echo "import random; random.seed($SEED)" > site/sitecustomize.py
ARGS="$T/PreProcess/emd_8731_half_map_1_RightH.mrc $T/PreProcess/emd_8731_half_map_2_RightH.mrc $T/RawData/emd_8731.mrc --mask $T/PreProcess/emd_8731_msk_1_RightH.mrc --iter $ITER"
export MPLBACKEND=Agg
echo "[1/2] AR-Decon 0.9 ($OLD) ..."
( time PYTHONPATH=$PWD/site conda run -n "$OLD" ardecon $ARGS --outdir old09 ) > old09.log 2>&1
echo "[2/2] AR-Decon current ($NEW) ..."
( time conda run -n "$NEW" ardecon $ARGS --outdir new10 --seed $SEED ) > new10.log 2>&1
python3 - <<'PY' | tee verdict.txt
import os, numpy as np
def data(p):
    raw = open(p, 'rb').read(); nxt = int(np.frombuffer(raw[92:96], '<i4')[0])
    return raw[:1024], np.frombuffer(raw[1024 + nxt:], '<u4')
def ulp(a, b):
    ha, da = data(a); hb, db = data(b)
    if ha == hb and np.array_equal(da, db): return 0, 0, "identical (whole file)"
    ia = da.view(np.int32).astype(np.int64); ib = db.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia); ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    d = np.abs(ia - ib)
    global NVOX; NVOX = d.size
    return int(d.max()), int((d > 0).sum()), "max %d ulp in %d voxels%s" % (d.max(), (d > 0).sum(), "" if ha == hb else ", header differs")
ok = True
root = [f for f in os.listdir("old09") if f.endswith("_Decon.mrc")][0][:-len("_Decon.mrc")]
for suffix in ["_HalfMapFibonacciPoints.txt", "_HalfMapdFSC1d.txt", "_HalfMapdFSCAvg.txt"]:
    same = open("old09/" + root + suffix, "rb").read() == open("new10/" + root + suffix, "rb").read()
    ok &= same; print("%-40s %s" % (root + suffix, "identical" if same else "DIFFERS"))
for suffix in [".mrc", "_HalfMapdFSC3d.mrc", "_HalfMapdFSC3d.otf"]:
    m, n, txt = ulp("old09/" + root + suffix, "new10/" + root + suffix)
    ok &= (m == 0); print("%-40s %s" % (root + suffix, txt))
m, n, txt = ulp("old09/" + root + "_Decon.mrc", "new10/" + root + "_Decon.mrc")
ok &= (m <= 2 and n <= 1e-3 * NVOX); print("%-40s %s" % (root + "_Decon.mrc", txt))
print("VERDICT: %s" % ("PASS" if ok else "FAIL"))
PY
grep real old09.log new10.log | sed 's/^/time /'
