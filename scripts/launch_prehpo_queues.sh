#!/usr/bin/env bash
# Queues of the pre-HPO experiment blocks (FL_Contribution/reports/experiment_register.md, 2026-10-11):
#   scenarios_drift  client count 8/16/40/80, FedAvg local epochs, quantity skew, fl.drift_metrics
#   scenarios_nolag  scarce history and new parks without the power lag
# Shared NFS markers under ${DATA_ROOT}/runs/{fl_drift,fl_nolag} let l1 and l2 work the same lists
# (scripts/run_fl_scenarios.py); FL runs use every GPU of the host, local/CL the given ones.
#   l2:  nohup scripts/launch_prehpo_queues.sh l2  > logs/prehpo_l2.out  2>&1 &   # local + CL first, then FL
#   l1:  nohup scripts/launch_prehpo_queues.sh l1a > logs/prehpo_l1a.out 2>&1 &   # FL forward
#        nohup scripts/launch_prehpo_queues.sh l1b > logs/prehpo_l1b.out 2>&1 &   # FL reverse (second queue)
cd "$(dirname "$0")/.." || exit 1
eval "$(grep -E '^export DATA_ROOT=' ~/.bashrc)"
P=frcst/bin/python
Q=scripts/run_fl_scenarios.py
D=${DATA_ROOT}/runs
DRIFT="--manifest configs/parks_v1/scenarios_drift/manifest.csv --log-dir $D/fl_drift"
NOLAG="--manifest configs/parks_v1/scenarios_nolag/manifest.csv --log-dir $D/fl_nolag"
FL="--methods fl_fedgradient fl_fedavg"
case "$1" in
  l2)
    $P $Q $NOLAG --scenarios scarce_nolag lopo_nolag --methods local cl80 --gpus 0-3 --per-gpu 2
    $P $Q $DRIFT --scenarios skew --methods local --gpus 0-3 --per-gpu 2
    $P $Q $NOLAG --scenarios scarce_nolag lopo_nolag $FL
    $P $Q $DRIFT --scenarios clients skew $FL ;;
  l1a)
    $P $Q $DRIFT --scenarios clients skew $FL
    $P $Q $NOLAG --scenarios scarce_nolag lopo_nolag $FL ;;
  l1b)
    $P $Q $DRIFT --scenarios clients skew $FL --reverse
    $P $Q $NOLAG --scenarios scarce_nolag lopo_nolag $FL --reverse ;;
  *) echo "usage: $0 l2|l1a|l1b"; exit 1 ;;
esac
