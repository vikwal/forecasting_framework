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
TRN="--manifest configs/trianel_newcomer/manifest.csv --log-dir $D/trianel_newcomer"
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
  trianel)   # real Trianel parks (configs/trianel, make_fl_scenarios.py --part trianel), small runs: one host
    TRI="--manifest configs/trianel/manifest.csv --log-dir $D/trianel"
    $P $Q $TRI --scenarios trianel_base --methods local cl80 --gpus 2-3 --per-gpu 6
    $P $Q $TRI --scenarios trianel_base $FL
    $P $Q $TRI --scenarios trianel_lopo --methods cl80 --gpus 2-3
    $P $Q $TRI --scenarios trianel_lopo $FL ;;
  trianel_newcomer_l2)   # onboarding curve on the real parks (configs/trianel_newcomer, --part trianel_newcomer)
    $P $Q $TRN --scenarios trianel_newcomer --methods local cl80 --gpus 2-3 --per-gpu 6 ;;
  trianel_newcomer_l1a)
    $P $Q $TRN --scenarios trianel_newcomer --methods fl_fedgradient ;;
  trianel_newcomer_l1b)
    $P $Q $TRN --scenarios trianel_newcomer --methods fl_fedgradient --reverse ;;
  newcomer_l1)   # onboarding curve (make_fl_scenarios.py --part newcomer): FL on l1
    $P $Q --manifest configs/parks_v1/scenarios_newcomer/manifest.csv --log-dir $D/fl_newcomer --scenarios newcomer $FL ;;
  newcomer_l2)   # local (16 clients) and central on l2
    $P $Q --manifest configs/parks_v1/scenarios_newcomer/manifest.csv --log-dir $D/fl_newcomer --scenarios newcomer \
          --methods local cl80 --gpus 0-3 --per-gpu 4 ;;
  physics)       # physics features nwp_power_curve / nwp_wind_hub (--part physics): FL, central and local on one host
    PHY="--manifest configs/parks_v1/scenarios_physics/manifest.csv --log-dir $D/fl_physics"
    $P $Q $PHY --scenarios physics --methods local cl80 --gpus 0-3 --per-gpu 2
    $P $Q $PHY --scenarios physics $FL ;;
  largebatch)    # central x4 with the FedGradient optimisation regime (--part largebatch)
    $P $Q --manifest configs/parks_v1/scenarios_regular/manifest.csv --log-dir $D/fl_regular --scenarios largebatch \
          --methods cl80 --gpus 1 ;;
  *) echo "usage: $0 l2|l1a|l1b|trianel|newcomer_l1|newcomer_l2|largebatch"; exit 1 ;;
esac
