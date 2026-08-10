"""Erzeugt die Trockenlauf-Configs des TFT-Benchmarks (Standard-Hyperparameter).

Pendant zu gen_stdhp_configs.py fuer DCRNN/MTGNN/WaveNet: die Kette
Retrain -> Evaluation wird einmal end-to-end mit festen Hyperparametern statt
HPO-Best-Params durchgerechnet, waehrend die HPO noch rechnet. Der Retrain- und
der Eval-Pfad muessen dafuer beide ohne --hpo-study laufen.

Quelle sind die reraeumlichen Fold-Configs config_wind_tft_sp_{base,hist}_fold{1,2,3}.yaml.
Geaendert wird ausschliesslich params.next_n_stations (5 -> 4); alles andere ist
bereits Standard und in base wie hist identisch:
  hidden_dim 64, n_heads 4, dropout 0.1, LSTM 1 Layer, static_embedding_dim 32,
  clipnorm 5, lr 3e-4, weight_decay 1e-5, batch 128, 60 Epochen, ES patience 15,
  next_n_grid_points 4, next_n_grid_ecmwf 4.

Der Config-Name bestimmt den model_tag (train_tft_bc_m-tft_c-<name>), deshalb das
eigene stdhp-Verzeichnis: die Trockenlauf-Modelle kollidieren so nicht mit einem
spaeteren HPO-Retrain aus denselben Fold-Configs.

Aufruf im Repo-Root:
    python -m geostatistics.stdrun.gen_tft_stdhp_configs [--next-n-stations 4]
"""
import argparse
import os

import yaml

SRC_DIR = "configs/tft_bc"
OUT_DIR = "configs/tft_bc/stdhp"
VARIANTS = ("base", "hist")
FOLDS = (1, 2, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--next-n-stations", type=int, default=4,
                    help="Feste Zahl Nachbarstationen (HPO-Bereich [0, 8]); Default 4")
    ap.add_argument("--next-n-grid-points", type=int, default=4,
                    help="Feste Zahl ICON-D2-Gitterpunkte (HPO-Bereich [1, 7]); Default 4")
    ap.add_argument("--next-n-grid-ecmwf", type=int, default=4,
                    help="Feste Zahl ECMWF-Gitterpunkte (HPO-Bereich [0, 4]); Default 4")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for v in VARIANTS:
        for f in FOLDS:
            src = os.path.join(SRC_DIR, f"config_wind_tft_sp_{v}_fold{f}.yaml")
            with open(src) as fh:
                cfg = yaml.safe_load(fh)

            p = cfg["params"]
            before = (p.get("next_n_grid_points"), p.get("next_n_grid_ecmwf"), p.get("next_n_stations"))
            p["next_n_grid_points"] = args.next_n_grid_points
            p["next_n_grid_ecmwf"] = args.next_n_grid_ecmwf
            p["next_n_stations"] = args.next_n_stations
            after = (p["next_n_grid_points"], p["next_n_grid_ecmwf"], p["next_n_stations"])

            # Ohne Studie zieht der Retrain die Architektur aus model/; die hpo-Bereiche
            # bleiben unangetastet, damit die Datei gegen die Kampagnen-Config diffbar bleibt.
            assert cfg["hpo"]["cv_mode"] == "spatial", cfg["hpo"].get("cv_mode")
            assert cfg["hpo"]["kfolds"] == 1, cfg["hpo"]["kfolds"]

            dst = os.path.join(OUT_DIR, f"config_wind_tft_sp_{v}_stdhp_fold{f}.yaml")
            with open(dst, "w") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
            written.append((dst, before, after,
                            len(cfg["data"]["files"]), len(cfg["data"]["val_files"])))

    print(f"{len(written)} Configs nach {OUT_DIR}/:")
    for dst, before, after, n_tr, n_va in written:
        print(f"  {os.path.basename(dst):48} next_n {before} -> {after}   "
              f"{n_tr} train / {n_va} target")


if __name__ == "__main__":
    main()
