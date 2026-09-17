#!/usr/bin/env python3
"""Configs der Solar-TFT-Kampagne aus configs/solar_folds.yaml erzeugen.

Zwei Arme, die sich NUR im Trainingspool unterscheiden — Val- und
Teststationen sind in beiden identisch, sonst waere der Vergleich nicht
gepaart:

    solar_tft       62 Poolstationen, wie in docs/station_splits_solar.md
    solar_tft_plus  62 + 9 Stationen, die erst durch die Imputation vom
                    2026-09-03 nutzbar wurden (interpol/solar, 94 Stationen)

Die 9 sind die Teilmenge der 11 neuen, die im TRAININGSFENSTER
2023-08..2024-07 tatsaechlich messen. Gemessener Anteil echter Werte dort:

    00427 .999   01078 .994   01605 .997   01766 .981   02907 .999
    03028 .999   03126 1.000  05629 .950   05839 .985
    ---- ausgeschlossen ----
    04642 .000   (Messreihe beginnt erst 2025-06)
    04887 .413   (unter der 50-%-Schwelle aus station_splits_solar.md §5)

Diese neun sind im Trainingsfenster also keine Imputationsstationen, sondern
schlicht neun weitere Messstationen — ihr Ausfall liegt im Val- und Testjahr,
wo sie in keinem Arm Zielstation sind.

Je Arm entstehen:

    config_solar_tft.yaml           HPO ueber die 3 raeumlichen Folds
    config_solar_tft_fold{1,2,3}.yaml   Retrain je Fold
    config_solar_tft_testyear.yaml  Schlussmessung auf den 21 Teststationen

Kein Suffix hinter ``_fold<N>`` — die Optuna-Studienaufloesung leitet den
Namen aus dem Dateinamen ab, Varianten laufen ueber das Verzeichnis.

Aufruf:
    frcst/bin/python scripts/make_solar_tft_configs.py [--force]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
VORLAGE = REPO / 'configs/solar_final/config_solar_final_lag.yaml'
FOLDS = REPO / 'configs/solar_folds.yaml'

#: Stationen, die erst durch interpol/solar dazukommen UND im Trainingsfenster
#: messen. Begruendung und Zahlen im Modul-Docstring.
NEU_IM_TRAINING = ['00427', '01078', '01605', '01766', '02907',
                   '03028', '03126', '05629', '05839']

#: Zeitfenster. train_start bleibt 2023-08-01: ICON-D2 SL beginnt am
#: 2023-07-24, der 15-UTC-Lauf erst am 2023-08-08 — einen Juli 2023 gibt es
#: fuer diesen Aufbau nicht.
FENSTER_CV = dict(train_start='2023-08-01', train_end='2024-07-31',
                  test_start='2024-08-01', test_end='2025-08-01')
#: Retrain je Fold: dieselbe Zeitachse wie die gefahrene HPO
#: (configs/solar_tft/config_solar_tft_hpo.yaml). cv_mode='spatial' schneidet
#: das Trainingsfenster bei val_start und das Val-Fenster bei test_start —
#: beides aus X_train, weshalb hier KEIN train_end stehen darf: es begrenzt
#: df_train (utils/preprocessing.py:412) und liesse das Val-Fenster leer.
#: test_end ist nur die obere Ladeschranke der Rohdaten; das Testjahr landet
#: in X_test und wird vom spatial-Pfad nie angefasst (Training < val_start,
#: Scaler _fit_global_scaler_x(fit_until=val_start)).
FENSTER_FOLD = dict(train_start='2023-08-01', val_start='2024-08-01',
                    test_start='2025-08-01', test_end='2026-07-31')
#: Schlussmessung: Training ueber beide bisherigen Jahre, Test auf dem
#: zurueckgehaltenen dritten. NICHT starten, solange die Modellwahl laeuft.
#:
#: Gleiche Bauart wie FENSTER_FOLD, nur ein Jahr weiter: val_start trennt
#: Training (< 2025-08-01, also beide bisherigen Jahre) vom Val-Fenster
#: [val_start, test_start) — und das IST das Testjahr. Early Stopping laeuft
#: damit auf denselben 21 Stationen und demselben Zeitraum, auf denen
#: anschliessend berichtet wird. Das ist die festgelegte Konvention, nicht
#: ein Versehen: docs/station_splits_solar.md §6 misst den Optimismus auf rund
#: ein Prozent und haelt ihn fuer vernachlaessigbar (Entscheidung Viktor,
#: 18.08.2026). Ein Val-Chunk im Trainingszeitraum waere die strengere
#: Variante, wich aber von Arm A und den Fold-Laeufen ab und machte die Zahlen
#: untereinander unvergleichbar.
FENSTER_TEST = dict(train_start='2023-08-01', val_start='2025-08-01',
                    test_start='2026-08-01', test_end='2026-08-01')

ARME = {
    'solar_tft':      {'extra': [],              'label': '62 Poolstationen'},
    'solar_tft_plus': {'extra': NEU_IM_TRAINING, 'label': '62 + 9 Stationen aus der Imputation'},
}


def _yaml_env_repr(dumper, data):
    return dumper.represent_scalar('!ENV', data.value, style="'")


class EnvTag:
    def __init__(self, value):
        self.value = value


yaml.add_representer(EnvTag, _yaml_env_repr)


def _lade_vorlage() -> dict:
    """Vorlage roh laden — !ENV-Knoten bleiben als EnvTag erhalten.

    Bewusst NICHT ueber utils.tools.load_config: das loest ${DATA_ROOT} auf und
    schriebe den Mountpfad dieses Hosts in die Datei. Die Configs muessen
    hostunabhaengig bleiben (CLAUDE.md, Commit ba6e567).
    """
    def env_ctor(loader, node):
        return EnvTag(loader.construct_scalar(node))

    loader = yaml.SafeLoader
    yaml.add_constructor('!ENV', env_ctor, Loader=loader)
    with open(VORLAGE) as fh:
        return yaml.load(fh, Loader=loader)


def _folds() -> tuple[list[str], list[tuple[str, list[str], list[str]]]]:
    with open(FOLDS) as fh:
        raw = yaml.safe_load(fh)
    test = [str(s).zfill(5) for s in raw['test_files']]
    folds = []
    for key in sorted(k for k in raw if k.startswith('spatial_fold')):
        folds.append((key,
                      [str(s).zfill(5) for s in raw[key]['files']],
                      [str(s).zfill(5) for s in raw[key]['val_files']]))
    return test, folds


def _grundgeruest(vorlage: dict, arm: str, extra: list[str]) -> dict:
    cfg = copy.deepcopy(vorlage)
    d, p, e = cfg['data'], cfg['params'], cfg['eval']

    # Lueckenfuellung aus dem Solar-Abschlussmodell (2026-09-03, 94 Stationen,
    # 30-min-Raster). utils/solar.py fuellt damit VOR dem dropna(); die
    # Auswertung nimmt die gefuellten Positionen ueber eval.exclude_imputed
    # wieder heraus.
    d['interpol_path'] = EnvTag('${DATA_ROOT}/synthetic/interpol/solar')
    p['impute_night_zero'] = True

    # Featuresatz der HPO nachziehen (Suche vom 16./17.09.2026, Studie
    # cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo). Ohne das trainiert der
    # Retrain ein anderes Modell als das, dessen Hyperparameter gesucht wurden.
    #   u_10m  staerkster Screening-Kandidat (DHI 0.161 %), ans ENDE der Listen,
    #          genau dort haengt hpo_tft_bc.py optionale Features an — so bleibt
    #          der Cache-Schluessel identisch und der Eintrag wiederverwendbar.
    #   next_n_grid_ecmwf  explizit 0 statt weggelassen: hpo_tft_bc._range faellt
    #          ohne hpo-Range auf params zurueck und der Trial schreibt den Wert
    #          nach config['params'], wo data_cache._get_config_hash ihn liest.
    #          Fehlt der Schluessel, hasht die Config None und der Trial 0 —
    #          zwei Cache-Schluessel fuer dieselben Daten.
    for schluessel in ('icond2_features', 'known_features'):
        if 'u_10m' not in p[schluessel]:
            p[schluessel] = list(p[schluessel]) + ['u_10m']
    p['next_n_grid_ecmwf'] = 0
    e['exclude_imputed'] = True
    e['results_path'] = f'results/{arm}'
    cfg['_arm'] = {'name': arm, 'zusatzstationen': list(extra)}
    return cfg


def _schreibe(cfg: dict, pfad: Path, kopf: str, force: bool) -> bool:
    if pfad.exists() and not force:
        print(f'  uebersprungen (existiert): {pfad.relative_to(REPO)}')
        return False
    pfad.parent.mkdir(parents=True, exist_ok=True)
    meta = cfg.pop('_arm', None)
    body = yaml.dump(cfg, sort_keys=False, allow_unicode=True, width=100)
    pfad.write_text(kopf.rstrip() + '\n\n' + body)
    if meta is not None:
        cfg['_arm'] = meta
    print(f'  geschrieben: {pfad.relative_to(REPO)}')
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='vorhandene Configs ueberschreiben')
    args = ap.parse_args()

    vorlage = _lade_vorlage()
    test_ids, folds = _folds()
    pool = sorted({s for _, tr, va in folds for s in tr + va})

    fehlend = [s for s in NEU_IM_TRAINING if s in pool or s in test_ids]
    if fehlend:
        sys.exit(f'FEHLER: {fehlend} sind bereits im Pool/Testsatz — der Arm waere nicht disjunkt.')

    for arm, spez in ARME.items():
        extra = spez['extra']
        print(f'\n{arm} ({spez["label"]}):')
        ziel = REPO / 'configs' / arm

        # --- HPO ueber die drei raeumlichen Folds -----------------------
        cfg = _grundgeruest(vorlage, arm, extra)
        cfg['data'].update(FENSTER_CV)
        cfg['data']['files'] = pool + extra
        cfg['data']['val_files'] = list(test_ids)   # Platzhalter; die Folds ueberschreiben ihn
        cfg['data'].pop('test_files', None)
        cfg['hpo']['cv_mode'] = 'spatial'
        cfg['hpo']['spatial_folds'] = 'configs/solar_folds.yaml'
        cfg['hpo']['extra_train_files'] = list(extra)
        _schreibe(cfg, ziel / f'config_{arm}.yaml', f"""# {arm} — HPO ueber die drei raeumlichen Folds aus configs/solar_folds.yaml
#
# Trainingspool: {spez['label']}.
# Die Stationsrollen je Fold setzt hpo_tft_bc.py aus spatial_folds; 'files' hier
# ist der Gesamtpool, aus dem gezogen wird. hpo.extra_train_files nennt die
# Stationen, die in JEDEM Fold Trainingsrolle haben und nie Zielstation werden.
#
# Erzeugt von scripts/make_solar_tft_configs.py — nicht von Hand pflegen.""", args.force)

        # --- Retrain je Fold --------------------------------------------
        for i, (_, train_ids, val_ids) in enumerate(folds, start=1):
            cfg = _grundgeruest(vorlage, arm, extra)
            cfg['data'].update(FENSTER_FOLD)
            # train_end der Vorlage MUSS weg: es begrenzt df_train
            # (utils/preprocessing.py:412), und cv_mode='spatial' schneidet sein
            # Val-Fenster [val_start, test_start) genau aus X_train heraus —
            # mit train_end 2024-07-31 bliebe es leer und der Lauf braeche in
            # _build_spatial_fold_data ab ("empty train or val split").
            cfg['data'].pop('train_end', None)
            cfg['data']['files'] = sorted(train_ids + extra)
            cfg['data']['val_files'] = list(val_ids)
            # test_files = val_files: die Zielstationen des Folds sind zugleich
            # die Auswertungsstationen. Ausgewertet wird mit
            # `get_test_results_tft_bc.py --eval-split val`, das test_files auf
            # val_files und das Fenster auf [val_start, test_start) setzt — also
            # auf das Validierungsjahr, in dem auch das Early Stopping misst.
            # Der zurueckgehaltene Testsatz bleibt unberuehrt; OHNE --eval-split
            # val liefe die Auswertung dagegen im Testjahr.
            cfg['data']['test_files'] = list(val_ids)
            # Dieselbe CV-Achse wie die gefahrene HPO: feste Zeitgrenze
            # val_start, rotierende Stationsrollen. train_cl_tft_bc.py waehlt
            # ueber hpo.cv_mode zwischen create_or_load_preprocessed_data
            # (temporal) und ..._spatial und verlangt genau einen Fold — mit dem
            # kfolds 12 der Vorlage braeche es nach dem kompletten Preprocessing ab.
            cfg['hpo']['cv_mode'] = 'spatial'
            cfg['hpo']['kfolds'] = 1
            cfg['hpo']['min_train_date'] = None
            _schreibe(cfg, ziel / f'config_{arm}_fold{i}.yaml', f"""# {arm}, Fold {i} — {len(train_ids)}+{len(extra)} Trainings-, {len(val_ids)} Zielstationen
#
# Trainingspool: {spez['label']}.
# Zielstationen identisch zum jeweils anderen Arm — nur so ist der Vergleich
# der beiden Arme gepaart.
#
# cv_mode: spatial, Zeitachse wie in der HPO-Studie
# cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo: Training < val_start
# (2024-08-01), Early Stopping auf den Zielstationen des Folds im
# Validierungsjahr [val_start, test_start). Kein train_end — siehe
# FENSTER_FOLD im Generator.
#
# Auswertung ZWINGEND mit --eval-split val:
#   get_test_results_tft_bc.py -c <diese Datei> \\
#       --hpo-study cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo \\
#       --model-tag train_tft_bc_m-tft_c-{arm}_fold{i} --eval-split val
# Ohne das Flag misst die Auswertung im Testjahr (test_start 2025-08-01).
#
# Erzeugt von scripts/make_solar_tft_configs.py — nicht von Hand pflegen.""", args.force)

        # --- Schlussmessung ---------------------------------------------
        cfg = _grundgeruest(vorlage, arm, extra)
        cfg['data'].update(FENSTER_TEST)
        cfg['data'].pop('train_end', None)   # begrenzt df_train, s. FENSTER_FOLD
        cfg['data']['files'] = sorted(pool + extra)
        cfg['data']['val_files'] = list(test_ids)
        cfg['data']['test_files'] = list(test_ids)
        # Wie die Fold-Configs: cv_mode spatial, Schnitt auf val_start. Training
        # sind die Poolstationen vor 2025-08-01, Validierung die 21 Teststationen
        # im Testjahr — also die Auswertungsdaten selbst (station_splits_solar.md
        # §6). Ausgewertet wird deshalb ebenfalls mit --eval-split val.
        # --test-mode ist hier verboten (val_files == test_files), s. Kopf.
        cfg['hpo']['cv_mode'] = 'spatial'
        cfg['hpo']['kfolds'] = 1
        cfg['hpo']['min_train_date'] = None
        _schreibe(cfg, ziel / f'config_{arm}_testyear.yaml', f"""# {arm}, Schlussmessung — {len(pool)}+{len(extra)} Trainings-, {len(test_ids)} Teststationen
#
# Training ueber beide bisherigen Jahre, Test auf dem zurueckgehaltenen dritten
# (2025-08..2026-07). NICHT starten, solange die Modellwahl laeuft — der
# Testsatz darf in keine Auswahl einfliessen.
#
# NIEMALS mit --test-mode fahren: das Flag zieht val_files in den Trainingspool,
# und val_files sind hier DIESELBEN 21 Teststationen wie test_files — die
# Schlussmessung waere wertlos. (Seit dem Wegfall von hpo.val_split bricht der
# Lauf in diesem Fall ohnehin ab.) Ohne das Flag trainiert die Config auf den
# Poolstationen bis val_start 2025-08-01 und misst auf den 21 Teststationen im
# Testjahr; Early Stopping laeuft auf genau diesen Auswertungsdaten — so
# festgelegt in docs/station_splits_solar.md §6, Optimismus rund ein Prozent.
#
# Auswertung deshalb mit --eval-split val (Fenster [val_start, test_start)):
#   get_test_results_tft_bc.py -c <diese Datei> \\
#       --hpo-study cl_m-tft-bc_out-96_freq-30min_solar_tft_hpo \\
#       --model-tag train_tft_bc_m-tft_c-<arm>_testyear --eval-split val
#
# Achtung Datenlage: im Testjahr faellt auch der Pool ab (mittlerer Anteil
# echter Messwerte 0.85, Minimum 0.12). Mit eval.exclude_imputed bleibt davon
# entsprechend weniger Auswertungsmasse — vor der Interpretation nachzaehlen.
#
# Erzeugt von scripts/make_solar_tft_configs.py — nicht von Hand pflegen.""", args.force)

    print(f'\nPool {len(pool)}, Testsatz {len(test_ids)}, Zusatzstationen {len(NEU_IM_TRAINING)}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
