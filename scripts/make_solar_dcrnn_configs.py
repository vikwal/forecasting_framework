#!/usr/bin/env python3
"""Configs der Solar-DCRNN-Ablationsleiter, analog zur Wind-Leiter.

Sechs Arme, jeder unterscheidet sich vom Referenzarm A in **einer** Einstellung
— dieselbe Konstruktion wie bei Wind (docs/study_overview.md, Abschnitt
„Ablationsvarianten B und C"):

    A        Referenz: voller Graph, Nachbarmessungen, GATv2 ueber NWP-Knoten
    base     nwp_nodes: false — k naechste Gitterpunkte direkt in station.x
    nomeas   neighbour_meas_available: false — keine Station traegt Messungen
    nograph  zusaetzlich station_connectivity: none — leeres Kantenset
    idw_alt  nwp_aggregation: idw_alt — Distanzgewichtung mit Hoehenkorrektur
    nwp_hist hist_wind_available: true — Zielstationen behalten ihre Historie

Die tragenden Differenzen sind ``A − nomeas`` (Wert der Nachbarmessungen) und
``nomeas − nograph`` (Wert des Geometrie- und Kontextkanals).

Unterschiede zur Wind-Leiter, alle aus den Solar-Entscheidungen vom 14.09.2026:

* **30-min-Raster**, also ``history_length``/``forecast_horizon`` 96 statt 48.
  Die Aufloesungs-Ablation vom August hat 30 min klar vor 15 und 10 min
  gestellt, und nur so sind die Zahlen mit den TFT-Laeufen vergleichbar.
* **Zwei Zielgroessen** (``target_col: [ghi, dhi]``) statt einer. Der GNN-Pfad
  kann das seit dem Multi-Target-Umbau; skill_nwp bekommt je Ziel seine eigene
  Referenzspalte (ghi_nwp / dhi_nwp).
* **Lueckenfuellung zwingend.** Ohne ``interpol_path`` sind ueber die 83 Pool-
  und Teststationen nur 11.4 % der Zeitschritte vollstaendig und der laengste
  zusammenhaengende Block misst 31 Schritte — bei 96+96 gebrauchten gibt das
  null nutzbare Fenster. Mit Imputation sind es 100 % und ein durchgehender
  Block ueber beide Jahre.
* **ECMWF als zweite NWP-Quelle**, wie beim TFT. Ohne sie misst der Vergleich
  zwischen den Architekturen nur den Featuresatz.
* **``target_transform: nwp_residual``**, ebenfalls wie beim TFT: Ziel und
  Messhistorie stehen als Abweichung von der ICON-D2-Prognose. Die Auswertung
  rechnet zurueck, die Metriken bleiben in W/m².
* **Fold 1 only.** Die Ablationen laufen auf Fold 1, die drei Folds sind erst
  fuer die HPO vorgesehen (Entscheidung Viktor, 14.09.2026).

Ein Unterschied zur TFT-Zeitachse bleibt und ist bewusst: der GNN-Pfad kennt
kein ``train_start`` und trainiert ab Datenbeginn (2023-07-24) statt ab
2023-08-01. Das sind acht Tage mehr, ueber alle Arme gleich — fuer den
Vergleich innerhalb der Leiter folgenlos, beim Halten gegen die TFT-Zahlen zu
erwaehnen.

Aufruf:
    frcst/bin/python scripts/make_solar_dcrnn_configs.py [--force]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

VORLAGE = REPO / 'configs/dcrnn/config_solar_dcrnn.yaml'
FOLDS = REPO / 'configs/solar_folds.yaml'
ZIEL = REPO / 'configs/solar_dcrnn'

#: Je Arm die Abweichung vom Referenzarm A, als Pfad in die dcrnn-Sektion.
ARME: dict[str, dict] = {
    'a':        {},
    'base':     {'nwp_nodes': False},
    'nomeas':   {'neighbour_meas_available': False},
    'nograph':  {'neighbour_meas_available': False, 'station_connectivity': 'none',
                 'direction_to_adj': False},
    'idw_alt':  {'nwp_aggregation': 'idw_alt'},
    'nwp_hist': {'hist_wind_available': True},
}

BESCHREIBUNG = {
    'a':        'Referenz: voller Graph, Nachbarmessungen, GATv2 ueber NWP-Knoten',
    'base':     'nwp_nodes=false — k naechste Gitterpunkte direkt in station.x',
    'nomeas':   'Ablation B: keine Station traegt Messungen',
    'nograph':  'Ablation C: zusaetzlich leeres station<->station Kantenset',
    "idw_alt":  "Ablation D': Distanzgewichtung mit Hoehenkorrektur",
    'nwp_hist': 'Zielstationen behalten ihre eigene Messhistorie',
}


class EnvTag:
    def __init__(self, value):
        self.value = value


yaml.add_representer(EnvTag, lambda d, x: d.represent_scalar('!ENV', x.value, style="'"))


def _lade_vorlage() -> dict:
    def ctor(loader, node):
        return EnvTag(loader.construct_scalar(node))
    yaml.add_constructor('!ENV', ctor, Loader=yaml.SafeLoader)
    with open(VORLAGE) as fh:
        return yaml.load(fh, Loader=yaml.SafeLoader)


def _fold1() -> tuple[list[str], list[str], list[str]]:
    with open(FOLDS) as fh:
        raw = yaml.safe_load(fh)
    f = raw['spatial_fold1']
    return ([str(s).zfill(5) for s in f['files']],
            [str(s).zfill(5) for s in f['val_files']],
            [str(s).zfill(5) for s in raw['test_files']])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    vorlage = _lade_vorlage()
    train_ids, val_ids, test_ids = _fold1()

    for arm, abweichung in ARME.items():
        cfg = copy.deepcopy(vorlage)
        d, g = cfg['data'], cfg['dcrnn']

        # --- Stationen und Zeitachse ---------------------------------
        d['files'] = train_ids
        d['val_files'] = val_ids
        d['test_files'] = test_ids
        d['freq'] = '30min'
        d['val_start'] = '2024-08-01'     # Training davor, Validierung danach
        d['test_start'] = '2025-08-01'    # zurueckgehalten
        d['test_end'] = '2026-07-31'      # Ende von ICON-D2 SL und der Imputation
        d['interpol_path'] = EnvTag('${DATA_ROOT}/synthetic/interpol/solar')
        d.pop('knnimputer_path', None)    # fuer Solar gibt es keinen KNN-Baum

        # --- ECMWF als zweite NWP-Quelle -----------------------------
        # Die Config-Notiz „ECMWF disabled for solar (no surface radiation
        # fields)" stammt vom August und ist ueberholt: seither liegen 759
        # Gitterpunkte unter ecmwf/parquet/solar. Ohne ECMWF misst ein
        # Vergleich DCRNN gegen TFT den Featuresatz statt der Architektur —
        # in der Augustablation war ECMWF mit -3.9 % der einzige grosse Hebel,
        # alles andere bewegte sich im Rauschen.
        #
        # Featurenamen identisch zum TFT; die Ableitungen kommen in beiden
        # Pfaden aus utils.solar_ecmwf, damit 'ecmwf_dhi' hier und dort
        # dasselbe bedeutet.
        d['ecmwf_path'] = EnvTag('${DATA_ROOT}/ecmwf/parquet/solar')
        g['ecmwf_features'] = ['ecmwf_ghi', 'ecmwf_dhi', 'ecmwf_bhi', 'ecmwf_toa',
                               'ecmwf_bhi_clearsky', 'ecmwf_t_2m', 'ecmwf_tp', 'ecmwf_fal']
        # 4 Gitterpunkte wie im Wind-DCRNN; der TFT nimmt einen. Die
        # Augustablation hat das gemessen und keinen Unterschied gefunden
        # (ab_ecmwf 75.16-75.49 gegen ab_ecmwf_g4 75.11-75.52 W/m²), die Zahl
        # ist also folgenlos — und die GATv2-Attention ueber NWP-Knoten
        # braucht mehr als einen Knoten, um ueberhaupt etwas zu tun.
        g['next_n_ecmwf'] = 4

        # --- Zielgroessen und Raster ---------------------------------
        g['target_col'] = ['ghi', 'dhi']
        g['history_length'] = 96
        g['forecast_horizon'] = 96
        g['impute_night_zero'] = True
        # Residuum gegen ICON-D2 wie im TFT-Pfad. Die Metriken bleiben
        # vergleichbar, weil die Auswertung zurueckrechnet — der Unterschied
        # ist der Induktivbias: das Modell muss nur die Korrektur lernen.
        g['target_transform'] = 'nwp_residual'

        # Alle vier ICON-D2-Laeufe, wie der TFT sie nutzt. Die Solar-Vorlage
        # stand auf [6] — ein Ueberbleibsel aus der Zeit, als der GNN-Pfad durch
        # die Messluecken blockiert und daher nie gelaufen war. Folge: das DCRNN
        # sah 1100 Laeufe ueber drei Jahre, also einen pro Tag, waehrend der TFT
        # mit vieren je Tag trainiert. Gemessen 349 nutzbare Trainings-Run-Paare
        # gegen 55624 TFT-Fenster.
        g['icond2_run_hours'] = [6, 9, 12, 15]

        # K_hop 2 statt 1: die Diffusionstiefe des Wind-DCRNN, die dort eine HPO
        # hinter sich hat. Fuer Solar gibt es keine; der Entwurfswert 1 ist
        # schlechter begruendet als der erprobte.
        g['K_hop'] = 2
        # handle_nans bleibt 'break': nach der Imputation darf keine Luecke mehr
        # kommen, und wenn doch, soll der Lauf abbrechen statt Stationen
        # stillschweigend zu verwerfen.
        g['handle_nans'] = 'break'

        for schluessel, wert in abweichung.items():
            g[schluessel] = wert

        pfad = ZIEL / f'config_solar_dcrnn_{arm}_fold1.yaml'
        if pfad.exists() and not args.force:
            print(f'  uebersprungen (existiert): {pfad.relative_to(REPO)}')
            continue
        pfad.parent.mkdir(parents=True, exist_ok=True)
        kopf = (f"# Solar-DCRNN, Arm '{arm}' auf Fold 1 — {BESCHREIBUNG[arm]}\n"
                f"#\n"
                f"# {len(train_ids)} Trainings-, {len(val_ids)} Zielstationen aus "
                f"configs/solar_folds.yaml.\n"
                f"# Abweichung von Arm A: "
                f"{abweichung if abweichung else 'keine (A ist die Referenz)'}\n"
                f"#\n"
                f"# Erzeugt von scripts/make_solar_dcrnn_configs.py — nicht von Hand pflegen.")
        pfad.write_text(kopf + '\n\n' + yaml.dump(cfg, sort_keys=False,
                                                  allow_unicode=True, width=100))
        print(f'  geschrieben: {pfad.relative_to(REPO)}')

    print(f'\nFold 1: {len(train_ids)} train / {len(val_ids)} Ziel, '
          f'{len(test_ids)} Teststationen zurueckgehalten.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
