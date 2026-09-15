#!/usr/bin/env python3
"""Configs fuer den Vergleich lokal / global-transduktiv / global-induktiv.

Frage: Bringt ein Modell je Station mehr als ein gemeinsames Modell ueber viele
Stationen? Der naive Vergleich (lokal gegen den vorhandenen Fold-1-Lauf)
vermischt dabei zwei Dinge, die nichts miteinander zu tun haben:

* **Transduktivitaet** — das lokale Modell hat seine Zielstation im Training
  gesehen, der Fold-1-Lauf nicht (die 21 Zielstationen sind dort zero-shot).
* **Datenmenge** — das lokale Modell sieht rund 1/62 der Trainingsfenster.

Deshalb drei Arme:

===================  ======================  =========================
Arm                  Training                Auswertung
===================  ======================  =========================
``lokal``            je 1 Station            dieselbe Station
``global_trans``     alle 62 Stationen       dieselben 62
``global_induk``     41 Pool-Stationen       21 andere  (liegt vor als
                                             ``configs/solar_tft/``
                                             ``config_solar_tft_fold1``)
===================  ======================  =========================

Die beiden ersten Arme stehen auf **derselben Stationsmenge und derselben
Zeitachse**; der einzige Unterschied ist, ob ein gemeinsames Modell ueber alle
62 gelernt wird oder 62 einzelne. Damit misst ihr Abstand den Pooling-Effekt
und sonst nichts.

Der dritte Arm ist der vorhandene Fold-1-Lauf. Er steht auf einer anderen
Stationsmenge (41 Training, 21 Auswertung) und ist deshalb **kein** direkter
Vergleichspartner, sondern der Bezugspunkt fuer die Frage, was die
Generalisierung auf unbekannte Stationen kostet — dort sind die
Auswertungsstationen zero-shot.

Alle drei teilen Zeitachse, Features, Lueckenfuellung und ``exclude_imputed``
mit dem Fold-1-Lauf; die Vorlage ist dessen Config. Training bis
``test_start`` (2024-08-01), Auswertung bis ``test_end`` (2025-08-01) — das
Testjahr bleibt unberuehrt.

Eine Station steht im transduktiven Fall in **beiden** Listen. Das ist Absicht
und kein Versehen: ``train_cl.py`` zieht die Trainingssequenzen aus ``files``
und die Testsequenzen aus ``val_files``, beide ueber dieselbe Zeitgrenze
getrennt und mit denselben, auf den Trainingsdaten gefitteten Scalern. Damit
laeuft der lokale Arm durch exakt denselben Codepfad wie die globalen.

Aufruf:
    frcst/bin/python scripts/make_solar_lokal_configs.py [--force]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

VORLAGE = REPO / 'configs/solar_tft/config_solar_tft_fold1.yaml'
FOLDS = REPO / 'configs/solar_folds.yaml'
ZIEL = REPO / 'configs/solar_lokal'


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


def _fold1() -> tuple[list[str], list[str]]:
    with open(FOLDS) as fh:
        raw = yaml.safe_load(fh)
    f = raw['spatial_fold1']
    return ([str(s).zfill(5) for s in f['files']],
            [str(s).zfill(5) for s in f['val_files']])


def _dateiname(sid: str) -> str:
    """Stationsliste im Format, das die Vorlage nutzt (synth_<id>.csv oder <id>)."""
    return sid


def _schreibe(cfg: dict, pfad: Path, kopf: str, force: bool) -> bool:
    if pfad.exists() and not force:
        print(f'  uebersprungen (existiert): {pfad.relative_to(REPO)}')
        return False
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(kopf + '\n\n' + yaml.dump(cfg, sort_keys=False,
                                              allow_unicode=True, width=100))
    print(f'  geschrieben: {pfad.relative_to(REPO)}')
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    vorlage = _lade_vorlage()
    train_ids, ziel_ids = _fold1()
    alle = train_ids + ziel_ids

    d = vorlage['data']
    zeit = (d.get('train_start'), d.get('test_start'), d.get('test_end'))
    print(f"Vorlage: {VORLAGE.relative_to(REPO)}")
    print(f"Zeitachse: Training ab {zeit[0]}, Auswertung {zeit[1]} … {zeit[2]}")
    print(f"Fold 1: {len(train_ids)} Pool-, {len(ziel_ids)} Zielstationen\n")

    n = 0
    # ── Arm 1: ein Modell je Zielstation ─────────────────────────────────
    print(f'lokal ({len(alle)} Stationen):')
    for sid in alle:
        cfg = copy.deepcopy(vorlage)
        cfg['data']['files'] = [_dateiname(sid)]
        cfg['data']['val_files'] = [_dateiname(sid)]
        cfg['eval']['results_path'] = 'results/solar_lokal'
        kopf = (f"# Solar-TFT, LOKAL — ein Modell nur fuer Station {sid} (1 von 62).\n"
                f"#\n"
                f"# Training und Auswertung auf derselben Station, getrennt allein durch die\n"
                f"# Zeitgrenze test_start. Transduktiv: die Station ist im Training bekannt,\n"
                f"# ein Schluss auf unbekannte Stationen ist daraus NICHT moeglich.\n"
                f"#\n"
                f"# Erzeugt von scripts/make_solar_lokal_configs.py — nicht von Hand pflegen.")
        n += _schreibe(cfg, ZIEL / f'config_solar_lokal_{sid}.yaml', kopf, args.force)

    # ── Arm 2: ein globales Modell, das die Zielstationen kennt ──────────
    print('global-transduktiv:')
    cfg = copy.deepcopy(vorlage)
    cfg['data']['files'] = [_dateiname(s) for s in alle]
    cfg['data']['val_files'] = [_dateiname(s) for s in alle]
    cfg['eval']['results_path'] = 'results/solar_lokal'
    kopf = (f"# Solar-TFT, GLOBAL-TRANSDUKTIV — ein Modell ueber alle {len(alle)} Stationen,\n"
            f"# ausgewertet auf denselben {len(alle)} Stationen im Validierungsjahr.\n"
            f"#\n"
            f"# Alle Stationen stehen bewusst in files UND val_files: Trainingssequenzen\n"
            f"# kommen aus dem Zeitraum vor test_start, Testsequenzen danach. Gegenstueck zu\n"
            f"# den {len(alle)} lokalen Modellen — gleiche Stationen, gleiche Zeitachse, der\n"
            f"# Abstand misst allein den Pooling-Effekt.\n"
            f"#\n"
            f"# Erzeugt von scripts/make_solar_lokal_configs.py — nicht von Hand pflegen.")
    n += _schreibe(cfg, ZIEL / 'config_solar_global_trans.yaml', kopf, args.force)

    print(f"\n{n} Config(s) geschrieben nach {ZIEL.relative_to(REPO)}.")
    print(f"Der dritte Arm (global-induktiv) ist bereits ausgewertet: "
          f"results/solar/cl_m-tft_*_solar_tft_fold1_*.pkl")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
