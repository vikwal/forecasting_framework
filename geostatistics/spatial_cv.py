"""Raeumliche Kreuzvalidierung fuer die HPO der Graph-Architekturen.

Die HPO-Skripte kannten bisher nur eine zeitliche Expanding-Window-CV: alle
Folds arbeiten auf derselben Stationsmenge und verschieben nur das Zeitfenster
(``hpo.n_folds``, Fold-Grenzen aus ``data.test_start``). Dieses Modul stellt die
Alternative bereit — dieselbe Zeitspanne in jedem Fold, dafuer rotierende
Stationsmengen aus ``configs/spatial_folds.yaml``.

Gesteuert ueber die Config:

.. code-block:: yaml

    mtgnn:            # bzw. dcrnn / wavenet
      hpo:
        cv_mode: spatial              # default: temporal (= bisheriges Verhalten)
        spatial_folds: configs/spatial_folds.yaml

``cv_mode: temporal`` ist der Default, damit bestehende Configs und laufende
Studien unveraendert weiterlaufen.

Im Modus ``spatial`` wird **eine** Stationsmenge geladen (die Vereinigung aller
Folds, sortiert und damit unabhaengig davon, welche Fold-Config uebergeben
wurde) und pro Fold nur umindiziert. Alles, was sonst an der Reihenfolge
"train zuerst" haengt — Scaler, Topo-z-Score —, muss deshalb die explizite
Indexliste des jeweiligen Folds benutzen, nicht ``[:N_train]``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpatialFold:
    """Ein raeumlicher Fold, als Indizes in die geladene Stationsliste."""

    name: str
    train_idx: list[int]
    val_idx: list[int]

    def __len__(self) -> int:
        return len(self.train_idx) + len(self.val_idx)


def _norm(sid) -> str:
    """Stations-IDs kommen aus YAML mal als int, mal als String."""
    return str(sid).zfill(5)


def fold_hash(path: str | Path) -> str:
    """MD5 (erste 12 Hex-Zeichen) der rohen ``spatial_folds.yaml``-Bytes.

    Fuer Provenienz-Logging (Trial-user_attrs / Log): zwei parallele Worker
    derselben Studie, die still unterschiedliche Fold-Definitionen laden
    (Datei zwischen den Starts geaendert), wuerden sonst unbemerkt zwei
    verschiedene Zielmengen in eine Optuna-Studie schreiben — dieselbe
    Fehlerklasse wie der Host-Sync-Blocker aus dem Review-Briefing, nur auf
    Fold-Ebene (topo_features_review_brief.md, 12.7 Punkt 6 / H3).
    """
    import hashlib
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:12]


def load_spatial_folds(path: str | Path) -> list[tuple[str, list[str], list[str]]]:
    """Liest ``spatial_folds.yaml`` -> [(fold_name, train_ids, val_ids), ...]."""
    raw = yaml.safe_load(Path(path).read_text())
    folds = []
    for name in sorted(k for k in raw if k.startswith("spatial_fold")):
        entry = raw[name]
        folds.append((
            name,
            [_norm(s) for s in entry["files"]],
            [_norm(s) for s in entry["val_files"]],
        ))
    if not folds:
        raise ValueError(f"Keine 'spatial_fold*'-Eintraege in {path}")

    # Jede Station muss in jedem Fold genau eine Rolle haben, und die Vereinigung
    # muss ueber alle Folds dieselbe sein — sonst waeren die Folds nicht
    # vergleichbar und der geladene Stationspool haenge davon ab, welche
    # Fold-Config gerade uebergeben wurde.
    pools = [set(t) | set(v) for _, t, v in folds]
    if any(p != pools[0] for p in pools[1:]):
        raise ValueError(f"{path}: die Folds decken unterschiedliche Stationen ab")
    for name, t, v in folds:
        if set(t) & set(v):
            raise ValueError(f"{path}: {name} hat Stationen in files UND val_files")
    return folds


def station_pool(folds: list[tuple[str, list[str], list[str]]]) -> list[str]:
    """Sortierte Vereinigung aller Fold-Stationen — der zu ladende Pool."""
    return sorted(set(folds[0][1]) | set(folds[0][2]))


def build_folds(
    folds: list[tuple[str, list[str], list[str]]],
    all_ids: list[str],
    max_val_stations: int | None = None,
) -> list[SpatialFold]:
    """Uebersetzt die ID-Listen in Indizes in ``all_ids``.

    ``max_val_stations`` (aus ``hpo.n_val_stations``) begrenzt die Anzahl der
    Zielstationen pro Fold; die uebrigen Val-Stationen bleiben dann in diesem
    Fold ungenutzt — sie werden **nicht** zu Trainingsnachbarn, sonst waere die
    Nachbarschaft je nach Einstellung eine andere.
    """
    pos = {sid: i for i, sid in enumerate(all_ids)}
    out = []
    for name, train_ids, val_ids in folds:
        missing = [s for s in train_ids + val_ids if s not in pos]
        if missing:
            raise KeyError(f"{name}: {len(missing)} Stationen nicht im geladenen "
                           f"Pool ({missing[:5]})")
        v = [pos[s] for s in val_ids]
        if max_val_stations is not None and len(v) > max_val_stations:
            # The folds in spatial_folds.yaml already define the exact target
            # set; a set hpo.n_val_stations here silently shrinks it, which is
            # never intended (review brief M3) — fail loudly instead.
            raise ValueError(
                f"{name}: hpo.n_val_stations={max_val_stations} would silently cut "
                f"{len(v) - max_val_stations} of {len(v)} target stations. Under "
                "cv_mode=spatial the folds define their target set completely — "
                "remove n_val_stations or set it to null."
            )
        out.append(SpatialFold(name=name,
                               train_idx=[pos[s] for s in train_ids],
                               val_idx=v))
    return out


def resolve_cv_mode(hpo_cfg: dict) -> tuple[str, str | None]:
    """(cv_mode, spatial_folds_path) aus der ``hpo``-Section.

    Default ist ``temporal`` — bestehende Configs aendern ihr Verhalten nicht.
    """
    mode = str(hpo_cfg.get("cv_mode", "temporal")).lower()
    if mode not in ("temporal", "spatial"):
        raise ValueError(f"hpo.cv_mode muss 'temporal' oder 'spatial' sein, ist '{mode}'")
    if mode == "temporal":
        return mode, None
    return mode, hpo_cfg.get("spatial_folds", "configs/spatial_folds.yaml")
