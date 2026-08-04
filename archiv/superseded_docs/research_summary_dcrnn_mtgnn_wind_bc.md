# Forschungsüberblick: Graph-basierte NWP-Bias-Correction für Windenergie-Prognosen mit DCRNN und MTGNN

**Zweck dieses Dokuments:** Vollständige, in sich geschlossene Zusammenfassung von Motivation, Methodik, Datenbasis und Stand der Umsetzung dieser Studie — als Ausgangspunkt für eine gezielte Literaturrecherche (verwandte Arbeiten, Einordnung der Novelty, mögliche Journal-Ziele) mit einer separaten KI-Session. Enthält bewusst auch Implementierungsdetails, die für Reviewer-Fragen zu Reproduzierbarkeit und methodischer Sauberkeit relevant sein könnten.

---

## 1. Forschungsfrage und Zielsetzung

**Kernproblem:** Numerische Wettervorhersagemodelle (NWP) wie ICON-D2 (DWD) und ECMWF HRES liefern Windprognosen auf einem Gitter, das nicht mit den Standorten von Windenergieanlagen übereinstimmt. Zusätzlich weisen NWP-Modelle systematische Fehler (Bias) gegenüber lokalen Messungen auf, u. a. durch unzureichend aufgelöste Orographie, Landnutzung und Rauigkeit. Für eine belastbare Windenergie-Prognose muss daher (a) räumlich von Gitterpunkten auf Anlagenstandorte interpoliert und (b) der NWP-Bias korrigiert werden — klassischerweise zwei getrennte Schritte (Regression Kriging / MOS, dann Post-Processing).

**Ziel der Arbeit:** Ein einziges, end-to-end lernendes Modell, das NWP-Gitterpunktvorhersagen **direkt** in kalibrierte, standortscharfe Windgeschwindigkeits- bzw. Windenergie-Prognosen übersetzt — im Idealfall **ausschließlich aus NWP-Daten**, d. h. **ohne historische Messungen an der Zielstation** als Modell-Input. Das Modell soll damit für neue, unbeobachtete Standorte ("unseen stations") ohne erneutes Training einsetzbar sein (induktives Setting, keine Kriging-Interpolation und kein Retraining pro Standort nötig).

**Zwei Zielgrößen** werden unterstützt:
- `wind_speed`: reine NWP-Bias-Correction/Downscaling der Windgeschwindigkeit auf Stationshöhe (bzw. Nabenhöhe via Power-Law-Extrapolation) — dient v. a. der Modell-Diagnostik (wie viel Skill gewinnt das Modell gegenüber dem rohen NWP-Wert am nächsten Gitterpunkt?).
- `power`: direkte Prognose der eingespeisten Leistung (synthetisch aus Leistungskurven pro Turbinentyp erzeugt) — die eigentliche betriebsrelevante Zielgröße.

**Anwendungsfall / Motivation aus der Praxis:** Direktvermarktung / Prognosedienstleister für Windparks, für die an sehr vielen (potenziell neuen) Standorten belastbare Kurzfristprognosen (bis 48 h) ohne jahrelange lokale Messhistorie benötigt werden.

---

## 2. Konzeptionelle Novelty

Drei zusammenhängende Neuheitsaspekte werden beansprucht:

1. **DCRNN und MTGNN erstmals für Windenergie-Prognose / NWP-Bias-Correction.** Beide Modelle sind State-of-the-Art-Architekturen aus dem Verkehrsfluss-Forecasting (Traffic Forecasting) — DCRNN (Li et al. 2018, *Diffusion Convolutional Recurrent Neural Network*) und MTGNN (Wu et al. 2020, *Connecting the Dots*) — und wurden nach Kenntnisstand des Autors bisher nicht auf die Windenergie-/NWP-Bias-Correction-Domäne übertragen. Beide sind ursprünglich **transduktiv** (feste, bekannte Knotenmenge mit lernbaren Knoten-Embeddings) und wurden hier für ein **induktives** Setting umgebaut (siehe Abschnitt 4).

2. **Heterogener Graph mit NWP-Gitterpunkten als eigenen Knotentypen.** Statt NWP-Werte nur als (interpolierte) Zusatz-Features an Stationsknoten zu behandeln, werden ICON-D2- und ECMWF-Gitterpunkte als **eigene Knotentypen** in einem heterogenen Graphen (`HeteroData`, PyTorch Geometric) modelliert, mit gerichteten Kantentypen `icond2 → station` ("informs") und `ecmwf → station` ("informs") zusätzlich zum klassischen `station ↔ station`-Nachbarschaftsgraphen. Die Aggregation von Gitterpunkt- zu Stationsinformation erfolgt nicht per festem Interpolationsgewicht, sondern **gelernt** über eine bipartite Graph-Attention-Schicht (GATv2), die pro Zeitschritt (bzw. pro Decoder-Schritt) neu berechnet wird und vom aktuellen Hidden State der Station abhängen kann ("NWP-Attention"). Damit wird das räumliche Downscaling Teil des End-to-End-Lernproblems statt eines vorgelagerten, separaten Preprocessing-Schritts.

3. **Kombination von räumlichem Kriging-artigem Transfer-Learning (IGNNK-Masking) mit Bias-Correction.** Trainingsaufgabe ist nicht reine Zeitreihenprognose an bekannten Knoten, sondern **induktives räumliches Interpolieren** (angelehnt an IGNNK — *Inductive Graph Neural Networks for Spatiotemporal Kriging*, Wu et al. 2021): Pro Trainingsbeispiel wird eine zufällige Teilmenge von Stationen als "Ziel" maskiert (Messwerte auf 0 gesetzt), während die übrigen Stationen als reale Nachbarn mit echten Messwerten dienen. Das Modell lernt so, für eine beliebige, im Training ungesehene Zielstation allein aus (a) deren NWP-Gitterpunkt-Nachbarschaft und (b) den echten Messungen benachbarter Stationen zu prognostizieren — und ist dadurch direkt auf neue Standorte übertragbar, ohne dass diese je Trainingsdaten geliefert haben müssen.

**Abgrenzung / Ablationsachsen im Experiment-Design** (siehe Konfigurationsvarianten in Abschnitt 6): Es wird explizit variiert, ob (a) NWP als eigene Graphknoten behandelt werden (`nwp_nodes: true/false`) oder nur als konkateniertes Feature am nächsten Gitterpunkt, (b) eigene Stationshistorie als Input verfügbar ist (`hist_wind`/"base" vs. "nwp_hist"-Varianten) oder nicht (reines NWP-only-Setting — der Kern-Anspruch "ohne historische Messungen"), und (c) ein TFT (Temporal Fusion Transformer) als **Nicht-Graph-Baseline** mit vergleichbaren Feature-/Split-Bedingungen mithält.

---

## 3. Datenbasis

### 3.1 Wetterstationen (Zielgrößen / Ground Truth)

- Synthetische Zeitreihen pro DWD-Station (5-stellige Stations-ID), stündliche Auflösung, UTC.
- Kernvariablen: `wind_speed` (10 m), `temperature_2m`, `relative_humidity`, `pressure`, `wind_direction`, `friction_wind`, `density`.
- **Synthetische Leistungswerte** `power_t1`…`power_t6` für 6 Turbinentypen sowie nabenhöhen-extrapolierte Windgeschwindigkeiten `wind_speed_t1`…`wind_speed_t6` — erlaubt sowohl `wind_speed`- als auch `power`-Targets aus derselben Datenbasis.
- Stationsmetadaten: `park_id`, `longitude`, `latitude`, `altitude`, `commissioning_date`.
- **Split:** explizite, nicht-zufällige Aufteilung in Train-/Val-/Test-Stationen (keine Zeit-basierte Zufallsziehung, sondern feste Stations-ID-Listen), zusätzlich ein zeitlicher Test-Zeitraum (`test_start`/`test_end`, i. d. R. 2025-08-01 bis 2025-10-31), der für **alle** Stationen gilt. D. h. Generalisierung wird gleichzeitig über **neue Standorte** (räumlich, "unseen stations") und über einen **zeitlich nachgelagerten** Zeitraum getestet — typische Größenordnung ~100 Train-, ~50 Val-, ~50 Test-Stationen (Beispielkonfiguration TFT-Baseline: 103 Train / 50 Val / 50 Test).
- Topographische Zusatz-Features pro Station (statisch, für Graph-Kanten und Knoten genutzt): `elevation`, `slope`, `aspect_sin/cos`, `tpi5`, `tpi75` (Topographic Position Index, zwei Radien), `tdi` (Terrain Dissection Index), `elev_std`, `z0` (Rauigkeitslänge), `dist_coast`.

### 3.2 ICON-D2 (DWD, hochaufgelöstes regionales NWP-Modell)

- Quelle: ursprünglich CSV je Gitterpunkt/Lauf, mittlerweile primär **PostgreSQL/PostGIS-Datenbank** (Migration dokumentiert in `docs/icond2_database_integration.md`), CSV-Fallback verfügbar.
- Modellläufe: 4× täglich (00/06/09/12/15 UTC, je nach Pipeline-Variante), Lead-Time 0–48 h.
- **6 Höhenschichten** (Modelllevel, keine Druckflächen), Schichtmitte als effektive Höhe: 10 m, 38 m, 78 m, 127 m, 184 m, 247 m — jeweils mit `u_wind`, `v_wind`, `temperature`, `pressure`, spezifischer Feuchte `qs`.
- Horizontale Auflösung: ICON-D2-Gitter (~2,2 km, Deutschland-Fokus).
- Abgeleitete Features: `wind_speed_Xm = sqrt(u² + v²)` je Höhenschicht; Luftdichte aus Temperatur/Druck/Feuchte.
- Zeitraum in den Kernstudien: 2023-07-24 bis 2026-03 (fortlaufend erweitert), ~957 Läufe je Run-Hour.

### 3.3 ECMWF HRES (globales NWP-Modell, gröber aufgelöst)

- Quelle: PostgreSQL/PostGIS (`ecmwf_wind_sl`-Tabelle, Grid-Topologie in `ecmwf_grid_points`, 759 Gitterpunkte im Untersuchungsgebiet, KNN-Query über PostGIS).
- Läufe: 00 und 12 UTC, Lead-Time bis 57 h.
- Variablen auf 3 Höhen (10 m, 100 m, 200 m): `u_wind`, `v_wind`; zusätzlich `temp_2m`, `dew_point_2m`, `specific_rho`, `friction_velocity`.
- Dient als **zweite, unabhängige NWP-Quelle** — sowohl als zusätzliche Skill-Referenz (gröberes, aber oft stabileres globales Modell) als auch als zweiter NWP-Knotentyp im heterogenen Graphen (Multi-NWP-Fusion).

### 3.4 Sampling-Logik (zentrales Prinzip aller Trainingspipelines)

Ein Trainingsbeispiel ist an **einen ICON-D2-Lauf** (`t_run`) gebunden, nicht an einen beliebigen Zeitpunkt:

```
t_run - 48h                    t_run                      t_run + 48h
   |                              |                              |
   [== historischer NWP-Lauf ==][== aktueller NWP-Lauf, Lead 0..48h ==]
   [======= Messungen (verfügbar, H=48h) =======]     (Zukunft, nicht verfügbar)
                                                [======== Zielgröße (F=48h) ========]
```

- Sequenzlänge insgesamt 96 Schritte (48 h Historie + 48 h Prognosehorizont), stündlich.
- Für Zielstationen werden Messwerte im gesamten Fenster auf 0 maskiert (induktives Kriging-Setting, s. u.); NWP-Daten sind für **alle** Knoten und den vollen 96h-Zeitraum bekannt (`known future covariates`).
- Skalierung (StandardScaler) wird **ausschließlich** auf Trainingsdaten/-stationen gefittet, um Data Leakage zu vermeiden.

---

## 4. Modellarchitekturen

Alle Modelle laufen im selben Experiment-Framework (`forecasting_framework/`, PyTorch) mit gemeinsamer Datenpipeline, gemeinsamem heterogenem Graphen (`HeterogeneousGraphBuilder`) und gemeinsamer HPO-Infrastruktur (Optuna).

### 4.1 Heterogener Graph (gemeinsame Grundlage für DCRNN und MTGNN)

- **Knotentypen:** `station` (Wetterstationen/Windparks), `icond2` (ICON-D2-Gitterpunkte), `ecmwf` (ECMWF-Gitterpunkte).
- **Kantentypen:** `("station","near","station")` — bidirektional, Konnektivität wahlweise Delaunay-Triangulation oder k-NN, mit Kantenfeatures `[Distanz normiert, sin(Bearing), cos(Bearing), (optional Höhendifferenz), (optional Topo-Features)]`; `("icond2","informs","station")` und `("ecmwf","informs","station")` — gerichtet, unidirektional, je Station die *n* nächsten Gitterpunkte (typ. 4).
- Der Graph wird **einmal statisch** aus den Koordinaten gebaut; Knotenfeatures (Zeitreihen) werden pro Sample vom Sampler befüllt.
- **IGNNK-artiges Masking:** pro Trainingsbeispiel wird eine zufällige Teilmenge der Stationen im Subgraph als Zielknoten (`target_mask`) gewählt, deren Messwerte über den gesamten 96h-Zeitraum auf 0 gesetzt werden; die verbleibenden Nachbarstationen liefern echte Historie. Damit generalisiert das Modell nachweislich auf Stationen, die im Training nie als Nachbar *oder* Ziel vorkamen (separate `val_files`-Stationsliste).

### 4.2 DCRNN (Diffusion Convolutional Recurrent Neural Network)

- Seq2Seq-Encoder-Decoder mit **DCGRU-Zellen** (Diffusionskonvolution über den Stationsgraphen, `K_hop`-Diffusionsschritte, bidirektional/BiDirDiffConv gemäß Original-Paper) statt Standard-GRU-Gates.
- **NWP-Attention:** an jedem Encoder-/Decoder-Zeitschritt aggregiert eine bipartite GATv2-Schicht die Merkmale der *k* nächsten ICON-D2- bzw. ECMWF-Knoten in eine Stationsrepräsentation; die Query basiert auf dem aktuellen Hidden State der Station (nicht auf einer Zero-Query), sodass die Aufmerksamkeit vom Systemzustand der Station abhängen kann.
- Decoder ist **autoregressiv** (Prognose des Vorschritts fließt in den nächsten Schritt ein) mit linear zerfallendem Teacher-Forcing (bzw. optional inverse-Sigmoid-Schedule gemäß Appendix E des DCRNN-Papers) und NWP-Injection an jedem Decoderschritt.
- Optionale **richtungsabhängige Kantengewichte** (`direction_to_adj`): Stations-Kantengewichte werden pro Zeitschritt anhand der (gemessenen bzw. NWP-abgeleiteten) Windrichtung neu berechnet — Kanten in Windrichtung werden hochgewichtet, entgegengesetzte abgeschwächt (physikalisch motivierte, zeitvariable Graphtopologie, im Traffic-Forecasting-Original nicht vorgesehen).
- Konfigurierbarer Ablations-Schalter `nwp_nodes`: `True` = Standardpfad mit expliziten NWP-Graphknoten + GATv2; `False` = NWP-Merkmale werden stattdessen direkt (ohne Graph-Attention) an die Stationsmerkmale konkateniert — direkter Ablationstest für die zentrale Novelty-Behauptung aus Abschnitt 2.2.
- Architektur wurde im Mai 2026 überarbeitet, um näher am Originalpaper zu sein (u. a. echte bidirektionale Diffusionskonvolution statt unidirektional, Attention pro Zeitschritt statt einmalig vorab berechnet) — dokumentiert in `docs/dcrnn_implementation_fixes.md`.

### 4.3 MTGNN (Connecting the Dots — Wu et al. 2020)

- Adaptiert für das **induktive** Setting: statt lernbarer, ID-gebundener Knoten-Embeddings (transduktiv im Original) werden Knoten-Embeddings aus statischen Stationsmerkmalen (Lage, Höhe, Typ) über ein kleines MLP erzeugt — dadurch auf neue Knoten übertragbar.
- **Graph-Learning-Modul** wie im Original: `A = ReLU(tanh(α(M1·M2ᵀ − M2·M1ᵀ)))`, gelernte, asymmetrische Adjazenz zusätzlich zum vorgegebenen Nachbarschaftsgraphen; getrennte, zeilennormierte Vorwärts-/Rückwärts-Adjazenz (analog Graph WaveNet/DCRNN-Konvention).
- **Dilated Inception** Temporal-Convolution-Blöcke (Kernelgrößen 2/3/6/7, kausal gepaddet) + **Mixhop-Propagation** (K-Hop-Graphdiffusion mit Restart-Wahrscheinlichkeit β) als räumliches Modul, gestapelt über mehrere Layer mit Skip-Connections.
- NWP-Einbindung analog zu DCRNN, aber über die homogene Variante `HomoNWPAttentionLayer` (reguläre bipartite GATv2, da MTGNN — anders als DCRNN — mit homogenen, gebatchten Tensor-Samples statt heterogenem PyG-Graph pro Sample arbeitet).
- **Curriculum Learning**: Trainingshorizont wird schrittweise von 1 auf den vollen Prognosehorizont (48 h) erhöht (`cl_steps`), wie im Originalpaper vorgeschlagen.

### 4.4 WaveNet-Variante (dritte GNN-Baseline im Framework)

- Weitere Architektur (`geostatistics/wavenet`), ebenfalls Teil der Retrain-/HPO-Pipeline (siehe Studienliste unten) — dient als zusätzlicher State-of-the-Art-Vergleich innerhalb der Graph-Modelle, ist aber nicht Teil der zentralen Novelty-Erzählung.

### 4.5 TFT (Temporal Fusion Transformer) — Nicht-Graph-Baseline

- Klassische, nicht graphbasierte Referenzarchitektur (LSTM-Encoder, Multi-Head-Attention, Variable-Selection-Networks, Quantil-Output — hier deterministisch mit `quantiles=[0.5]`, d. h. MSE-Loss analog zu den GNN-Modellen für Vergleichbarkeit).
- Explizit **mit identischem Stations-Split, identischem Testfenster und identischen NWP-/Topo-Features** wie DCRNN/MTGNN konfiguriert ("copied verbatim … so folds & held-out stations are directly comparable"), um einen fairen Vergleich "Graph vs. kein Graph" bei sonst gleicher Informationsbasis zu ermöglichen.
- Räumlicher Kontext wird hier **nicht** über einen Graphen, sondern über die *n* nächsten Nachbarstationen als zusätzliche Zeitreihen-Inputs (`next_n_stations`) hergestellt — der methodische Kontrast zur GNN-Novelty.
- Feature-Gruppen explizit nach TFT-Konvention getrennt: `known_features` (NWP, für den ganzen Horizont im Voraus bekannt), `observed_features` (nur Nachbarstations-Historie, keine eigene Stationshistorie — "base"-Variante), `static_features` (Topo + Lage).

---

## 5. Ablations- und Feature-Achsen (wichtig für die Bias-Correction-Fragestellung)

| Achse | Ausprägungen | Zweck |
|---|---|---|
| **Eigene Stationshistorie verfügbar?** | "base"/"nwp only" (keine eigene Historie, NUR NWP + Nachbarstationen) vs. "nwp_hist" (zusätzlich eigene vergangene Messwerte, dort wo vorhanden) | Testet den zentralen Anspruch: Wie gut funktioniert reine NWP-Bias-Correction ohne jede Standort-Messhistorie im Vergleich zu einem Modell, das (wo verfügbar) trotzdem Eigenhistorie nutzen darf? |
| **NWP als Graphknoten vs. Feature** | `nwp_nodes: true` (GATv2 über Gitterpunkt-Knoten) vs. `false` (Gitterpunkt-Werte direkt an Stationsmerkmale konkateniert) | Ablation der Kern-Novelty (heterogener Graph) |
| **Ein- vs. Zwei-NWP-Quellen** | nur ICON-D2 vs. ICON-D2 + ECMWF gemeinsam im Graphen | Mehrwert der Multi-NWP-Fusion |
| **Zielgröße** | `wind_speed` vs. `power` | Trennung von reiner meteorologischer Bias-Correction und betrieblicher Leistungsprognose |
| **Windrichtungs-Feature / richtungsabhängige Adjazenz** | `direction_to_adj: true/false` | Physikalisch motivierte, zeitvariable Graphtopologie vs. statischer Distanzgraph |
| **Architektur** | DCRNN, MTGNN, WaveNet (GNN) vs. TFT (kein Graph) | Kern-Modellvergleich der Publikation |
| **Extrapolation auf Nabenhöhe** | Power-Law-Extrapolation (`extrapolate`) der NWP-/Messwindgeschwindigkeit von Referenzhöhe auf Turbinen-Nabenhöhe mit adaptivem, pro Zeitschritt geschätztem Hellmann-Exponenten | Physikalisch fundierte Vorverarbeitung, relevant für `power`-Target |

**Skill-Metriken** (zentral für die Bewertung des NWP-Mehrwerts):
- `Skill` = 1 − RMSE(Modell)/RMSE(zeitliche Persistenz)
- `Skill_NWP` bzw. `skill_icond2` = 1 − RMSE(Modell)/RMSE(rohes ICON-D2 am nächsten Gitterpunkt) — quantifiziert direkt den Bias-Correction-Mehrwert des Modells gegenüber dem unkorrigierten NWP
- `skill_ecmwf` analog gegenüber ECMWF

---

## 6. Stand der Umsetzung (Juni–Juli 2026)

- Vollständige Trainings-/Evaluationspipeline für DCRNN, MTGNN, WaveNet und TFT implementiert und lauffähig (`geostatistics/train_dcrnn.py`, `train_mtgnn.py`, `train_wavenet.py`, TFT über `utils/prepare_data_for_tft` + `train_cl.py`).
- **Hyperparameter-Optimierung** über Optuna (zentral in PostgreSQL, `OPTUNA_STORAGE`), pro Architektur/Feature-Variante eine eigene Studie; u. a. 7 abgeschlossene HPO-Studien für DCRNN-Varianten (`wind_dcrnn`, `wind_dcrnn_base`, `wind_dcrnn_nwp_hist`), MTGNN-Varianten (`wind_mtgnn`, `wind_mtgnn_nwp`, `wind_mtgnn_nwp_hist`) und WaveNet.
- **Retrain-/Evaluations-Pipeline**: beste HPO-Parameter je Studie werden über 3 zeitliche Folds (rollierendes Train/Val-Fenster, `min_train_date: 2024-03-31`, Test ab `2025-08-01`) nachtrainiert und in zwei Szenarien evaluiert — `excl_val` (Zielstationen nur mit Trainingsstationen als Nachbarn) und `incl_val` (echtes Leave-One-Out über alle Stationen inkl. Validierungsstationen als reale Nachbarn).
- Ergebnisse werden pro Station mit RMSE/MAE/R² sowie den Skill-Scores gegen Persistenz und rohes NWP (ICON-D2/ECMWF) erfasst.
- Räumliche Interpolationsbaseline (Regression Kriging, `wind_interpol`) als klassischer, nicht-neuronaler Vergleichspunkt separat implementiert.

---

## 7. Für die Literaturrecherche relevante Themenfelder

Zur gezielten Recherche mit einer neuen KI-Session sollten folgende Themenblöcke abgedeckt werden:

1. **NWP-Bias-Correction / Model Output Statistics (MOS)** für Windenergie — klassische (lineare Regression, Kriging, Quantile Mapping) und ML-basierte Ansätze (Random Forest, Gradient Boosting, LSTM-basierte Post-Processing).
2. **Graph Neural Networks für Wetterprognose und Downscaling** — u. a. GraphCast, Neural-LAM und verwandte GNN-Wettermodelle (Vergleich: dort meist reine NWP-Emulation/globale Modelle, nicht standortscharfe Bias-Correction einzelner Energieanlagen).
3. **Spatiotemporal GNNs aus dem Traffic-Forecasting-Bereich**: DCRNN (Li et al. 2018), Graph WaveNet (Wu et al. 2019), MTGNN (Wu et al. 2020), STGCN (Yu et al. 2018) — Originalarbeiten und deren bisherige Übertragungen auf andere Domänen (Luftqualität, Energie, Hydrologie), um die Neuheit "erstmals für Windenergie" wasserdicht zu belegen bzw. eng verwandte Arbeiten zu finden.
4. **Inductive Spatiotemporal Kriging**: IGNNK (Wu et al. 2021) und Nachfolgearbeiten — zentrale methodische Grundlage des Masking-/Trainingsschemas.
5. **Heterogene Graphen / bipartite Graph Attention** in räumlichen Prognosemodellen (GATv2, heterogene GNN-Frameworks wie PyG `HeteroData`) — insbesondere Arbeiten, die verschiedene Datenquellen (Sensor vs. Modellgitter) als unterschiedliche Knotentypen behandeln.
6. **Windenergie-Kurzfristprognose allgemein** (0–48 h Horizont): Stand der Technik, übliche Baselines (Persistenz, physikalische NWP-Power-Curve-Modelle, TFT/Transformer-basierte Ansätze), typische Fehlermetriken und Benchmarks in der Windenergie-Literatur.
7. **Temporal Fusion Transformer** (Lim et al. 2021) und dessen bisherige Anwendung auf Energie-/Wetterprognose — als Referenzpunkt für die Nicht-Graph-Baseline.
8. **Power-Law-Extrapolation der Windgeschwindigkeit auf Nabenhöhe** (Hellmann-Exponent) — Stand der Technik und Alternativen (Log-Law, stabilitätsabhängige Exponenten).
9. **Multi-NWP-Fusion / Ensemble-Postprocessing** — Arbeiten, die mehrere NWP-Quellen unterschiedlicher Auflösung (regional vs. global) kombinieren.
10. **Publikationsziel:** Journals mit Fokus auf (a) angewandte Energieprognose (z. B. *Wind Energy*, *Applied Energy*, *Renewable Energy*, *Energy and AI*) oder (b) ML-für-Geowissenschaften/Wetter (z. B. *npj Climate and Atmospheric Science*, *Environmental Data Science*) — Literaturrecherche sollte beide Communities abdecken, da die Novelty an der Schnittstelle liegt.

---

## 8. Offene Punkte / mögliche Reviewer-Fragen (zur Vorbereitung)

- Wie wird der Unterschied zwischen "räumlichem Downscaling" (Kriging-artig) und "temporaler Bias-Correction" methodisch sauber getrennt und einzeln evaluiert (Ablationen in Abschnitt 5)?
- Ist die behauptete Neuheit "DCRNN/MTGNN erstmals für Windenergie" durch eine systematische Literatursuche abgesichert, oder gibt es nahe verwandte, aber nicht identische Arbeiten (z. B. DCRNN für Solarenergie, MTGNN für Luftqualität mit Wetter-Kovariaten)?
- Wie robust ist die synthetische Leistungsgenerierung (`power_t1`…`power_t6` aus Leistungskurven) im Vergleich zu realen SCADA-Daten — Limitation, die im Paper adressiert werden sollte.
- Vergleichbarkeit von ICON-D2 (2,2 km, regional, DWD) und ECMWF HRES (global, gröber) als Prädiktoren — Diskussion nötig, warum beide sinnvoll kombiniert werden (komplementäre Fehlercharakteristik).
- Generalisierung über Terrain-Typen (Küste, Mittelgebirge, Flachland) — die Topo-Features (`slope`, `tpi`, `z0`, `dist_coast`) legen nahe, dass Orographie eine Rolle spielt; ggf. stratifizierte Auswertung nötig.
