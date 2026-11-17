# GFS-Test (2026-10-27)

Frage: Bringt GFS als drittes NWP-Modell neben ICON-D2 und ECMWF etwas? TFT (`cl_m-tft`), je mit und ohne GFS-Features,
einmal für einen zufällig gezogenen Park (`one`) und einmal zentral über 90 Parks (`all`), Seeds 42–44.
Train 2023-07-24 .. 2024-07-31, Test 2024-08-01 .. 2025-07-31. Configs: `gen_configs.py`.

- Wind: `parks_v1` (90 Parks), GFS-Features `gfs_wind_speed_h{50,80,100}`
- PV: `parks_pv_v2_30min` (90 Parks), GFS-Features `gfs_ghi`, `gfs_tcc`

Ergebnis (Testmittel über 3 Seeds, normierte Leistung; Pickles in `results/parks_pv_v2_30min` bzw. `results/parks_v1`):

| Fall | RMSE ohne GFS | RMSE mit GFS | Δ RMSE | MAE ohne | MAE mit |
|---|---|---|---|---|---|
| PV all | 0.0479 | 0.0474 | −1,0 % | 0.0219 | 0.0215 |
| PV one | 0.0338 | 0.0333 | −1,5 % | 0.0155 | 0.0161 |
| Wind all | 0.0887 | 0.0853 | −3,8 % | 0.0560 | 0.0532 |
| Wind one | 0.0511 | 0.0488 | −4,5 % | 0.0327 | 0.0309 |

Entscheidung (Viktor, 2026-11-05): GFS-Test behalten. Die PV-Eingangsdaten `parks_pv_v2_30min` sind am 2026-11-05
gelöscht (durch v3/v4 ersetzt); für eine Wiederholung des PV-Teils `pv()` in `gen_configs.py` auf das aktuelle Release
umstellen. Ergebnisse, Logs (`logs/gfs_test`) und Configs bleiben.
