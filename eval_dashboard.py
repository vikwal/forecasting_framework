"""
Streamlit Dashboard für die Evaluation von FL und CL Ergebnissen
"""
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from collections import defaultdict

# Seitenkonfiguration
st.set_page_config(
    page_title="Forecasting Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Hilfsfunktionen
def concatenate_results(results_dir: str,
                        results: list,
                        index_cols: list,
                        get_skill: bool = False,
                        pers: pd.DataFrame = pd.DataFrame(),
                        sort_skill: bool = False) -> pd.DataFrame:
    """Konkatiniert Ergebnisse aus mehreren pkl-Files"""
    indices = defaultdict(list)
    metrics = []
    if not results:
        st.warning('Keine Daten gefunden.')
        return None

    for file in results:
        with open(os.path.join(results_dir, file), 'rb') as f:
            pkl = pickle.load(f)
        df = pkl['evaluation']
        df.reset_index(inplace=True)

        if get_skill:
            df = pd.merge(df, pers[['RMSE', 'key', 'output_dim', 'freq']],
                         on=['key', 'output_dim', 'freq'],
                         how='left',
                         suffixes=('', '_p'))
            df['Skill'] = 1 - df.RMSE / df.RMSE_p
            df.drop('RMSE_p', axis=1, inplace=True)

        for col in index_cols:
            indices[col].append(df[col].iloc[0])
        df.drop(index_cols, axis=1, inplace=True)
        metric = df.mean(axis=0)
        metrics.append(metric)

    df = pd.DataFrame(metrics, columns=metric.index)
    df_index = pd.DataFrame(indices)
    df = pd.concat([df, df_index], axis=1)
    df.sort_values(['output_dim', 'freq', 'Models'], inplace=True)

    if sort_skill:
        df.sort_values(['Skill'], ascending=False, inplace=True)

    return df

def read_sim(results_dir, sim):
    """Liest ein einzelnes Simulations-File"""
    try:
        with open(os.path.join(results_dir, sim), 'rb') as f:
            sim_results = pickle.load(f)
    except:
        sim_results = None
    return sim_results

def plot_interactive_history(history: dict, key_prefix: str = "cl"):
    """
    Erstellt einen interaktiven Plotly-Plot für die Training History.

    Args:
        history: Dictionary mit den History-Daten (z.B. {'loss': [...], 'val_loss': [...]})
        key_prefix: Eindeutiger Prefix für Streamlit-Keys, um Konflikte zu vermeiden.
    """
    if not history:
        st.info("Keine History-Daten verfügbar")
        return

    # 1. Metriken gruppieren (Base Name -> {'train': key, 'val': key})
    metric_groups = {}
    for key in history.keys():
        if key.startswith('val_'):
            base = key[4:]
            type_ = 'val'
        else:
            base = key
            type_ = 'train'

        if base not in metric_groups:
            metric_groups[base] = {}
        metric_groups[base][type_] = key

    available_base_metrics = list(metric_groups.keys())

    # 2. Standardauswahl definieren
    default_metrics = []
    # Priorität: mse > loss > rmse > mae
    priority_metrics = ['mse', 'loss', 'rmse', 'mae']
    for m in priority_metrics:
        if m in available_base_metrics:
            default_metrics.append(m)
            break # Nur eine Standard-Metrik für den Anfang, um Überladung zu vermeiden

    # Filtere Defaults, die tatsächlich existieren
    default_metrics = [m for m in default_metrics if m in available_base_metrics]

    # 3. Multiselect für Basis-Metriken
    selected_base_metrics = st.multiselect(
        "Metriken auswählen:",
        options=available_base_metrics,
        default=default_metrics,
        key=f"{key_prefix}_metrics_select"
    )

    if not selected_base_metrics:
        st.warning("Bitte wählen Sie mindestens eine Metrik aus.")
        return

    # 4. Plot erstellen
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Farb-Mapping
    # R^2 -> Orange
    # Loss/Error -> Blau
    # Andere -> Zyklisch

    # Farben definieren (Plotly Standardfarben)
    COLOR_BLUE = 'rgb(31, 119, 180)'
    COLOR_ORANGE = 'rgb(255, 127, 14)'
    COLOR_GREEN = 'rgb(44, 160, 44)'
    COLOR_RED = 'rgb(214, 39, 40)'

    for base_metric in selected_base_metrics:
        group = metric_groups[base_metric]

        # Bestimmen, ob rechte Achse verwendet werden soll (für R^2)
        is_r2 = 'r2' in base_metric.lower() or 'r^2' in base_metric.lower()
        use_secondary_y = is_r2

        # Farbe bestimmen
        if is_r2:
            base_color = COLOR_ORANGE
        elif base_metric in ['loss', 'mse', 'rmse', 'mae']:
            base_color = COLOR_BLUE
        else:
            base_color = COLOR_GREEN # Fallback

        # Train Plot
        if 'train' in group:
            key = group['train']
            values = history[key]
            epochs = list(range(1, len(values) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    name=f"{base_metric} (Train)",
                    mode='lines',
                    line=dict(color=base_color)
                ),
                secondary_y=use_secondary_y,
            )

        # Val Plot
        if 'val' in group:
            key = group['val']
            values = history[key]
            epochs = list(range(1, len(values) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=values,
                    name=f"{base_metric} (Val)",
                    mode='lines',
                    line=dict(color=base_color, dash='dot') # Gestrichelt für Validation
                ),
                secondary_y=use_secondary_y,
            )

    # Layout anpassen
    fig.update_layout(
        title_text="Training History",
        xaxis_title="Epochs",
        hovermode="x unified",
        font=dict(size=14), # Globale Schriftgröße
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14) # Legende Schriftgröße
        )
    )

    # Achsenbeschriftungen
    fig.update_xaxes(
        title_font=dict(size=16),
        tickfont=dict(size=14)
    )
    fig.update_yaxes(
        title_text="Loss / Error",
        secondary_y=False,
        title_font=dict(size=16),
        tickfont=dict(size=14)
    )
    fig.update_yaxes(
        title_text="R^2 Score",
        secondary_y=True,
        title_font=dict(size=16),
        tickfont=dict(size=14)
    )

    # Fix für use_container_width Deprecation Warning
    # User Request: For `use_container_width=True`, use `width='stretch'`.
    try:
        st.plotly_chart(fig, width="stretch")
    except:
         # Fallback für ältere Streamlit Versionen
         st.plotly_chart(fig, use_container_width=True)

# Haupttitel
st.title("📊 Forecasting Evaluation Dashboard")

# Datensatz-Auswahl
results_base = 'results'
if not os.path.exists(results_base):
    st.error(f"Results Verzeichnis '{results_base}' nicht gefunden!")
    st.stop()

available_datasets = [d for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d))]

if not available_datasets:
    st.error(f"Keine Unterordner in '{results_base}' gefunden!")
    st.stop()

# Hauptauswahl für Datensatz (prominent auf der Hauptseite)
st.markdown("### 📁 Datensatz auswählen")
st.markdown("Wählen Sie einen Datensatz aus den verfügbaren Results-Ordnern:")

data = st.selectbox(
    "Verfügbare Datensätze:",
    available_datasets,
    index=None,
    placeholder="Bitte wählen Sie einen Datensatz aus...",
    label_visibility="collapsed"
)

# Nur weitermachen wenn ein Datensatz ausgewählt wurde
if data is None:
    st.info("👆 Bitte wählen Sie einen Datensatz aus, um fortzufahren.")
    st.stop()

# Datensatz wurde ausgewählt - jetzt laden
results_dir = os.path.join(results_base, data)

if not os.path.exists(results_dir):
    st.error(f"Results Verzeichnis '{results_dir}' nicht gefunden!")
    st.stop()

# Dateien laden
result_files = os.listdir(results_dir)
sims = [f for f in result_files if (f.endswith('.pkl')) & ('cl' not in f) & ('fl' not in f)]
cl_sims = [f for f in result_files if 'cl' in f]
cl_sims.sort()
fl_sims = [f for f in result_files if 'fl' in f]
persistence_files = [f for f in result_files if 'persistence' in f]

persistence_file = None
pers = pd.DataFrame()
if persistence_files:
    persistence_file = persistence_files[0]
    pers = pd.read_csv(os.path.join(results_dir, persistence_file))

# Info über gefundene Files in der Sidebar
st.sidebar.header("📊 Datensatz Info")
st.sidebar.metric("Ausgewählter Datensatz", data)
st.sidebar.metric("FL Simulations", len(fl_sims))
st.sidebar.metric("CL Simulations", len(cl_sims))
st.sidebar.metric("Total Files", len(result_files))

# Trennlinie
st.markdown("---")

# Tabs für verschiedene Ansichten
tab1, tab2, tab3 = st.tabs(["📈 Aggregierte Ansicht", "🔍 Detailansicht", "⚖️ Vergleichsansicht"])

# Tab 1: Aggregierte Ansicht
with tab1:
    st.header("Aggregierte Ansicht über alle Experimente")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Federated Learning (FL)")
        if fl_sims:
            index_cols_fl = ['Models', 'output_dim', 'freq', 't_0', 'strategy', 'personalization']
            df_fl = concatenate_results(
                results_dir=results_dir,
                results=fl_sims,
                index_cols=index_cols_fl
            )

            if df_fl is not None and not df_fl.empty:
                display_cols = ['R^2', 'RMSE', 'MAE']
                if 'Skill' in df_fl.columns:
                    display_cols.append('Skill')
                display_cols.extend(['Models', 'strategy', 'personalization'])

                df_display = df_fl[display_cols].round(4)
                st.dataframe(df_display, height=400)

                # Download Button
                csv = df_fl.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 FL Ergebnisse als CSV herunterladen",
                    data=csv,
                    file_name=f'fl_results_{data}.csv',
                    mime='text/csv',
                )
            else:
                st.info("Keine FL Daten verfügbar")
        else:
            st.info("Keine FL Simulationen gefunden")

    with col2:
        st.subheader("Centralized Learning (CL)")
        if cl_sims:
            index_cols_cl = ['Models', 'output_dim', 'freq', 't_0', 'key']
            df_cl = concatenate_results(
                results_dir=results_dir,
                get_skill=False,
                results=cl_sims,
                index_cols=index_cols_cl
            )

            if df_cl is not None and not df_cl.empty:
                display_cols = ['R^2', 'RMSE', 'MAE']
                if 'Skill' in df_cl.columns:
                    display_cols.append('Skill')
                display_cols.extend(['Models', 'output_dim'])

                df_display = df_cl[display_cols].round(4)
                st.dataframe(df_display, height=400)

                # Download Button
                csv = df_cl.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 CL Ergebnisse als CSV herunterladen",
                    data=csv,
                    file_name=f'cl_results_{data}.csv',
                    mime='text/csv',
                )
            else:
                st.info("Keine CL Daten verfügbar")
        else:
            st.info("Keine CL Simulationen gefunden")

# Tab 2: Detailansicht
with tab2:
    st.header("Detailansicht für einzelne Simulationen")

    # Auswahl zwischen FL und CL
    mode = st.radio("Modus auswählen", ["Centralized Learning (CL)", "Federated Learning (FL)"], horizontal=True)

    if mode == "Centralized Learning (CL)":
        if cl_sims:
            # Simulation auswählen
            sim_options = {f"{i}: {sim}": sim for i, sim in enumerate(cl_sims)}
            selected_sim_key = st.selectbox("CL Simulation auswählen", list(sim_options.keys()))
            selected_sim = sim_options[selected_sim_key]

            # Simulation laden
            cl_results = read_sim(results_dir, selected_sim)

            if cl_results:
                st.success(f"✅ Simulation geladen: {selected_sim}")

                # Detect scenario: Check if 'individual_evaluations' exists
                is_multi_training = 'individual_evaluations' in cl_results

                if is_multi_training:
                    st.info(f"📊 Multi-Training Szenario erkannt: {len(cl_results['individual_evaluations'])} Trainingszyklen")
                else:
                    st.info("📊 Single-Training Szenario")

                # Konfigurations-Info
                with st.expander("🔧 Konfiguration anzeigen"):
                    params = cl_results['config']['params']
                    data_params = cl_results['config']['data']
                    hyperparams = cl_results.get('hyperparameters', {})

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**Features:**")
                        st.write(f"- Known features: `{params.get('known_features', 'N/A')}`")
                        st.write(f"- Observed features: `{params.get('observed_features', 'N/A')}`")
                        st.write(f"- Static features: `{params.get('static_features', 'N/A')}`")

                    with col2:
                        st.write("**Hyperparameter:**")
                        if hyperparams:
                            for key, value in hyperparams.items():
                                st.write(f"- {key}: `{value}`")
                        else:
                            st.write("- N/A")

                    with col3:
                        test_start = pd.to_datetime(data_params.get("test_start", "N/A"))
                        test_end = pd.to_datetime(data_params.get("test_end", "N/A"))
                        retrain_interval = data_params.get("retrain_interval", "N/A")
                        st.write('**More Info:**')
                        st.write(f'- Next n grid points: {params.get("next_n_grid_points", "N/A")}')
                        st.write(f'- Turbines per park: {params.get("turbines_per_park", "N/A")}')
                        st.write(f'- Test start: {test_start.strftime("%Y-%m-%d")}')
                        st.write(f'- Test end: {test_end.strftime("%Y-%m-%d")}')
                        st.write(f'- Retrain interval: {retrain_interval}')

                # Metriken anzeigen - Always show the averaged evaluation at the top
                st.subheader("📊 Evaluations-Metriken (Durchschnitt über alle Zyklen)" if is_multi_training else "📊 Evaluations-Metriken")
                eval_df = cl_results['evaluation'][['R^2', 'RMSE', 'MAE']].copy()
                if 'Skill' in cl_results['evaluation'].columns:
                    eval_df['Skill'] = cl_results['evaluation']['Skill']
                if 'key' in cl_results['evaluation'].columns:
                    eval_df['key'] = cl_results['evaluation']['key']

                st.dataframe(eval_df.round(4))

                # Bar chart for multi-training scenario showing all cycles
                if is_multi_training:
                    st.subheader("📊 Metriken-Vergleich über alle Zyklen")

                    # Selectbox for metric selection
                    available_metrics = ['R^2', 'RMSE', 'MAE']
                    if 'Skill' in cl_results['evaluation'].columns:
                        available_metrics.append('Skill')

                    selected_metric = st.selectbox(
                        "Metrik auswählen:",
                        available_metrics,
                        key=f"metric_select_{selected_sim_key}"
                    )

                    # Collect metric values for all cycles
                    if 'individual_evaluations' in cl_results:
                        cycle_labels = [f"Z{i+1}" for i in range(len(cl_results['individual_evaluations']))]
                        metric_values = []

                        for eval_df_cycle in cl_results['individual_evaluations']:
                            # Calculate mean of the metric across all parks for this cycle
                            if 'mean' in eval_df_cycle.index:
                                metric_values.append(eval_df_cycle.loc['mean', selected_metric])
                            else:
                                metric_values.append(eval_df_cycle[selected_metric].mean())

                        # Create bar chart with plotly
                        fig = go.Figure(data=[
                            go.Bar(
                                x=cycle_labels,
                                y=metric_values,
                                marker_color='rgb(31, 119, 180)',
                                text=[f'{v:.4f}' for v in metric_values],
                                textposition='auto',
                            )
                        ])

                        # Set y-axis range based on metric
                        y_range = [0, 1] if selected_metric == 'R^2' else [0, 0.5]

                        fig.update_layout(
                            xaxis_title="Zyklus",
                            yaxis_title=selected_metric,
                            hovermode="x",
                            font=dict(size=14),
                            showlegend=False
                        )

                        fig.update_xaxes(
                            title_font=dict(size=16),
                            tickfont=dict(size=14)
                        )
                        fig.update_yaxes(
                            title_font=dict(size=16),
                            tickfont=dict(size=14),
                            range=y_range
                        )

                        try:
                            st.plotly_chart(fig, width="stretch")
                        except:
                            st.plotly_chart(fig, use_container_width=True)

                # For multi-training scenario, add selectbox for test_dates
                selected_period_idx = 0  # Default to first period
                if is_multi_training:
                    st.markdown("---")
                    st.subheader("🔍 Einzelne Trainingszyklen")

                    # Create options for selectbox
                    test_dates = cl_results['test_dates']
                    period_options = {}
                    for idx, (start, end) in enumerate(test_dates):
                        # Format the dates nicely
                        if hasattr(start, 'strftime'):
                            start_str = start.strftime('%Y-%m-%d')
                            end_str = end.strftime('%Y-%m-%d')
                        else:
                            start_str = str(start)
                            end_str = str(end)
                        period_options[f"Zyklus {idx + 1}: {start_str} bis {end_str}"] = idx

                    selected_period_key = st.selectbox(
                        "Wählen Sie einen Trainingszyklus:",
                        list(period_options.keys()),
                        key=f"period_select_{selected_sim_key}"
                    )
                    selected_period_idx = period_options[selected_period_key]

                    # Display individual evaluation for selected period
                    st.subheader(f"📊 Evaluations-Metriken für {selected_period_key}")
                    if 'individual_evaluations' in cl_results and len(cl_results['individual_evaluations']) > selected_period_idx:
                        individual_eval = cl_results['individual_evaluations'][selected_period_idx]
                        individual_eval_df = individual_eval[['R^2', 'RMSE', 'MAE']].copy()
                        if 'Skill' in individual_eval.columns:
                            individual_eval_df['Skill'] = individual_eval['Skill']
                        if 'key' in individual_eval.columns:
                            individual_eval_df['key'] = individual_eval['key']
                        st.dataframe(individual_eval_df.round(4))
                    else:
                        st.warning("Evaluations-Daten für den gewählten Zyklus nicht verfügbar")

                # Training History Plot
                fontsize=11
                st.subheader("📈 Training History")
                if 'history' in cl_results and cl_results['history']:
                    # Handle both single dict and list of dicts
                    if is_multi_training:
                        # Multiple trainings: history is a list of dicts
                        if isinstance(cl_results['history'], list) and len(cl_results['history']) > selected_period_idx:
                            selected_history = cl_results['history'][selected_period_idx]
                            plot_interactive_history(selected_history, key_prefix=f"cl_{selected_sim_key}_period_{selected_period_idx}")
                        else:
                            st.warning("History-Daten für den gewählten Zyklus nicht verfügbar")
                    else:
                        # Single training: history is a single dict
                        plot_interactive_history(cl_results['history'], key_prefix=f"cl_{selected_sim_key}")
                else:
                    st.info("Keine History-Daten verfügbar")
            else:
                st.error("Fehler beim Laden der Simulation")
        else:
            st.info("Keine CL Simulationen verfügbar")

    else:  # Federated Learning
        if fl_sims:
            # Simulation auswählen
            sim_options = {f"{i}: {sim}": sim for i, sim in enumerate(fl_sims)}
            selected_sim_key = st.selectbox("FL Simulation auswählen", list(sim_options.keys()))
            selected_sim = sim_options[selected_sim_key]

            # Simulation laden
            fl_results = read_sim(results_dir, selected_sim)

            if fl_results:
                st.success(f"✅ Simulation geladen: {selected_sim}")

                # Konfigurations-Info
                with st.expander("🔧 Konfiguration anzeigen"):
                    params = fl_results['config']['params']
                    model_params = fl_results['config']['model']
                    hyperparams = fl_results['config'].get('hyperparameters', {})

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Features:**")
                        st.write(f"- Known features: `{params.get('known_features', 'N/A')}`")
                        st.write(f"- Observed features: `{params.get('observed_features', 'N/A')}`")
                        st.write(f"- Static features: `{params.get('static_features', 'N/A')}`")

                    with col2:
                        st.write("**Hyperparameter:**")
                        if hyperparams:
                            for key, value in hyperparams.items():
                                st.write(f"- {key}: `{value}`")
                        else:
                            st.write("- N/A")

                # Metriken anzeigen
                st.subheader("📊 Evaluations-Metriken")
                eval_df = fl_results['evaluation'][['R^2', 'RMSE', 'MAE']].copy()
                if 'Skill' in fl_results['evaluation'].columns:
                    eval_df['Skill'] = fl_results['evaluation']['Skill']
                if 'key' in fl_results['evaluation'].columns:
                    eval_df['key'] = fl_results['evaluation']['key']

                st.dataframe(eval_df.round(4))

                fontsize=12
                # Training History Plot
                st.subheader("📈 Training History")
                if 'history' in fl_results and fl_results['history']:
                    if 'metrics_aggregated' in fl_results['history']:
                        metrics_agg = fl_results['history']['metrics_aggregated']
                        plot_interactive_history(metrics_agg, key_prefix=f"fl_{selected_sim_key}")
                    else:
                        st.info("Keine aggregierten Metriken verfügbar")

                else:
                    st.info("Keine History-Daten verfügbar")
            else:
                st.error("Fehler beim Laden der Simulation")
        else:
            st.info("Keine FL Simulationen verfügbar")

# Tab 3: Vergleichsansicht
with tab3:
    st.header("Vergleichsansicht für verschiedene Simulationen")

    if not cl_sims:
        st.info("Keine CL Simulationen für Vergleich verfügbar")
    else:
        # Two selectboxes side by side
        col1, col2 = st.columns(2)

        with col1:
            # Selectbox 1: Primary model (global or local)
            sim_options_1 = {f"{i}: {sim}": sim for i, sim in enumerate(cl_sims)}
            selected_sim_key_1 = st.selectbox(
                "Modell 1 auswählen:",
                list(sim_options_1.keys()),
                key="comparison_sim1"
            )
            selected_sim_1 = sim_options_1[selected_sim_key_1]

        with col2:
            # Selectbox 2: Default "Local Models" or other global models
            comparison_options = ["Local Models"] + list(sim_options_1.keys())
            selected_comparison = st.selectbox(
                "Modell 2 auswählen:",
                comparison_options,
                key="comparison_sim2"
            )

        # Load the first model
        results_1 = read_sim(results_dir, selected_sim_1)

        if not results_1:
            st.error("Fehler beim Laden von Modell 1")
            st.stop()

        # Get evaluation dataframe from model 1
        eval_df_1 = results_1['evaluation'].copy()

        # Check if model 1 has 'key' column
        if 'key' not in eval_df_1.columns:
            st.error("Modell 1 hat keine 'key' Spalte für den Vergleich")
            st.stop()

        # Extract model name from filename (for Excel export)
        model_1_name = selected_sim_1.replace('.pkl', '').replace('cl_m-tft_out-48_freq-1h_', '').split('_2025')[0]

        # Determine comparison type
        if selected_comparison == "Local Models":
            st.subheader(f"Vergleich: {model_1_name} (Global) vs. Lokale Modelle")

            # Extract station IDs from eval_df_1
            # The key format is synth_xxxxx.csv where xxxxx is the station_id
            def extract_station_id(key):
                """Extract station_id from synth_xxxxx.csv format"""
                if isinstance(key, str) and key.startswith('synth_') and key.endswith('.csv'):
                    return key.replace('synth_', '').replace('.csv', '')
                return None

            # Get all station IDs from model 1 (excluding mean/std rows)
            eval_df_1_filtered = eval_df_1[~eval_df_1.index.isin(['mean', 'std'])].copy()
            eval_df_1_filtered['station_id'] = eval_df_1_filtered['key'].apply(extract_station_id)
            eval_df_1_filtered = eval_df_1_filtered.dropna(subset=['station_id'])

            if eval_df_1_filtered.empty:
                st.warning("Keine Station-IDs in Modell 1 gefunden")
                st.stop()

            # Find all local model files
            # Local models have format: wind_xxxxx_timestamp.pkl
            local_model_files = {}
            for file in result_files:
                if file.endswith('.pkl') and file.startswith('cl_m-tft'):
                    # Extract station_id from filename
                    # Format: cl_m-tft_out-48_freq-1h_wind_xxxxx_timestamp.pkl
                    parts = file.split('_')
                    for i, part in enumerate(parts):
                        if part == 'wind' and i + 1 < len(parts):
                            potential_station_id = parts[i + 1]
                            # Check if it's a valid station_id (5 digits)
                            if potential_station_id.isdigit() and len(potential_station_id) == 5:
                                local_model_files[potential_station_id] = file
                                break

            if not local_model_files:
                st.warning("Keine lokalen Modell-Dateien gefunden")
                st.info(f"Verfügbare Dateien: {result_files}")
                st.stop()

            st.info(f"Gefundene lokale Modelle: {len(local_model_files)}")

            # Create comparison dataframe
            comparison_data = []

            for idx, row in eval_df_1_filtered.iterrows():
                station_id = row['station_id']

                # Check if local model exists for this station
                if station_id not in local_model_files:
                    continue  # Skip if no local model found

                # Load local model
                local_results = read_sim(results_dir, local_model_files[station_id])
                if not local_results or 'evaluation' not in local_results:
                    continue

                # Get metrics from local model
                local_eval = local_results['evaluation']

                # Get mean values if available, otherwise calculate mean
                if 'mean' in local_eval.index:
                    local_r2 = local_eval.loc['mean', 'R^2']
                    local_rmse = local_eval.loc['mean', 'RMSE']
                    local_mae = local_eval.loc['mean', 'MAE']
                else:
                    local_r2 = local_eval['R^2'].mean()
                    local_rmse = local_eval['RMSE'].mean()
                    local_mae = local_eval['MAE'].mean()

                # Get metrics from global model
                global_r2 = row['R^2']
                global_rmse = row['RMSE']
                global_mae = row['MAE']

                # Calculate deltas
                # Δ abs. = Global - Local
                delta_r2_abs = global_r2 - local_r2
                delta_rmse_abs = global_rmse - local_rmse
                delta_mae_abs = global_mae - local_mae

                # Δ rel. = (Global - Local) / Local × 100%
                # For R²: only calculate if local_r2 is positive (negative R² makes relative comparison meaningless)
                delta_r2_rel = (delta_r2_abs / local_r2 * 100) if local_r2 > 0 else float('nan')
                delta_rmse_rel = (delta_rmse_abs / local_rmse * 100) if local_rmse != 0 else 0
                delta_mae_rel = (delta_mae_abs / local_mae * 100) if local_mae != 0 else 0

                comparison_data.append({
                    'station_id': station_id,
                    'R²_G': global_r2,
                    'R²_L': local_r2,
                    'Δ_R²_abs': delta_r2_abs,
                    'Δ_R²_rel': delta_r2_rel,
                    'RMSE_G': global_rmse,
                    'RMSE_L': local_rmse,
                    'Δ_RMSE_abs': delta_rmse_abs,
                    'Δ_RMSE_rel': delta_rmse_rel,
                    'MAE_G': global_mae,
                    'MAE_L': local_mae,
                    'Δ_MAE_abs': delta_mae_abs,
                    'Δ_MAE_rel': delta_mae_rel,
                })

            if not comparison_data:
                st.warning("Keine übereinstimmenden Stationen zwischen globalem und lokalen Modellen gefunden")
                st.stop()

            # Create DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('station_id')
            comparison_df = comparison_df.set_index('station_id')

            # Calculate mean and std for base metrics only
            metric_cols = ['R²_G', 'R²_L', 'RMSE_G', 'RMSE_L', 'MAE_G', 'MAE_L']
            mean_metrics = comparison_df[metric_cols].mean()
            std_metrics = comparison_df[metric_cols].std()

            # Compute deltas from the mean metrics
            mean_r2_g = mean_metrics['R²_G']
            mean_r2_l = mean_metrics['R²_L']
            mean_rmse_g = mean_metrics['RMSE_G']
            mean_rmse_l = mean_metrics['RMSE_L']
            mean_mae_g = mean_metrics['MAE_G']
            mean_mae_l = mean_metrics['MAE_L']

            # Calculate deltas from means
            mean_delta_r2_abs = mean_r2_g - mean_r2_l
            mean_delta_rmse_abs = mean_rmse_g - mean_rmse_l
            mean_delta_mae_abs = mean_mae_g - mean_mae_l

            mean_delta_r2_rel = (mean_delta_r2_abs / mean_r2_l * 100) if mean_r2_l > 0 else float('nan')
            mean_delta_rmse_rel = (mean_delta_rmse_abs / mean_rmse_l * 100) if mean_rmse_l != 0 else 0
            mean_delta_mae_rel = (mean_delta_mae_abs / mean_mae_l * 100) if mean_mae_l != 0 else 0

            # Create mean row
            mean_row = pd.Series({
                'R²_G': mean_r2_g,
                'R²_L': mean_r2_l,
                'Δ_R²_abs': mean_delta_r2_abs,
                'Δ_R²_rel': mean_delta_r2_rel,
                'RMSE_G': mean_rmse_g,
                'RMSE_L': mean_rmse_l,
                'Δ_RMSE_abs': mean_delta_rmse_abs,
                'Δ_RMSE_rel': mean_delta_rmse_rel,
                'MAE_G': mean_mae_g,
                'MAE_L': mean_mae_l,
                'Δ_MAE_abs': mean_delta_mae_abs,
                'Δ_MAE_rel': mean_delta_mae_rel,
            })

            # For std, just compute std of metrics (deltas std would need more complex calculation)
            std_delta_cols = comparison_df[['Δ_R²_abs', 'Δ_RMSE_abs', 'Δ_MAE_abs']].std()
            std_delta_rel_cols = comparison_df[['Δ_R²_rel', 'Δ_RMSE_rel', 'Δ_MAE_rel']].std()

            std_row = pd.Series({
                'R²_G': std_metrics['R²_G'],
                'R²_L': std_metrics['R²_L'],
                'Δ_R²_abs': std_delta_cols['Δ_R²_abs'],
                'Δ_R²_rel': std_delta_rel_cols['Δ_R²_rel'],
                'RMSE_G': std_metrics['RMSE_G'],
                'RMSE_L': std_metrics['RMSE_L'],
                'Δ_RMSE_abs': std_delta_cols['Δ_RMSE_abs'],
                'Δ_RMSE_rel': std_delta_rel_cols['Δ_RMSE_rel'],
                'MAE_G': std_metrics['MAE_G'],
                'MAE_L': std_metrics['MAE_L'],
                'Δ_MAE_abs': std_delta_cols['Δ_MAE_abs'],
                'Δ_MAE_rel': std_delta_rel_cols['Δ_MAE_rel'],
            })

            # Add mean and std rows
            comparison_df.loc['mean'] = mean_row
            comparison_df.loc['std'] = std_row

            model_2_name = "local"

        else:
            # Comparing two global models
            selected_sim_2 = sim_options_1[selected_comparison]
            model_2_name = selected_sim_2.replace('.pkl', '').replace('cl_m-tft_out-48_freq-1h_', '').split('_2025')[0]

            st.subheader(f"Vergleich: {model_1_name} vs. {model_2_name}")

            # Load the second model
            results_2 = read_sim(results_dir, selected_sim_2)

            if not results_2:
                st.error("Fehler beim Laden von Modell 2")
                st.stop()

            # Get evaluation dataframe from model 2
            eval_df_2 = results_2['evaluation'].copy()

            if 'key' not in eval_df_2.columns:
                st.error("Modell 2 hat keine 'key' Spalte für den Vergleich")
                st.stop()

            # Merge on 'key' - left join with model 2 as left
            # Filter out mean/std rows before merging
            eval_df_1_filtered = eval_df_1[~eval_df_1.index.isin(['mean', 'std'])].copy()
            eval_df_2_filtered = eval_df_2[~eval_df_2.index.isin(['mean', 'std'])].copy()

            # Reset index to merge on key
            eval_df_1_filtered = eval_df_1_filtered.reset_index(drop=True)
            eval_df_2_filtered = eval_df_2_filtered.reset_index(drop=True)

            # Merge
            merged_df = eval_df_2_filtered[['key', 'R^2', 'RMSE', 'MAE']].merge(
                eval_df_1_filtered[['key', 'R^2', 'RMSE', 'MAE']],
                on='key',
                how='left',
                suffixes=('_2', '_1')
            )

            if merged_df.empty:
                st.warning("Keine übereinstimmenden Keys zwischen den Modellen gefunden")
                st.stop()

            # Extract station_id for sorting
            def extract_station_id(key):
                if isinstance(key, str) and key.startswith('synth_') and key.endswith('.csv'):
                    return key.replace('synth_', '').replace('.csv', '')
                return key

            merged_df['station_id'] = merged_df['key'].apply(extract_station_id)

            # Calculate deltas
            # Model 1 is "Global" (first selectbox), Model 2 is "Local" (second selectbox)
            comparison_data = []

            for idx, row in merged_df.iterrows():
                station_id = row['station_id']

                # Get metrics
                global_r2 = row['R^2_1']
                local_r2 = row['R^2_2']
                global_rmse = row['RMSE_1']
                local_rmse = row['RMSE_2']
                global_mae = row['MAE_1']
                local_mae = row['MAE_2']

                # Skip if any value is NaN
                if pd.isna(global_r2) or pd.isna(local_r2):
                    continue

                # Calculate deltas
                delta_r2_abs = global_r2 - local_r2
                delta_rmse_abs = global_rmse - local_rmse
                delta_mae_abs = global_mae - local_mae

                # For R²: only calculate if local_r2 is positive (negative R² makes relative comparison meaningless)
                delta_r2_rel = (delta_r2_abs / local_r2 * 100) if local_r2 > 0 else float('nan')
                delta_rmse_rel = (delta_rmse_abs / local_rmse * 100) if local_rmse != 0 else 0
                delta_mae_rel = (delta_mae_abs / local_mae * 100) if local_mae != 0 else 0

                comparison_data.append({
                    'station_id': station_id,
                    'R²_G': global_r2,
                    'R²_L': local_r2,
                    'Δ_R²_abs': delta_r2_abs,
                    'Δ_R²_rel': delta_r2_rel,
                    'RMSE_G': global_rmse,
                    'RMSE_L': local_rmse,
                    'Δ_RMSE_abs': delta_rmse_abs,
                    'Δ_RMSE_rel': delta_rmse_rel,
                    'MAE_G': global_mae,
                    'MAE_L': local_mae,
                    'Δ_MAE_abs': delta_mae_abs,
                    'Δ_MAE_rel': delta_mae_rel,
                })

            # Create DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('station_id')
            comparison_df = comparison_df.set_index('station_id')

            # Calculate mean and std for base metrics only
            metric_cols = ['R²_G', 'R²_L', 'RMSE_G', 'RMSE_L', 'MAE_G', 'MAE_L']
            mean_metrics = comparison_df[metric_cols].mean()
            std_metrics = comparison_df[metric_cols].std()

            # Compute deltas from the mean metrics
            mean_r2_g = mean_metrics['R²_G']
            mean_r2_l = mean_metrics['R²_L']
            mean_rmse_g = mean_metrics['RMSE_G']
            mean_rmse_l = mean_metrics['RMSE_L']
            mean_mae_g = mean_metrics['MAE_G']
            mean_mae_l = mean_metrics['MAE_L']

            # Calculate deltas from means
            mean_delta_r2_abs = mean_r2_g - mean_r2_l
            mean_delta_rmse_abs = mean_rmse_g - mean_rmse_l
            mean_delta_mae_abs = mean_mae_g - mean_mae_l

            mean_delta_r2_rel = (mean_delta_r2_abs / mean_r2_l * 100) if mean_r2_l > 0 else float('nan')
            mean_delta_rmse_rel = (mean_delta_rmse_abs / mean_rmse_l * 100) if mean_rmse_l != 0 else 0
            mean_delta_mae_rel = (mean_delta_mae_abs / mean_mae_l * 100) if mean_mae_l != 0 else 0

            # Create mean row
            mean_row = pd.Series({
                'R²_G': mean_r2_g,
                'R²_L': mean_r2_l,
                'Δ_R²_abs': mean_delta_r2_abs,
                'Δ_R²_rel': mean_delta_r2_rel,
                'RMSE_G': mean_rmse_g,
                'RMSE_L': mean_rmse_l,
                'Δ_RMSE_abs': mean_delta_rmse_abs,
                'Δ_RMSE_rel': mean_delta_rmse_rel,
                'MAE_G': mean_mae_g,
                'MAE_L': mean_mae_l,
                'Δ_MAE_abs': mean_delta_mae_abs,
                'Δ_MAE_rel': mean_delta_mae_rel,
            })

            # For std, just compute std of metrics (deltas std would need more complex calculation)
            std_delta_cols = comparison_df[['Δ_R²_abs', 'Δ_RMSE_abs', 'Δ_MAE_abs']].std()
            std_delta_rel_cols = comparison_df[['Δ_R²_rel', 'Δ_RMSE_rel', 'Δ_MAE_rel']].std()

            std_row = pd.Series({
                'R²_G': std_metrics['R²_G'],
                'R²_L': std_metrics['R²_L'],
                'Δ_R²_abs': std_delta_cols['Δ_R²_abs'],
                'Δ_R²_rel': std_delta_rel_cols['Δ_R²_rel'],
                'RMSE_G': std_metrics['RMSE_G'],
                'RMSE_L': std_metrics['RMSE_L'],
                'Δ_RMSE_abs': std_delta_cols['Δ_RMSE_abs'],
                'Δ_RMSE_rel': std_delta_rel_cols['Δ_RMSE_rel'],
                'MAE_G': std_metrics['MAE_G'],
                'MAE_L': std_metrics['MAE_L'],
                'Δ_MAE_abs': std_delta_cols['Δ_MAE_abs'],
                'Δ_MAE_rel': std_delta_rel_cols['Δ_MAE_rel'],
            })

            # Add mean and std rows
            comparison_df.loc['mean'] = mean_row
            comparison_df.loc['std'] = std_row

        # Display the comparison table with formatting
        st.subheader("📊 Vergleichstabelle")

        # Format DataFrame for display (without using pandas.style to avoid jinja2 dependency)
        display_df = comparison_df.copy()

        # Helper function to add color emoji based on metric type and value
        def format_delta_with_color(value, metric_type, is_percentage=False):
            """Add color emoji to delta values"""
            if pd.isna(value):
                return ""

            # Determine if global model is better
            if metric_type == 'r2':
                # R²: higher is better, so positive delta means global is better
                is_global_better = value > 0
            else:
                # RMSE/MAE: lower is better, so negative delta means global is better
                is_global_better = value < 0

            # Format the value
            if is_percentage:
                formatted = f"{value:.2f}%"
            else:
                formatted = f"{value:.4f}"

            # Add emoji
            if is_global_better:
                return f"🟢 {formatted}"
            else:
                return f"🔴 {formatted}"

        # Round and format numeric columns with color indicators
        for col in display_df.columns:
            if col.startswith('Δ_R²'):
                # R² deltas
                is_percentage = 'rel' in col
                display_df[col] = display_df[col].apply(
                    lambda x: format_delta_with_color(x, 'r2', is_percentage)
                )
            elif col.startswith('Δ_RMSE'):
                # RMSE deltas
                is_percentage = 'rel' in col
                display_df[col] = display_df[col].apply(
                    lambda x: format_delta_with_color(x, 'rmse', is_percentage)
                )
            elif col.startswith('Δ_MAE'):
                # MAE deltas
                is_percentage = 'rel' in col
                display_df[col] = display_df[col].apply(
                    lambda x: format_delta_with_color(x, 'mae', is_percentage)
                )
            else:
                # Regular metric columns (non-delta)
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")

        # Display dataframe
        st.dataframe(display_df, height=600)

        # Add color legend
        st.markdown("""
        **Legende für Delta-Werte:**
        - 🟢 Positives Δ bei R²: Globales Modell besser
        - 🔴 Negatives Δ bei R²: Lokales Modell besser
        - 🟢 Negatives Δ bei RMSE/MAE: Globales Modell besser
        - 🔴 Positives Δ bei RMSE/MAE: Lokales Modell besser
        """)

        # Show summary statistics
        st.markdown("---")
        st.subheader("📈 Zusammenfassung")

        col1, col2, col3 = st.columns(3)

        # Get numeric values for metrics
        mean_delta_r2 = comparison_df.loc['mean', 'Δ_R²_abs']
        mean_delta_r2_rel = comparison_df.loc['mean', 'Δ_R²_rel']
        mean_delta_rmse = comparison_df.loc['mean', 'Δ_RMSE_abs']
        mean_delta_rmse_rel = comparison_df.loc['mean', 'Δ_RMSE_rel']
        mean_delta_mae = comparison_df.loc['mean', 'Δ_MAE_abs']
        mean_delta_mae_rel = comparison_df.loc['mean', 'Δ_MAE_rel']

        with col1:
            st.metric(
                "Durchschn. Δ R²",
                f"{mean_delta_r2:.4f}",
                delta=f"{mean_delta_r2_rel:.2f}%"
            )

        with col2:
            st.metric(
                "Durchschn. Δ RMSE",
                f"{mean_delta_rmse:.4f}",
                delta=f"{mean_delta_rmse_rel:.2f}%",
                delta_color="inverse"  # Lower is better for RMSE
            )

        with col3:
            st.metric(
                "Durchschn. Δ MAE",
                f"{mean_delta_mae:.4f}",
                delta=f"{mean_delta_mae_rel:.2f}%",
                delta_color="inverse"  # Lower is better for MAE
            )

        # Scatter Plot Visualization
        st.markdown("---")
        st.subheader("📊 Scatter Plot: Global vs. Local Model Performance")

        # Filter out mean and std rows for plotting
        plot_df = comparison_df[~comparison_df.index.isin(['mean', 'std'])].copy()

        if not plot_df.empty:
            # Create subplots: 1 row, 3 columns
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=("R² Score", "RMSE", "MAE"),
                horizontal_spacing=0.12
            )

            # Define common styling
            marker_size = 10
            marker_color = 'rgb(31, 119, 180)'
            line_color = 'rgb(200, 200, 200)'
            font_size = 18
            title_font_size = 20

            # Plot 1: R²
            r2_local = plot_df['R²_L'].values
            r2_global = plot_df['R²_G'].values

            # Add scatter points
            fig.add_trace(
                go.Scatter(
                    x=r2_local,
                    y=r2_global,
                    mode='markers',
                    name='R²',
                    marker=dict(size=marker_size, color=marker_color),
                    showlegend=False,
                    hovertemplate='Local: %{x:.3f}<br>Global: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )

            # Add diagonal reference line (y=x)
            r2_min = min(r2_local.min(), r2_global.min())
            r2_max = max(r2_local.max(), r2_global.max())
            fig.add_trace(
                go.Scatter(
                    x=[r2_min, r2_max],
                    y=[r2_min, r2_max],
                    mode='lines',
                    name='y=x',
                    line=dict(color=line_color, width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

            # Plot 2: RMSE
            rmse_local = plot_df['RMSE_L'].values
            rmse_global = plot_df['RMSE_G'].values

            fig.add_trace(
                go.Scatter(
                    x=rmse_local,
                    y=rmse_global,
                    mode='markers',
                    name='RMSE',
                    marker=dict(size=marker_size, color=marker_color),
                    showlegend=False,
                    hovertemplate='Local: %{x:.4f}<br>Global: %{y:.4f}<extra></extra>'
                ),
                row=1, col=2
            )

            # Add diagonal reference line
            rmse_min = min(rmse_local.min(), rmse_global.min())
            rmse_max = max(rmse_local.max(), rmse_global.max())
            fig.add_trace(
                go.Scatter(
                    x=[rmse_min, rmse_max],
                    y=[rmse_min, rmse_max],
                    mode='lines',
                    name='y=x',
                    line=dict(color=line_color, width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=2
            )

            # Plot 3: MAE
            mae_local = plot_df['MAE_L'].values
            mae_global = plot_df['MAE_G'].values

            fig.add_trace(
                go.Scatter(
                    x=mae_local,
                    y=mae_global,
                    mode='markers',
                    name='MAE',
                    marker=dict(size=marker_size, color=marker_color),
                    showlegend=False,
                    hovertemplate='Local: %{x:.4f}<br>Global: %{y:.4f}<extra></extra>'
                ),
                row=1, col=3
            )

            # Add diagonal reference line
            mae_min = min(mae_local.min(), mae_global.min())
            mae_max = max(mae_local.max(), mae_global.max())
            fig.add_trace(
                go.Scatter(
                    x=[mae_min, mae_max],
                    y=[mae_min, mae_max],
                    mode='lines',
                    name='y=x',
                    line=dict(color=line_color, width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=3
            )

            # Update axes labels with larger fonts (English)
            fig.update_xaxes(title_text="Local Model", row=1, col=1, title_font=dict(size=font_size), tickfont=dict(size=font_size-2))
            fig.update_yaxes(title_text="Global Model", row=1, col=1, title_font=dict(size=font_size), tickfont=dict(size=font_size-2))

            fig.update_xaxes(title_text="Local Model", row=1, col=2, title_font=dict(size=font_size), tickfont=dict(size=font_size-2))
            fig.update_yaxes(title_text="Global Model", row=1, col=2, title_font=dict(size=font_size), tickfont=dict(size=font_size-2))

            fig.update_xaxes(title_text="Local Model", row=1, col=3, title_font=dict(size=font_size), tickfont=dict(size=font_size-2))
            fig.update_yaxes(title_text="Global Model", row=1, col=3, title_font=dict(size=font_size), tickfont=dict(size=font_size-2))

            # Update subplot titles font size
            for annotation in fig['layout']['annotations']:
                annotation['font'] = dict(size=title_font_size)

            # Update layout
            fig.update_layout(
                height=400,
                width=1400,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=font_size)
            )

            # Make grid lines visible
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

            # Export option for high-resolution PNG
            st.markdown("**Export Plot:**")

            try:
                # Create high-resolution image
                # Use same dimensions as displayed (1400x400) but higher scale for resolution
                # scale=6 gives approx 600 DPI when exported
                img_bytes = fig.to_image(format="png", width=1400, height=400, scale=6)

                st.download_button(
                    label="📥 Download Plot as PNG (600 DPI)",
                    data=img_bytes,
                    file_name=f"comparison_scatter_{model_1_name}_vs_{model_2_name}.png",
                    mime="image/png"
                )
            except Exception as e:
                st.info("""
                **PNG Export nicht verfügbar**

                Für den hochauflösenden PNG-Export wird Chrome benötigt.

                **Installation:**
                ```bash
                pip install -U kaleido
                plotly_get_chrome
                ```

                **Alternative:** Nutzen Sie die Browser-Screenshot-Funktion oder den Plotly-eigenen Export-Button (📷 oben rechts im Plot).
                """)
        else:
            st.warning("Keine Daten für Visualisierung verfügbar")

        # Grouped Bar Chart Visualization
        st.markdown("---")
        st.subheader("📊 Grouped Bar Chart: Station-wise Model Comparison")

        if not plot_df.empty:
            # Sort by station_id for consistent ordering
            plot_df_sorted = plot_df.sort_index()
            # Explicitly convert to strings to prevent numeric interpretation
            station_ids = [str(sid) for sid in plot_df_sorted.index.tolist()]

            # Create subplots: 1 row, 3 columns for horizontal bar charts
            fig_bar = make_subplots(
                rows=1, cols=3,
                subplot_titles=("R² Score", "RMSE", "MAE"),
                horizontal_spacing=0.10
            )

            # Define common styling
            color_global = 'rgb(31, 119, 180)'  # Blue
            color_local = 'rgb(255, 127, 14)'   # Orange
            font_size = 18
            title_font_size = 20

            # Calculate height based on number of stations (minimum 400, scale with stations)
            height = max(400, len(station_ids) * 25)

            # Plot 1: R²
            fig_bar.add_trace(
                go.Bar(
                    y=station_ids,
                    x=plot_df_sorted['R²_G'].values,
                    name='Global',
                    orientation='h',
                    marker=dict(color=color_global),
                    showlegend=True,
                    hovertemplate='Global: %{x:.3f}<extra></extra>'
                ),
                row=1, col=1
            )

            fig_bar.add_trace(
                go.Bar(
                    y=station_ids,
                    x=plot_df_sorted['R²_L'].values,
                    name='Local',
                    orientation='h',
                    marker=dict(color=color_local),
                    showlegend=True,
                    hovertemplate='Local: %{x:.3f}<extra></extra>'
                ),
                row=1, col=1
            )

            # Plot 2: RMSE
            fig_bar.add_trace(
                go.Bar(
                    y=station_ids,
                    x=plot_df_sorted['RMSE_G'].values,
                    name='Global',
                    orientation='h',
                    marker=dict(color=color_global),
                    showlegend=False,
                    hovertemplate='Global: %{x:.4f}<extra></extra>'
                ),
                row=1, col=2
            )

            fig_bar.add_trace(
                go.Bar(
                    y=station_ids,
                    x=plot_df_sorted['RMSE_L'].values,
                    name='Local',
                    orientation='h',
                    marker=dict(color=color_local),
                    showlegend=False,
                    hovertemplate='Local: %{x:.4f}<extra></extra>'
                ),
                row=1, col=2
            )

            # Plot 3: MAE
            fig_bar.add_trace(
                go.Bar(
                    y=station_ids,
                    x=plot_df_sorted['MAE_G'].values,
                    name='Global',
                    orientation='h',
                    marker=dict(color=color_global),
                    showlegend=False,
                    hovertemplate='Global: %{x:.4f}<extra></extra>'
                ),
                row=1, col=3
            )

            fig_bar.add_trace(
                go.Bar(
                    y=station_ids,
                    x=plot_df_sorted['MAE_L'].values,
                    name='Local',
                    orientation='h',
                    marker=dict(color=color_local),
                    showlegend=False,
                    hovertemplate='Local: %{x:.4f}<extra></extra>'
                ),
                row=1, col=3
            )

            # Update axes labels with larger fonts (English)
            fig_bar.update_xaxes(title_text="R² Score", row=1, col=1, title_font=dict(size=font_size), tickfont=dict(size=font_size-2))
            fig_bar.update_yaxes(title_text="Station ID", row=1, col=1, title_font=dict(size=font_size), tickfont=dict(size=font_size-4), type='category')

            fig_bar.update_xaxes(title_text="RMSE", row=1, col=2, title_font=dict(size=font_size), tickfont=dict(size=font_size-2))
            fig_bar.update_yaxes(title_text="", row=1, col=2, title_font=dict(size=font_size), tickfont=dict(size=font_size-4), showticklabels=True, type='category')

            fig_bar.update_xaxes(title_text="MAE", row=1, col=3, title_font=dict(size=font_size), tickfont=dict(size=font_size-2))
            fig_bar.update_yaxes(title_text="", row=1, col=3, title_font=dict(size=font_size), tickfont=dict(size=font_size-4), showticklabels=True, type='category')

            # Update subplot titles font size
            for annotation in fig_bar['layout']['annotations']:
                annotation['font'] = dict(size=title_font_size)

            # Update layout
            fig_bar.update_layout(
                height=height,
                width=1400,
                barmode='group',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=font_size),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font=dict(size=font_size)
                )
            )

            # Make grid lines visible
            fig_bar.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig_bar.update_yaxes(showgrid=False)

            # Display the plot
            st.plotly_chart(fig_bar, use_container_width=True)

            # Export option for high-resolution PNG
            st.markdown("**Export Bar Chart:**")

            try:
                # Create high-resolution image
                img_bytes_bar = fig_bar.to_image(format="png", width=1400, height=height, scale=6)

                st.download_button(
                    label="📥 Download Bar Chart as PNG (600 DPI)",
                    data=img_bytes_bar,
                    file_name=f"comparison_barchart_{model_1_name}_vs_{model_2_name}.png",
                    mime="image/png"
                )
            except Exception as e:
                st.info("""
                **PNG Export nicht verfügbar**

                Für den hochauflösenden PNG-Export wird Chrome benötigt.

                **Alternative:** Nutzen Sie die Browser-Screenshot-Funktion oder den Plotly-eigenen Export-Button (📷 oben rechts im Plot).
                """)
        else:
            st.warning("Keine Daten für Visualisierung verfügbar")

        # Excel Export
        st.markdown("---")
        excel_filename = f"{model_1_name}_vs_{model_2_name}.xlsx"

        # Create Excel file in memory
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            comparison_df.to_excel(writer, sheet_name='Comparison')

        excel_data = output.getvalue()

        st.download_button(
            label="📥 Vergleich als Excel herunterladen",
            data=excel_data,
            file_name=excel_filename,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )

