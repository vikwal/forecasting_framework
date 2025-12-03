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
tab1, tab2 = st.tabs(["📈 Aggregierte Ansicht", "🔍 Detailansicht"])

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

                # Konfigurations-Info
                with st.expander("🔧 Konfiguration anzeigen"):
                    params = cl_results['config']['params']
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
                        st.write('**More Info:**')
                        st.write(f'- Next n grid points: {params.get("next_n_grid_points", "N/A")}')
                        st.write(f'- Aggregate grid points: {params.get("aggregate_grid_points", "N/A")}')
                        st.write(f'- NWP models: {params["openmeteo"].get("nwp_models", "N/A")}')
                        st.write(f'- Turbines per park: {params.get("turbines_per_park", "N/A")}')

                # Metriken anzeigen
                st.subheader("📊 Evaluations-Metriken")
                eval_df = cl_results['evaluation'][['R^2', 'RMSE', 'MAE']].copy()
                if 'Skill' in cl_results['evaluation'].columns:
                    eval_df['Skill'] = cl_results['evaluation']['Skill']
                if 'key' in cl_results['evaluation'].columns:
                    eval_df['key'] = cl_results['evaluation']['key']

                st.dataframe(eval_df.round(4))

                # Training History Plot
                fontsize=11
                st.subheader("📈 Training History")
                if 'history' in cl_results and cl_results['history']:
                    plot_interactive_history(cl_results['history'], key_prefix=f"cl_{selected_sim_key}")
                    # fig, ax = plt.subplots(figsize=(7, 3))

                    # if 'loss' in cl_results['history']:
                    #     ax.plot(cl_results['history']['loss'], label='Train Loss')
                    # if 'val_loss' in cl_results['history']:
                    #     ax.plot(cl_results['history']['val_loss'], label='Val Loss')

                    # ax.set_xlabel('Epochs', fontsize=fontsize)
                    # ax.set_ylabel('Loss', fontsize=fontsize)
                    # ax.legend(fontsize=fontsize)
                    # ax.grid(True, alpha=0.3)

                    # st.pyplot(fig)
                    # plt.close()
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

