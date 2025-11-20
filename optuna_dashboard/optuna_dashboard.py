import streamlit as st
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
from optuna.storages import RDBStorage
from sklearn.linear_model import LinearRegression
import time
from datetime import datetime
import seaborn as sns

# Configuration
st.set_page_config(
    page_title="Optuna HPO Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Improve checkbox and text display */
    .stCheckbox > label {
        width: 100% !important;
        max-width: none !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        font-size: 14px !important;
    }
    .stExpander > div > div > div {
        padding: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=30)  # Cache für 30 Sekunden
def get_available_studies(studies_dir="./data"):
    """Finds all available study files in the data directory"""
    import glob

    pattern = os.path.join(studies_dir, "*.db")
    study_files = glob.glob(pattern)

    study_info = []
    for file_path in study_files:
        filename = os.path.basename(file_path)
        study_name = os.path.splitext(filename)[0]
        study_info.append({
            'name': study_name,
            'file_path': file_path,
            'filename': filename
        })

    return study_info

@st.cache_data(ttl=30)  # Cache for 30 seconds
def load_optuna_data(study_file_path):
    """Loads a single Optuna study file and prepares the data"""
    try:
        storage = RDBStorage(url=f"sqlite:///{study_file_path}")
        study_summaries = optuna.get_all_study_summaries(storage=storage)

        if not study_summaries:
            return None, None

        study_names = [summary.study_name for summary in study_summaries]
        studies = [optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_file_path}")
                  for study_name in study_names]

        entries = []
        for study in studies:
            completed_trials = [trial for trial in study.trials
                              if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE]
            pruned_trials = [trial for trial in study.trials
                           if trial.state == optuna.trial.TrialState.PRUNED]

            study_name = study.study_name
            study_setting = study_name[0:2] if len(study_name) > 2 else "NA"

            # Regex for various parameters
            model_match = re.search(r'_m-([^_]+)', study_name)
            model_type = model_match.group(1) if model_match else "Unknown"

            out_match = re.search(r'_out-([^_]+)', study_name)
            out_type = out_match.group(1) if out_match else "Unknown"

            freq_match = re.search(r'_freq-([^_]+)', study_name)
            freq_type = freq_match.group(1) if freq_match else "Unknown"

            additional = None
            try:
                best_value = study.best_trial.value if study.best_trial else None
                best_params = study.best_trial.params if study.best_trial else {}
            except (ValueError, AttributeError):
                best_value = None
                best_params = {}

            if len(study_name.split('_')) > 4:
                additional = study_name.split('_')[-1]

            entry = {
                'study_name': study.study_name,
                'model': model_type,
                'setting': study_setting,
                'output_type': out_type,
                'frequency': freq_type,
                'additional': additional,
                'n_trials': len(study.trials),
                'n_completed_trials': len(completed_trials),
                'n_pruned_trials': len(pruned_trials),
                'best_value': round(best_value, 4) if best_value is not None else None,
                'best_params': best_params,
                'last_trial_time': max([trial.datetime_complete for trial in completed_trials
                                      if trial.datetime_complete], default=None)
            }
            entries.append(entry)

        results_df = pd.DataFrame(entries)
        return results_df, studies

    except Exception as e:
        st.error(f"Error loading Optuna data: {str(e)}")
        return None, None

def create_progress_plot(study, study_name):
    """Creates a Plotly progress plot for a study"""
    trials = [trial for trial in study.trials
              if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE]

    if not trials:
        return None

    sorted_by_number = sorted(trials, key=lambda x: x.number)
    progress = [trial.value for trial in sorted_by_number]
    trial_numbers = [trial.number for trial in sorted_by_number]

    # Linear regression for trend
    x = np.arange(len(progress)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, progress)
    y_pred = lr.predict(x)
    improvement_per_trial = lr.coef_[0] * -1

    fig = go.Figure()

    # Progress Line
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=progress,
        mode='lines+markers',
        name='HPO Progress',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))

    # Trend Line
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=y_pred,
        mode='lines',
        name='Trend',
        line=dict(color='red', dash='dash', width=2)
    ))

    fig.update_layout(
        title=f'HPO Progress: {study_name}',
        xaxis_title='Trial Number',
        yaxis_title='Objective Value',
        height=400,
        showlegend=True
    )

    return fig, improvement_per_trial

def get_parameter_color_palette():
    """Returns a fixed color palette for parameters"""
    return {
        'batch_size': '#FF6B35',      # Orange
        'learning_rate': '#004E89',   # Blue
        'lr': '#004E89',              # Blue (alternative name)
        'dropout': '#95190C',         # Red
        'n_estimators': '#7209B7',    # Purple
        'max_depth': '#2F9F3F',       # Green
        'hidden_size': '#F77E21',     # Orange variant
        'hidden_dim': '#FFD23F',      # Yellow (different from hidden_size)
        'n_heads': '#8338EC',         # Violet (different from hidden_dim)
        'num_layers': '#A663CC',      # Light Purple
        'epochs': '#FF006E',          # Pink
        'alpha': '#06FFA5',           # Mint Green
        'gamma': '#FB5607',           # Orange-Red
        'C': '#4CC9F0',               # Light Blue
        'kernel': '#F72585',          # Magenta
        'filters': '#00B4D8',         # Cyan (unique)
        'kernel_size': '#90E0EF',     # Light Cyan (unique)
        'units': '#0077B6',           # Dark Blue (unique)
        'n_components': '#2F9F3F',    # Green variant
        'min_samples_split': '#7209B7', # Purple variant
        'min_samples_leaf': '#FF6B35',  # Orange variant
        'subsample': '#95190C',       # Red variant
        'colsample_bytree': '#004E89', # Blue variant
        'reg_alpha': '#F77E21',       # Orange variant
        'reg_lambda': '#FFD23F',      # Yellow variant
        'window_size': '#8338EC',     # Violet variant
        'seq_length': '#06FFA5',      # Mint variant
        'embed_dim': '#FB5607',       # Orange-Red variant
        'num_attention_heads': '#4CC9F0', # Light Blue variant
        'intermediate_size': '#F72585',   # Magenta variant
        'stride': '#CAF0F8',          # Very Light Blue
        'padding': '#ADE8F4',         # Light Blue variant
        'activation': '#023E8A',      # Dark Blue variant
        'n_cnn_layers': '#7B2CBF',    # Deep Purple (unique)
        'n_rnn_layers': '#E71D36',    # Red-Pink (unique)
    }

def get_parameter_symbol_mapping():
    """Returns a mapping of parameters to plotly symbols"""
    symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up',
              'triangle-down', 'triangle-left', 'triangle-right', 'pentagon',
              'hexagon', 'star', 'hourglass', 'bowtie', 'asterisk', 'hash',
              'circle-open', 'square-open', 'diamond-open', 'cross-open',
              'triangle-ne', 'triangle-nw']

    return {
        'batch_size': 'circle',
        'learning_rate': 'square',
        'lr': 'square',
        'dropout': 'diamond',
        'n_estimators': 'cross',
        'max_depth': 'triangle-up',
        'hidden_size': 'triangle-down',
        'hidden_dim': 'pentagon',
        'n_heads': 'hexagon',
        'num_layers': 'star',
        'epochs': 'triangle-left',
        'alpha': 'triangle-right',
        'gamma': 'hourglass',
        'C': 'bowtie',
        'kernel': 'asterisk',
        'filters': 'circle-open',         # Unique symbol
        'kernel_size': 'square-open',     # Unique symbol
        'units': 'diamond-open',          # Unique symbol
        'n_components': 'hash',
        'min_samples_split': 'cross-open',
        'min_samples_leaf': 'triangle-up',
        'subsample': 'triangle-down',
        'colsample_bytree': 'pentagon',
        'reg_alpha': 'hexagon',
        'reg_lambda': 'star',
        'window_size': 'hourglass',
        'seq_length': 'bowtie',
        'embed_dim': 'asterisk',
        'num_attention_heads': 'hash',
        'intermediate_size': 'circle',
        'stride': 'square',
        'padding': 'diamond',
        'activation': 'cross',
        'n_cnn_layers': 'triangle-ne',    # Unique symbol
        'n_rnn_layers': 'triangle-nw',    # Unique symbol
    }

def get_all_parameters_from_studies(studies):
    """Extract all unique parameters from all studies"""
    all_params = set()
    for study in studies:
        for trial in study.trials:
            if trial.params:
                all_params.update(trial.params.keys())
    return sorted(list(all_params))

def create_progress_plot_with_parameters(study, study_name, selected_params=None):
    """Creates a progress plot with optional parameter visualization"""
    trials = [trial for trial in study.trials
              if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE]

    if not trials:
        return None, None

    sorted_by_number = sorted(trials, key=lambda x: x.number)
    progress = [trial.value for trial in sorted_by_number]
    trial_numbers = [trial.number for trial in sorted_by_number]

    # Linear regression for trend
    x = np.arange(len(progress)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, progress)
    y_pred = lr.predict(x)
    improvement_per_trial = lr.coef_[0] * -1

    # Create figure with optional secondary y-axis
    if selected_params and len(selected_params) == 1:
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=[f'HPO Progress: {study_name}']
        )
    else:
        fig = go.Figure()

    # Main progress line (ohne Marker)
    fig.add_trace(
        go.Scatter(
            x=trial_numbers,
            y=progress,
            mode='lines',  # Entferne 'markers'
            name='HPO Progress',
            line=dict(color='#1f77b4', width=2)
        )
    )

    # Trend line
    fig.add_trace(
        go.Scatter(
            x=trial_numbers,
            y=y_pred,
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash', width=2)
        )
    )

    # Add parameter traces if selected
    if selected_params:
        color_palette = get_parameter_color_palette()
        symbol_mapping = get_parameter_symbol_mapping()

        for param in selected_params:
            param_values = []
            param_trial_numbers = []

            for trial in sorted_by_number:
                if param in trial.params:
                    param_values.append(trial.params[param])
                    param_trial_numbers.append(trial.number)

            if param_values:
                color = color_palette.get(param, '#808080')  # Default gray if not in palette
                symbol = symbol_mapping.get(param, 'circle')  # Default circle if not in mapping

                if len(selected_params) == 1:
                    # Single parameter: use secondary y-axis
                    fig.add_trace(
                        go.Scatter(
                            x=param_trial_numbers,
                            y=param_values,
                            mode='markers',
                            name=f'{param}',
                            marker=dict(color=color, size=8, symbol=symbol),
                            yaxis='y2'
                        )
                    )
                else:
                    # Multiple parameters: use main plot with legend
                    # Normalize parameter values to fit in main plot range
                    if param_values:
                        param_min, param_max = min(param_values), max(param_values)
                        progress_min, progress_max = min(progress), max(progress)

                        if param_max != param_min:
                            normalized_values = [
                                progress_min + (val - param_min) / (param_max - param_min) * (progress_max - progress_min)
                                for val in param_values
                            ]
                        else:
                            normalized_values = [progress_min] * len(param_values)

                        fig.add_trace(
                            go.Scatter(
                                x=param_trial_numbers,
                                y=normalized_values,
                                mode='markers',
                                name=f'{param} (normalized)',
                                marker=dict(color=color, size=8, symbol=symbol),
                                customdata=param_values,
                                hovertemplate=f'{param}: %{{customdata}}<br>Trial: %{{x}}<extra></extra>'
                            )
                        )    # Update layout
    if selected_params and len(selected_params) == 1:
        fig.update_layout(
            title=f'HPO Progress with {selected_params[0]}: {study_name}',
            xaxis_title='Trial Number',
            height=400,
            showlegend=True
        )
        fig.update_yaxes(title_text='Objective Value', secondary_y=False)
        fig.update_yaxes(title_text=selected_params[0], secondary_y=True)
    else:
        fig.update_layout(
            title=f'HPO Progress: {study_name}',
            xaxis_title='Trial Number',
            yaxis_title='Objective Value',
            height=400,
            showlegend=True
        )

    return fig, improvement_per_trial

def create_study_comparison_plot(results_df):
    """Creates a comparison plot between studies"""
    if results_df is None or results_df.empty:
        return None

    # Filter studies with best_value
    filtered_df = results_df[results_df['best_value'].notna()].copy()

    if filtered_df.empty:
        return None

    fig = px.scatter(
        filtered_df,
        x='n_completed_trials',
        y='best_value',
        color='model',
        size='n_trials',
        hover_data=['study_name', 'setting', 'frequency'],
        title='Study Comparison: Best Value vs Completed Trials'
    )

    fig.update_layout(height=500)
    return fig

def create_model_performance_plot(results_df):
    """Creates a box plot for model performance"""
    if results_df is None or results_df.empty:
        return None

    filtered_df = results_df[results_df['best_value'].notna()].copy()

    if filtered_df.empty or filtered_df['model'].nunique() <= 1:
        return None

    fig = px.box(
        filtered_df,
        x='model',
        y='best_value',
        title='Model Performance Distribution'
    )

    fig.update_layout(height=400)
    return fig# Hauptapp
def main():
    st.markdown('<h1 class="main-header">🔬 Optuna HPO Dashboard</h1>', unsafe_allow_html=True)

    # Session State initialisieren um Tab-Zustand zu erhalten
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    # Session State für Selektoren initialisieren
    if 'current_study' not in st.session_state:
        st.session_state.current_study = None
    if 'selected_params' not in st.session_state:
        st.session_state.selected_params = []

    # Sidebar für Konfiguration
    st.sidebar.header("⚙️ Konfiguration")

    studies_dir = "./data"

    # Lade verfügbare Studies
    available_studies = get_available_studies(studies_dir=studies_dir)

    if not available_studies:
        st.sidebar.error(f"Keine Study-Dateien in {studies_dir} gefunden!")
        st.error(f"Keine Study-Dateien gefunden! Stelle sicher, dass sich .db Dateien in {studies_dir} befinden.")
        st.stop()

    # Study Selector
    study_options = [study['name'] for study in available_studies]
    selected_study_name = st.sidebar.selectbox(
        "📊 Select Study:",
        options=study_options,
        index=0
    )

    # Finde die ausgewählte Study
    selected_study = next(study for study in available_studies if study['name'] == selected_study_name)
    study_file_path = selected_study['file_path']

    # Info über ausgewählte Study
    st.sidebar.info(f"📁 Datei: {selected_study['filename']}")

    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    refresh_button = st.sidebar.button("🔄 Manuell Aktualisieren")    # Info Box

    # Daten laden
    if refresh_button or auto_refresh:
        results_df, studies = load_optuna_data(study_file_path)
    else:
        results_df, studies = load_optuna_data(study_file_path)

    if results_df is None:
        st.error(f"Keine Optuna Studies in {selected_study['filename']} gefunden oder Fehler beim Laden!")
        st.stop()    # Header Metriken
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Aktive Studies", len(results_df))

    with col2:
        total_trials = results_df['n_trials'].sum()
        st.metric("🧪 Total Trials", total_trials)

    with col3:
        total_completed = results_df['n_completed_trials'].sum()
        st.metric("✅ Completed Trials", total_completed)

    with col4:
        best_overall = results_df['best_value'].min() if not results_df['best_value'].isna().all() else "N/A"
        st.metric("🏆 Best Overall Value", f"{best_overall:.4f}" if best_overall != "N/A" else "N/A")

    # Tabs for different views - mit Session State Key für Persistenz
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Study Overview", "🔄 Progress Tracking", "🏆 Best Trials", "📊 Comparison"])

    with tab1:
        st.subheader("Study Overview")

        # Filter
        col1, col2 = st.columns(2)
        with col1:
            model_filter = st.multiselect("Filter by Model:",
                                        options=results_df['model'].unique(),
                                        default=results_df['model'].unique())
        with col2:
            setting_filter = st.multiselect("Filter by Setting:",
                                          options=results_df['setting'].unique(),
                                          default=results_df['setting'].unique())

        # Filtered data
        filtered_df = results_df[
            (results_df['model'].isin(model_filter)) &
            (results_df['setting'].isin(setting_filter))
        ]

        # Display table
        display_columns = ['study_name', 'model', 'setting', 'n_trials', 'n_completed_trials',
                          'n_pruned_trials', 'best_value']
        st.dataframe(filtered_df[display_columns])

    with tab2:
        st.subheader("Progress Tracking")

        if studies:
            # Verwende 60/40 Split für mehr Platz für Study-Namen
            col1, col2 = st.columns([0.6, 0.4])

            with col1:
                study_names = [study.study_name for study in studies]

                # Use container for vertical display with improved presentation
                with st.container():
                    if len(study_names) <= 5:
                        # For few studies: Checkboxes for better visibility
                        selected_studies = []
                        for study_name in study_names:
                            # Use full names without truncation
                            if st.checkbox(
                                study_name,
                                value=True,
                                key=f"study_{study_name}"
                            ):
                                selected_studies.append(study_name)
                    else:
                        # For many studies: Use expandable section for full name display
                        with st.expander("📋 Study Selection (click to expand)", expanded=False):
                            selected_studies = []
                            for i, study_name in enumerate(study_names):
                                col_left, col_right = st.columns([0.1, 0.9])
                                with col_left:
                                    selected = st.checkbox(
                                        "Select study",
                                        value=i < 3,  # Default first 3 selected
                                        key=f"select_{study_name}",
                                        label_visibility="collapsed"
                                    )
                                with col_right:
                                    st.write(f"**{study_name}**")

                                if selected:
                                    selected_studies.append(study_name)

                        # Limit to 5 studies for performance
                        if len(selected_studies) > 5:
                            st.warning("⚠️ Please select maximum 5 studies for optimal performance.")
                            selected_studies = selected_studies[:5]

            with col2:
                # Get all available parameters from selected studies
                selected_study_objects = [study for study in studies if study.study_name in selected_studies]
                all_params = get_all_parameters_from_studies(selected_study_objects)

                # Filter Session State parameters to keep only available ones
                valid_cached_params = [param for param in st.session_state.selected_params if param in all_params]

                # Use container for consistent styling
                with st.container():

                    if len(all_params) <= 8:
                        # For few parameters: Direct checkboxes
                        selected_params = []
                        for param in all_params:
                            if st.checkbox(
                                param,
                                value=param in valid_cached_params,
                                key=f"param_{param}"
                            ):
                                selected_params.append(param)
                    else:
                        # For many parameters: Use expandable section
                        with st.expander("⚙️ Parameter Selection (click to expand)", expanded=False):
                            selected_params = []
                            for param in all_params:
                                if st.checkbox(
                                    param,
                                    value=param in valid_cached_params,
                                    key=f"param_exp_{param}"
                                ):
                                    selected_params.append(param)

                # Update Session State with valid parameters
                st.session_state.selected_params = selected_params            # Show parameter info
            if selected_params:
                if len(selected_params) == 1:
                    st.info(f"📊 Showing {selected_params[0]} on secondary y-axis")
                else:
                    st.info(f"📊 Showing {len(selected_params)} parameters (normalized) with fixed colors")

            for study in studies:
                if study.study_name in selected_studies:
                    if selected_params:
                        result = create_progress_plot_with_parameters(
                            study, study.study_name, selected_params
                        )
                    else:
                        result = create_progress_plot(study, study.study_name)

                    if result is not None:
                        fig, improvement = result
                        if fig:
                            st.plotly_chart(fig, width='stretch')
                            if improvement:
                                st.info(f"💡 Improvement per trial: {improvement:.6f}")

    with tab3:
        st.subheader("Best Trials Analysis")

        for i, study in enumerate(studies):
            with st.expander(f"🎯 {study.study_name}"):
                try:
                    if study.best_trial:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Best Value:**")
                            st.write(f"{study.best_trial.value:.6f}")

                        with col2:
                            st.write("**Best Parameters:**")
                            for key, value in study.best_trial.params.items():
                                st.write(f"• {key}: {value}")
                    else:
                        st.write("No best trial available yet.")
                except:
                    st.write("Error loading best trial.")

    with tab4:
        st.subheader("Study Comparison")

        # Scatter Plot
        comparison_fig = create_study_comparison_plot(results_df)
        if comparison_fig:
            st.plotly_chart(comparison_fig, width='stretch')

        # Model Performance Box Plot
        performance_fig = create_model_performance_plot(results_df)
        if performance_fig:
            st.plotly_chart(performance_fig, width='stretch')

    # Auto Refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()