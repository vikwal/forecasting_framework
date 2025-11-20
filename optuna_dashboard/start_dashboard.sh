#!/bin/bash

# Optuna Dashboard Starter Script
# Dieses Skript startet das Streamlit Dashboard für Optuna HPO Monitoring

echo "🚀 Starte Optuna Dashboard..."

# Aktiviere die virtuelle Umgebung (falls vorhanden)
if [ -d "../frcst" ]; then
    echo "🐍 Aktiviere virtuelle Umgebung..."
    source ../frcst/bin/activate
fi

# Setze Umgebungsvariablen
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Navigiere zum Dashboard Verzeichnis
cd "$(dirname "$0")"

# Starte Streamlit
echo "🌐 Dashboard verfügbar unter: http://localhost:8501"
echo "📊 Drücke Ctrl+C zum Beenden"

streamlit run optuna_dashboard.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --server.headless=true \
    --browser.gatherUsageStats=false