#!/bin/bash
# Monitor kriging generation progress

LOG_FILE="/home/viktorwalter/Work/forecasting_framework/logs/kriging_generation.log"
OUTPUT_DIR="/home/viktorwalter/Work/forecasting_framework/data/kriging_stations"

echo "=== Kriging Generation Monitor ==="
echo ""

# Check if process is running
if ps aux | grep -q "[g]enerate_kriging_timeseries.py"; then
    echo "✓ Process is running"
    PID=$(ps aux | grep "[g]enerate_kriging_timeseries.py" | awk '{print $2}')
    echo "  PID: $PID"
    
    # Get CPU and memory usage
    CPU=$(ps aux | grep "[g]enerate_kriging_timeseries.py" | awk '{print $3}')
    MEM=$(ps aux | grep "[g]enerate_kriging_timeseries.py" | awk '{print $4}')
    echo "  CPU: ${CPU}%, Memory: ${MEM}%"
else
    echo "✗ Process not running"
fi

echo ""
echo "=== Latest log entries ==="
tail -15 "$LOG_FILE" 2>/dev/null || echo "Log file not found"

echo ""
echo "=== Output progress ==="
if [ -d "$OUTPUT_DIR" ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR"/kriging_*.csv 2>/dev/null | wc -l)
    echo "Files generated: $COUNT / 160"
    
    if [ $COUNT -gt 0 ]; then
        echo ""
        echo "Latest files:"
        ls -lt "$OUTPUT_DIR"/kriging_*.csv 2>/dev/null | head -5
    fi
else
    echo "Output directory not yet created"
fi
