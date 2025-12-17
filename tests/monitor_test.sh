#!/bin/bash
# Monitor robust HMM testing progress

LOG_FILE="robust_test_full.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

echo "================================"
echo "HMM ROBUST TESTING - PROGRESS"
echo "================================"
echo ""

# Count completed stocks
completed=$(grep -c "✓.*completed" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Stocks completed: $completed / 60"
echo ""

# Show category progress
echo "Categories tested:"
grep "CATEGORY:" "$LOG_FILE" | tail -5
echo ""

# Show current stock
echo "Currently testing:"
grep "--- Testing" "$LOG_FILE" | tail -1
echo ""

# Show recent results
echo "Latest results:"
grep "✓.*completed" "$LOG_FILE" | tail -3
echo ""

# Check for errors
errors=$(grep -c "✗.*failed" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Errors encountered: $errors"

# Show progress percentage
progress=$((completed * 100 / 60))
echo ""
echo "Overall progress: $progress%"
echo "================================"
