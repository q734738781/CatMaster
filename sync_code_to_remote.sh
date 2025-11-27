#!/bin/bash
# Sync catmaster code to remote workers
# Syncs from local catmaster/ directory to remote PYTHONPATH locations

set -e  # Exit on error

echo "======================================================================="
echo "Syncing CatMaster Code to Remote Workers"
echo "======================================================================="

# CPU Worker - sync to /public/home/chenhh/catmaster/
echo ""
echo "→ Syncing to CPU worker (166.111.35.183:31125)..."
echo "   Target: /public/home/chenhh/catmaster/"

rsync -avz --delete \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='*.egg-info' \
  --exclude='catmaster_workspace' \
  --exclude='test_download' \
  --exclude='.git' \
  --exclude='*.log' \
  --exclude='.pytest_cache' \
  -e "ssh -p 31125" \
  catmaster/ \
  chenhh@166.111.35.183:/public/home/chenhh/catmaster_code/catmaster/

if [ $? -eq 0 ]; then
    echo "   [OK] CPU worker sync successful"
else
    echo "   [FAIL] CPU worker sync failed"
    exit 1
fi

# GPU Worker - sync to /ssd/chenhh/catmaster/
echo ""
echo "→ Syncing to GPU worker (166.111.35.177:22)..."
echo "   Target: /ssd/chenhh/catmaster/"

rsync -avz --delete \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='*.egg-info' \
  --exclude='catmaster_workspace' \
  --exclude='test_download' \
  --exclude='.git' \
  --exclude='*.log' \
  --exclude='.pytest_cache' \
  catmaster/ \
  chenhh@166.111.35.177:/ssd/chenhh/catmaster_code/catmaster/

if [ $? -eq 0 ]; then
    echo "   [OK] GPU worker sync successful"
else
    echo "   [FAIL] GPU worker sync failed"
    exit 1
fi

echo ""
echo "======================================================================="
echo "[OK] Code Sync Completed Successfully!"
echo "======================================================================="
echo ""
echo "Synced directories:"
echo "  CPU: /public/home/chenhh/catmaster_code/catmaster/"
echo "  GPU: /ssd/chenhh/catmaster_code/catmaster/"
echo ""
echo "These match the PYTHONPATH settings in your worker configs:"
echo "  CPU pre_run: export PYTHONPATH=/public/home/chenhh/catmaster_code:\$PYTHONPATH"
echo "  GPU pre_run: export PYTHONPATH=/ssd/chenhh/catmaster_code:\$PYTHONPATH"
echo ""
echo "Now you can run: python demo_o2_calculation.py"

