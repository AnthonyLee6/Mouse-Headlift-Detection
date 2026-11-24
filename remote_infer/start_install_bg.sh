#!/bin/bash
# run install detached
nohup bash /root/dlc_infer_2166_2171/install_dlc.sh > /root/dlc_infer_2166_2171/install.log 2>&1 &
echo "PID $!"
sleep 1
tail -20 /root/dlc_infer_2166_2171/install.log || true
ps aux | grep -E 'pip|conda|install_dlc' | grep -v grep | head
