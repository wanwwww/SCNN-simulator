#!/bin/bash

# 并行执行多个 script.sh 命令

./script_bank.sh ./cfg_files/tpu_16_HCW_bank.cfg ./output_HCW_bank_16 &
./script_bank.sh ./cfg_files/tpu_32_HCW_bank.cfg ./output_HCW_bank_32 &
./script_bank.sh ./cfg_files/tpu_64_HCW_bank.cfg ./output_HCW_bank_64 &
./script_bank.sh ./cfg_files/tpu_128_HCW_bank.cfg ./output_HCW_bank_128 &

# 等待所有后台任务完成
wait

echo "所有 script.sh 命令已完成执行！"