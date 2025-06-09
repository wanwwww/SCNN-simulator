#!/bin/bash

# 并行执行多个 script.sh 命令
# ./script.sh ./cfg_files/tpu_16_CHW.cfg ./output_CHW_16 &
# ./script.sh ./cfg_files/tpu_32_CHW.cfg ./output_CHW_32 &
# ./script.sh ./cfg_files/tpu_64_CHW.cfg ./output_CHW_64 &
# ./script.sh ./cfg_files/tpu_128_CHW.cfg ./output_CHW_128 &

# ./script.sh ./cfg_files/tpu_16_HWC.cfg ./output_HWC_16 &
# ./script.sh ./cfg_files/tpu_32_HWC.cfg ./output_HWC_32 &
# ./script.sh ./cfg_files/tpu_64_HWC.cfg ./output_HWC_64 &
# ./script.sh ./cfg_files/tpu_128_HWC.cfg ./output_HWC_128 &

./script.sh ./cfg_files/tpu_16_HCW.cfg ./output_HCW_16 &
./script.sh ./cfg_files/tpu_32_HCW.cfg ./output_HCW_32 &
./script.sh ./cfg_files/tpu_64_HCW.cfg ./output_HCW_64 &
./script.sh ./cfg_files/tpu_128_HCW.cfg ./output_HCW_128 &

# 等待所有后台任务完成
wait

echo "所有 script.sh 命令已完成执行！"