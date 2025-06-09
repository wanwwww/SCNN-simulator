#!/bin/bash

# 检查是否提供了配置文件参数
if [ $# -eq 0 ]; then
    echo "错误：请提供配置文件路径作为参数"
    echo "用法：$0 <配置文件路径>"
    exit 1
fi

CONFIG_FILE="$1"
OUTPUT_DIR="${2:-./output_script}"  # 第二个参数；如果没提供则默认值为 ./output_script

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误：配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 检查输出目录是否存在，不存在则创建
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "创建输出目录：$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

for topo in ./topo_files/*.csv
do
    name=$(basename "$topo" .csv)
    echo "使用配置 $CONFIG_FILE 运行 $name..."
    ./stonne "$topo" "$CONFIG_FILE" > "${OUTPUT_DIR}/${name}_output.txt" &
done

wait

echo "所有拓扑测试已完成！使用配置：$CONFIG_FILE"