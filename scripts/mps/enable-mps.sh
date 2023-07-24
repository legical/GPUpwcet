#!/bin/bash

# 检查当前用户是否为root用户
if [[ $EUID -ne 0 ]]; then
    echo "请使用sudo或以root用户身份执行此脚本。"
    exit 1
fi

echo -e "***** \033[32mEnable MPS\033[0m *****"
export CUDA_VISIBLE_DEVICES="0"
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d

# 检查MPS是否已启用
ps -ef | grep mps 
count=$(ps -ef | grep nvidia-cuda-mps-control | wc -l)

# 如果行数大于等于2，则说明MPS成功启动
if [ $count -ge 2 ]; then
    echo "NVIDIA MPS已成功启用。"
else
    echo "启用NVIDIA MPS时出现问题，请检查相关配置。"
fi