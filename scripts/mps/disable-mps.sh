#!/bin/bash

# 检查当前用户是否为root用户
if [[ $EUID -ne 0 ]]; then
    echo "请使用sudo或以root用户身份执行此脚本。"
    exit 1
fi

echo -e "***** \033[34mDisable MPS\033[0m *****"
echo quit | nvidia-cuda-mps-control
nvidia-smi -i 0 -c DEFAULT

# 检查MPS是否已禁用
ps -ef | grep nvidia-cuda-mps-control 
count=$(ps -ef | grep mps | wc -l)

# 如果行数等于1，则说明MPS成功禁用
if [ $count -eq 1 ]; then
    echo "NVIDIA MPS已成功禁用。"
else
    echo "禁用NVIDIA MPS时出现问题，请检查相关配置。"
fi