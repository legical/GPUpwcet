#!/bin/bash

source ~/.bashrc
# 获取当前目录和主目录（上一级目录）
proj_dir=$(pwd)
root_dir=$proj_dir/..

# 编译 CUDA 程序
nvcc vectoradd.cu -o vectoradd

echo "1. 执行single测试..."
start_time=$(date +%s.%N)
./vectoradd
end_time=$(date +%s.%N)
execution_time_a=$(echo "$end_time - $start_time" | bc)
echo "single测试时间：$execution_time_a 秒"

# 执行未启用 MPS 的测试
echo "2. 执行未启用 MPS 的测试..."
start_time=$(date +%s.%N)
./vectoradd & ./vectoradd
end_time=$(date +%s.%N)
execution_time_a=$(echo "$end_time - $start_time" | bc)
echo "未启用 MPS 的测试执行时间：$execution_time_a 秒"

# 启用 MPS
chmod +x $root_dir/scripts/mps/*.sh
echo ""
sudo $root_dir/scripts/mps/enable-mps.sh

# 执行已启用 MPS 的测试
cd $proj_dir
echo ""
echo "3. 执行已启用 MPS 的测试..."
start_time=$(date +%s.%N)
./vectoradd & ./vectoradd
end_time=$(date +%s.%N)
execution_time_b=$(echo "$end_time - $start_time" | bc)
echo "已启用 MPS 的测试执行时间：$execution_time_b 秒"

# 比较执行时间
echo ""
time_difference=$(echo "$execution_time_a - $execution_time_b" | bc)
if (($(echo "$time_difference > 1" | bc -l))); then
    echo -e "===== \033[32m测试结果：当前 GPU 支持 MPS\033[0m ====="
else
    echo -e "===== \033[34m测试结果：当前 GPU 未启用或不支持 MPS\033[0m ====="
fi

# 关闭 MPS
echo ""
sudo $root_dir/scripts/mps/disable-mps.sh
