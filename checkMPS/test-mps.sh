#!/bin/bash

source ~/.bashrc
# 获取当前目录和主目录（上一级目录）
proj_dir=$(pwd)
root_dir=$proj_dir/..

source $root_dir/scripts/tool.sh

check_GPU_Driver
echo "--------------------- 测试 GPU$GPU_name 是否真的支持MPS ---------------------"
echo ""

# 编译 CUDA 程序
nvcc vectoradd.cu -o vectoradd

echo "1. 基准测试：执行单个程序..."
start_time=$(date +%s.%N)
./vectoradd
end_time=$(date +%s.%N)
basetime=$(echo "$end_time - $start_time" | bc)
echo "基准测试时间：$basetime s"

# 执行未启用 MPS 的测试
echo "2. 执行未启用 MPS 的测试..."
start_time=$(date +%s.%N)
./vectoradd & ./vectoradd
end_time=$(date +%s.%N)
execution_time_a=$(echo "$end_time - $start_time" | bc)
echo "未启用 MPS 的测试执行时间：$execution_time_a s"

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
echo "已启用 MPS 的测试执行时间：$execution_time_b s"

# 清理编译文件
rm -rf vectoradd

# 比较执行时间
echo ""
time_difference=$(echo "$execution_time_a - $execution_time_b" | bc)
if (($(echo "$time_difference > 2" | bc -l))); then
    echo -e "===== \033[32m测试结果：GPU$GPU_name 支持 MPS\033[0m ====="
else
    echo -e "===== \033[34m测试结果：GPU$GPU_name 未启用或不支持 MPS\033[0m ====="
fi

# 关闭 MPS
echo ""
sudo $root_dir/scripts/mps/disable-mps.sh
