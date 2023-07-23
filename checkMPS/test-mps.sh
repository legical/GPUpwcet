#!/bin/bash

# 获取当前目录和主目录（上一级目录）
proj_dir=$(pwd)
root_dir=$(dirname "$current_dir")

# 编译 CUDA 程序
nvcc vectoradd.cu -o vectoradd -lpthread

# 执行未启用 MPS 的测试
echo "执行未启用 MPS 的测试..."
start_time=$(date +%s.%N)
./vectoradd && ./vectoradd
end_time=$(date +%s.%N)
execution_time_a=$(echo "$end_time - $start_time" | bc)
echo "未启用 MPS 的测试执行时间：$execution_time_a 秒"

# 启用 MPS
sudo $root_dir/scripts/mps/enable-mps.sh

# 执行已启用 MPS 的测试
cd $proj_dir
echo "执行已启用 MPS 的测试..."
start_time=$(date +%s.%N)
./vectoradd && ./vectoradd
end_time=$(date +%s.%N)
execution_time_b=$(echo "$end_time - $start_time" | bc)
echo "已启用 MPS 的测试执行时间：$execution_time_b 秒"

# 比较执行时间
time_difference=$(echo "$execution_time_a - $execution_time_b" | bc)
if (( $(echo "$time_difference > 0.5" | bc -l) )); then
  echo "当前 GPU 支持 MPS"
else
  echo "当前 GPU 未启用或不支持 MPS"
fi

# 关闭 MPS
sudo $root_dir/scripts/mps/disable-mps.sh
