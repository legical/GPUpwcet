#!/bin/bash
echo -e "*****\033[32mEnable MPS\033[0m *****"
export CUDA_VISIBLE_DEVICES="0"
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
