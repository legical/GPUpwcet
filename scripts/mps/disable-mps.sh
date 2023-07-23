#!/bin/bash
echo -e "*****\033[34mDisable MPS\033[0m *****"
echo quit | nvidia-cuda-mps-control
nvidia-smi -i 0 -c DEFAULT
