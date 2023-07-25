#!/bin/bash
# 是否继续执行脚本
function isGoon() {
    read -p "Do you want to continue? [y/n] " input

    case $input in
    [yY]*) ;;

    [nN]*)
        exit
        ;;
    *)
        echo -e "\033[31mJust enter y or n, please.\033[0m\n"
        exit
        ;;
    esac
}

function check_GPU_Driver {
    # 3060 or 1070
    GPU_name_line=$(nvidia-smi -q | grep "Product Name")
    if [ $? -ne 0 ]; then
        # nvidia-smi 无输出，证明未安装NVIDIA Driver
        echo -e "***** \033[34mPlease install NVIDIA Driver first !!!\033[0m *****"
        exit 1
    else
        GPU_name=$(echo $GPU_name_line | awk -F ' ' '{print $NF}')
    fi
}