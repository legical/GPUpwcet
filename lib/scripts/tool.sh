#!/bin/bash

COLOR_FG="\e[30m"
COLOR_BG="\e[42m"
RESTORE_FG="\e[39m"
RESTORE_BG="\e[49m"

trap 'onCtrlC' INT
function onCtrlC() {
    #捕获CTRL+C，当脚本被ctrl+c的形式终止时同时终止程序的后台进程
    kill -9 ${do_sth_pid} ${progress_pid}
    echo
    echo 'Ctrl+C 中断进程'
    exit 1
}

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
    else
        GPU_name=$(echo $GPU_name_line | awk -F ' ' '{print $NF}')
    fi
}

# 检查 build 目录是否存在, recreate build 目录
function recreate_build() {
    # 检查 build 目录是否存在
    PROJ_DIR=$1
    if [ -d $PROJ_DIR/build ]; then
        rm -r $PROJ_DIR/build
    fi
    mkdir -m 754 $PROJ_DIR/build
    # echo -e "\033[34mbuild directory successfully created.\033[0m"
}

# parameter: line numbers want to delete
function del_line() {
    # if do not input parameter
    if [ -z $1 ]; then
        del_num=1
    else
        del_num=$1
    fi
    for ((j = 0; j < del_num; j++)); do
        # \33[2K 删除光标当前所在的整行
        # \033[A将光标上移一行，但移至同一列，即不移至该行的开头
        # \r将光标移至行首(r用于快退)，但不删除任何内容
        printf "\r \33[2K \033[A"
    done
}

function del_this_line() {
    printf "\33[2K"
}
# 打印一个重复 $1字符串 $2次 的新字符串
# printf_new '-' 10 输出一个长度为10的-字符字符串
function printf_new() {
    str=$1
    num=$2
    v=$(printf "%-${num}s" "$str")
    echo -ne "${v// /$str}"
}

# 命令行进度条，该函数接受3个参数，1-进度的整型数值，2-是总数的整型数值, 3-out running times
function progress_bar {
    if [ -z $1 ] || [ -z $2 ]; then
        echo "[E]>>>Missing parameter. \$1=$1, \$2=$2"
        return
    fi

    pro=$1                     # 进度的整型数值
    total=$2                   # 总数的整型数值
    if [ $pro -gt $total ]; then
        echo "[E]>>>It's impossible that 'pro($pro) > total($total)'."
        return
    fi

    # GREEN_SHAN='\E[5;32;49;1m' # 亮绿色闪动
    # RES='\E[0m'                # 清除颜色
    local cols=$(tput cols)    # 获取列长度
    local color="${COLOR_FG}${COLOR_BG}"
    # arr=('|' '/' '-' '\\')
    # let index=pro%4
    if [ $cols -le 30 ]; then
        printf "Testing... No.${1}/${2}"
        return
    fi

    percentage=$(($pro * 100 / $total))
    if [ -z $3 ] || [ -z $4 ]; then
        PRE_STR=$(echo -ne "Test Process:${percentage}")
    else
        PRE_STR=$(echo -ne "Test ${3}/${4} Process:${percentage}")
    fi

    let bar_size=$cols-27
    let complete_size=$(($pro * $bar_size / $total))
    let remainder_size=$bar_size-$complete_size

    progress_bar=$(echo -ne "%% ["; echo -en "${color}"; printf_new "#" $complete_size; echo -en "${RESTORE_FG}${RESTORE_BG}"; printf_new "." $remainder_size; echo -ne "]");
    #echo "pro=$pro, percent=$percent"

    # Print progress bar
    printf "${PRE_STR}${progress_bar}\r"
    # printf "\33[2K"

    if [ $pro -eq $total ]; then
        echo
    fi
}
# ---下面是测试代码---
# 总数为200
# 进度从1增加到200
# 循环调用ProgressBar函数，直到进度增加到200结束
# Tot=200
# for i in `seq 1 $Tot`;do
#     progress_bar $i $Tot
#     sleep 0.1
# done

# 判断是否安装了 python 模块
function python_model_check() {
    if python3 -c "import $1" >/dev/null 2>&1; then
        echo "$1 has been installed."
    else
        echo -e "\033[31mInstalling $1.\033[0m"
        python3 -m pip install -U $1
    fi
}
# ---下面是测试代码---
# 判断是否安装了 numpy matplotlib 模块
# python_model_check numpy
# python_model_check matplotlib


# https://github.com/pollev/bash_progress_bar - See license at end of file

# Usage:
# Source this script
# enable_trapping <- optional to clean up properly if user presses ctrl-c
# setup_scroll_area <- create empty progress bar
# draw_progress_bar 10 <- advance progress bar
# draw_progress_bar 40 <- advance progress bar
# block_progress_bar 45 <- turns the progress bar yellow to indicate some action is requested from the user
# draw_progress_bar 90 <- advance progress bar
# destroy_scroll_area <- remove progress bar

# Constants
CODE_SAVE_CURSOR="\033[s"
CODE_RESTORE_CURSOR="\033[u"
CODE_CURSOR_IN_SCROLL_AREA="\033[1A"
COLOR_BG_BLOCKED="\e[43m"

# Variables
PROGRESS_BLOCKED="false"
TRAPPING_ENABLED="false"
TRAP_SET="false"

CURRENT_NR_LINES=0

setup_scroll_area() {
    # If trapping is enabled, we will want to activate it whenever we setup the scroll area and remove it when we break the scroll area
    if [ "$TRAPPING_ENABLED" = "true" ]; then
        trap_on_interrupt
    fi

    lines=$(tput lines)
    CURRENT_NR_LINES=$lines
    let lines=$lines-1
    # Scroll down a bit to avoid visual glitch when the screen area shrinks by one row
    echo -en "\n"

    # Save cursor
    echo -en "$CODE_SAVE_CURSOR"
    # Set scroll region (this will place the cursor in the top left)
    echo -en "\033[0;${lines}r"

    # Restore cursor but ensure its inside the scrolling area
    echo -en "$CODE_RESTORE_CURSOR"
    echo -en "$CODE_CURSOR_IN_SCROLL_AREA"

    # Start empty progress bar
    draw_progress_bar 0
}

destroy_scroll_area() {
    lines=$(tput lines)
    # Save cursor
    echo -en "$CODE_SAVE_CURSOR"
    # Set scroll region (this will place the cursor in the top left)
    echo -en "\033[0;${lines}r"

    # Restore cursor but ensure its inside the scrolling area
    echo -en "$CODE_RESTORE_CURSOR"
    echo -en "$CODE_CURSOR_IN_SCROLL_AREA"

    # We are done so clear the scroll bar
    clear_progress_bar

    # Scroll down a bit to avoid visual glitch when the screen area grows by one row
    echo -en "\n\n"

    # Once the scroll area is cleared, we want to remove any trap previously set. Otherwise, ctrl+c will exit our shell
    if [ "$TRAP_SET" = "true" ]; then
        trap - INT
    fi
}

draw_progress_bar() {
    percentage=$1
    lines=$(tput lines)
    let lines=$lines

    # Check if the window has been resized. If so, reset the scroll area
    if [ "$lines" -ne "$CURRENT_NR_LINES" ]; then
        setup_scroll_area
    fi

    # Save cursor
    echo -en "$CODE_SAVE_CURSOR"

    # Move cursor position to last row
    echo -en "\033[${lines};0f"

    # Clear progress bar
    tput el

    # Draw progress bar
    PROGRESS_BLOCKED="false"
    print_bar_text $percentage

    # Restore cursor position
    echo -en "$CODE_RESTORE_CURSOR"
}

block_progress_bar() {
    pro=$1                     # 进度的整型数值
    total=$2                   # 总数的整型数值
    if [ $pro -gt $total ]; then
        echo "[E]>>>It's impossible that 'pro($pro) > total($total)'."
        return
    fi
    # percentage=$1
    percentage=$(($pro * 100 / $total))
    lines=$(tput lines)
    let lines=$lines
    # Save cursor
    echo -en "$CODE_SAVE_CURSOR"

    # Move cursor position to last row
    echo -en "\033[${lines};0f"

    # Clear progress bar
    tput el

    # Draw progress bar
    PROGRESS_BLOCKED="true"
    print_bar_text $percentage

    # Restore cursor position
    echo -en "$CODE_RESTORE_CURSOR"
}

clear_progress_bar() {
    lines=$(tput lines)
    let lines=$lines
    # Save cursor
    echo -en "$CODE_SAVE_CURSOR"

    # Move cursor position to last row
    echo -en "\033[${lines};0f"

    # clear progress bar
    tput el

    # Restore cursor position
    echo -en "$CODE_RESTORE_CURSOR"
}

print_bar_text() {
    local percentage=$1
    local cols=$(tput cols)
    let bar_size=$cols-17

    local color="${COLOR_FG}${COLOR_BG}"
    if [ "$PROGRESS_BLOCKED" = "true" ]; then
        color="${COLOR_FG}${COLOR_BG_BLOCKED}"
    fi

    # Prepare progress bar
    let complete_size=($bar_size*$percentage)/100
    let remainder_size=$bar_size-$complete_size
    progress_bar=$(echo -ne "["; echo -en "${color}"; printf_new "#" $complete_size; echo -en "${RESTORE_FG}${RESTORE_BG}"; printf_new "." $remainder_size; echo -ne "]");

    # Print progress bar
    echo -ne " Progress ${percentage}% ${progress_bar}"
}

enable_trapping() {
    TRAPPING_ENABLED="true"
}

trap_on_interrupt() {
    # If this function is called, we setup an interrupt handler to cleanup the progress bar
    TRAP_SET="true"
    trap cleanup_on_interrupt INT
}

cleanup_on_interrupt() {
    destroy_scroll_area
    exit
}

printf_new() {
    str=$1
    num=$2
    v=$(printf "%-${num}s" "$str")
    echo -ne "${v// /$str}"
}


# SPDX-License-Identifier: MIT
#
# Copyright (c) 2018--2020 Polle Vanhoof
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.