#!/bin/bash

# Source progress bar
source ./tool.sh

generate_some_output_and_sleep() {
    echo "Here is some output"
    head -c 500 /dev/urandom | tr -dc 'a-zA-Z0-9~!@#$%^&*_-'
    head -c 500 /dev/urandom | tr -dc 'a-zA-Z0-9~!@#$%^&*_-'
    head -c 500 /dev/urandom | tr -dc 'a-zA-Z0-9~!@#$%^&*_-'
    head -c 500 /dev/urandom | tr -dc 'a-zA-Z0-9~!@#$%^&*_-'
    echo -e "\n\n------------------------------------------------------------------"
    echo -e "\n\n Now sleeping briefly \n\n"
    sleep 0.3
}


main() {
    for i in {1..99}
    do
        generate_some_output_and_sleep
        progress_bar $i 100
    done
}

main