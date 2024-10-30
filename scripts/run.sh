#!/bin/bash

task_name=qhdetr_sapm1  # 请替换为你的任务名称

find_unused_port() {
    while true; do
        port=$(shuf -i 10000-60000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo "$port"
            return 0
        fi
    done
}
UNUSED_PORT=$(find_unused_port)

task_id=$UNUSED_PORT

cd ..

nohup \
bash train.sh \
> ./scripts/${task_name}.log 2>&1 &

sleep 10
ps aux | grep 'main.py' | grep -v 'grep' | awk '{print $2}' > ./scripts/${task_name}.pid


# cat ./qhdetr_sapm1.pid | xargs kill