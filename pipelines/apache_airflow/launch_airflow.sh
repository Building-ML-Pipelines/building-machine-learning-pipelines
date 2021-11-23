#!/bin/sh
tmux new-session -d -s 'airflow_launch' -n 'htop' htop
tmux new-window 'jupyter notebook --ip=0.0.0.0 --port=8081 --allow-root'
tmux split-window -v 'airflow webserver -p 7070'
tmux split-window -h 'airflow scheduler'
tmux -2 attach-session -d
