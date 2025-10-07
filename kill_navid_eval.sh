ps aux | grep 'run.py' | awk '{print $2}' | xargs kill
