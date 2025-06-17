ps -ef | grep qwen_filter_edit_obj_removal.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep run_removal.sh | grep -v grep | awk '{print $2}' | xargs kill -9