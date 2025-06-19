ps -ef | grep qwen_filter_edit_obj_addition.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep run_filter.sh | grep -v grep | awk '{print $2}' | xargs kill -9