ps -ef | grep preprocess_senorita.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep inference_vie_score_filter.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep run.py | grep -v grep | awk '{print $2}' | xargs kill -9