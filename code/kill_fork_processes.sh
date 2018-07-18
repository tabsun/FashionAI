for pid in $(ps -ef |grep python|grep -v grep|awk '{print $2}'); do kill -9 $pid; done
