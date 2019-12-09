#/bin/bash
# 設定bash環境

PYTHON=python3
# 設定變數PYTHON為指令python3
#SERVEO="ssh -M 0 -f -T -N -R aitree.serveo.net:80:localhost:8000 serveo.net"
SERVEO="ssh -R  aitree:80:localhost:8000 serveo.net"
# 設定變數SERVEO的指令
RUNAPP="$PYTHON tree_recog_1104_chatbot.py"
# 設定變數RUNAPP的指令

if [ $# -eq 1 ]
# 如果參數長度等於1
then
	if [ $1 == "serveo" ]
	# 如果第1個參數等於"serveo"
	then
		echo "SERVEO init ..."
		$SERVEO
		echo "SERVEO ready."
	elif [ $1 == "start" ]
	then
		echo "SERVEO init ..."
		$SERVEO
		echo "SERVEO ready."
		echo "sleep for 2 secs."
		sleep 2
		echo "RUNAPP init ..."
		$RUNAPP &
		echo "RUNAPP ready."
	elif [ $1 == "runapp" ]
	then
		echo "RUNAPP init ..."
		$RUNAPP &
		echo "RUNAPP ready."
	fi
elif [ $# -eq 0 ]
then
	echo "APP running ..."
	$RUNAPP &
	echo "APP ready."
	echo "sleep for 2 secs."
	sleep 2
	echo "SERVEO init ..."
	$SERVEO
	echo "SERVEO ready."
fi
