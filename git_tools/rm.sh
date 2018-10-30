if $# != 2; then
	echo "missing arg1_file arg2_message"
fi
FILE = $1
MESSAGE = $2
git pull
git rm $1
git commit -m $2
git push
