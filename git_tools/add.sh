if $# != 2; then
	echo "missing arg1_file arg2_message"
fi
git pull
git add $1
git commit -m $2
git push
