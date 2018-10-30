#!/bin/bash
if (($# != 2)); then
	echo "missing [arg1_file arg2_message_no_spaces!], this will ADD the items"
else
	echo "adding to git origin"
	git pull
	git add $1
	git commit -m $2
	git push
fi
