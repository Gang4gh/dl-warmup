#!/bin/bash

CONFIG_FILES=".profile .bash_aliases .vimrc .screenrc .gitconfig"

echo Setup hard links for user files in $HOME
echo CONFIG_FILES=$CONFIG_FILES

if [ "$1" == "setup" ] && [ "$2" == "$HOME" ]; then
	for fi in $CONFIG_FILES; do
		cp -fl user$fi ~/$fi
	done
	echo done.
else
	echo Invalid arguments.
	echo Usage: $0 setup $HOME
fi

