#!/bin/bash

CONFIG_FILES=".bash_aliases .bashrc .gitconfig .profile .screenrc .vimrc"

echo Setup symlink links for user files in $HOME
echo CONFIG_FILES=$CONFIG_FILES

if [ "$1" == "setup" ] && [ "$2" == "$HOME" ]; then
	for fi in $CONFIG_FILES; do
		ln -srf user$fi ~/$fi
	done
	echo done.
else
	echo Invalid arguments.
	echo Usage: $0 setup $HOME
fi
