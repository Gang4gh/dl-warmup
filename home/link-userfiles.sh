#!/bin/bash

CONFIG_FILES=".bash_aliases .bashrc .gitconfig .profile .screenrc .vimrc"

echo Setup symlink links for user files in $HOME
echo CONFIG_FILES=$CONFIG_FILES

if [ "$1" == "setup" ] && [ "$2" == "$HOME" ]; then
	for fi in $CONFIG_FILES; do
		ln -srf user$fi ~/$fi
	done
	curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
	mkdir -p ~/.vim/after/syntax/
	cat > ~/.vim/after/syntax/python.vim << EOL
" Highlight docstrings as comments, not string, for Python
syn region pythonDocstring  start=+^\s*[uU]\?[rR]\?"""+ end=+"""+ keepend excludenl contains=pythonEscape,@Spell,pythonDoctest,pythonDocTest2,pythonSpaceError
syn region pythonDocstring  start=+^\s*[uU]\?[rR]\?'''+ end=+'''+ keepend excludenl contains=pythonEscape,@Spell,pythonDoctest,pythonDocTest2,pythonSpaceError
hi def link pythonDocstring pythonComment
EOL
	echo done.
else
	echo Invalid arguments.
	echo Usage: $0 setup $HOME
fi
