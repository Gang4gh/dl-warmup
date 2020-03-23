" display line nubmers
set nu
set backspace=2

" tab step and overwrite tabstop=4 for python files
set tabstop=4
augroup tabstopFour
    autocmd!
    autocmd FileType python set tabstop=4
augroup END

" search options: highlight search, incremental search, case insensitive search, and smart search
set hlsearch
set incsearch
set smartcase

" enable syntax highlighting, set 256 color and colorscheme
syntax on
set t_Co:256
colorscheme desert
hi Constant ctermfg=darkgreen
hi Identifier ctermfg=lightgreen
hi LineNr ctermfg=darkyellow

" enable showing a '|' for each tab
set list
set listchars=tab:\|\ 

" make vimdiff looks better
highlight DiffAdd    cterm=bold ctermfg=10 ctermbg=17 gui=none guifg=bg guibg=Red
highlight DiffDelete cterm=bold ctermfg=10 ctermbg=17 gui=none guifg=bg guibg=Red
highlight DiffChange cterm=bold ctermfg=10 ctermbg=17 gui=none guifg=bg guibg=Red
highlight DiffText   cterm=bold ctermfg=10 ctermbg=88 gui=none guifg=bg guibg=Red

" increase / decrease vertical windows size by '+'/'-'
nmap + :vertical resize +10<CR>
nmap - :vertical resize -10<CR>

" turn on/off diff mode, and go to next/previous diff
map <expr> <F6> &diff ? ':diffoff!<CR>' : ':windo difft<CR><C-W>='
imap <F6> <ESC><F6>
map <F9> [c
map <F10> ]c

" fold code by indent
set foldmethod=indent
set foldnestmax=1
set nofoldenable
set foldlevel=2

" save current file by <Ctrl-S>
map <C-S> :w<CR>
imap <C-S> <ESC><C-S>
" save and quite by <Ctrl-Q>
map <C-Q> :x<CR>
imap <C-Q> <ESC><C-Q>

" vim-plugins
call plug#begin('~/.vim/plugged')
" vim statusline
Plug 'itchyny/lightline.vim'
set noshowmode
set laststatus=2

" file explorer
Plug 'scrooloose/nerdtree', { 'on':  'NERDTreeToggle' }
map <C-O> :NERDTreeToggle<CR>
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif

" comment code
Plug 'preservim/nerdcommenter'

" git support
Plug 'tpope/vim-fugitive'

" toggle location/quickfix lists
Plug 'Valloric/ListToggle'
map <C-L> :LToggle<CR>
map <C-J> :lnext<CR>
map <C-K> :lprevious<CR>

" syntax checker and auto complete
Plug 'vim-syntastic/syntastic'
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*
let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0
let g:syntastic_loc_list_height = 5
let g:syntastic_python_checkers = ['flake8']
let g:syntastic_quiet_messages = { 'regex': ['\[E111\]', '\[E114\]', '\[E265\]', '\[E501\]'] }
" ignored error list
" E111: indentation is not a multiple of four
" E114: indentation is not a multiple of four
" E265: block comment should start with '# '
" E501: line too long

call plug#end()
