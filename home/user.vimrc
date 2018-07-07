" tab step
set tabstop=4

" display line nubmers
set nu

" search options: highlight search, incremental search, case insensitive search, and smart search
set hlsearch
set incsearch
set ignorecase
set smartcase

" enable syntax highlighting, set 256 color and colorscheme
syntax on
set t_Co:256
colorscheme desert
hi Constant ctermfg=darkgreen
hi Identifier ctermfg=red
hi LineNr ctermfg=darkyellow

" enable showing a '|' for each tab
set list
set listchars=tab:\|\ 

" increase / decrease vertical windows size by '+'/'-'
nmap + :vertical resize +10<CR>
nmap - :vertical resize -10<CR>

" turn on/off diff mode
map <expr> <F6> &diff ? ':diffoff!<RETURN>' : ':windo difft<RETURN><C-W>='
imap <F6> <ESC><F6>
