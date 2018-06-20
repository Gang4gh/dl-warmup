set tabstop=4
set nu

set hlsearch
set incsearch

set ignorecase
set smartcase

syntax on
colorscheme desert

set list
set listchars=tab:\|\ 

" increase / decrease vertical windows size by '+'/'-'
nmap + :vertical resize +10<CR>
nmap - :vertical resize -10<CR>

" turn on/off diff mode
map <expr> <F6> &diff ? ':diffoff!<RETURN>' : ':windo difft<RETURN><C-W>='
imap <F6> <ESC><F6>

