CPUS ?= $(shell nproc)
#MAKEFLAGS += -j24	# doesn't work at make 4.1
TAG = $(shell date +'%Y%m%d%H%M%S')

all:
	$(MAKE) -j$(CPUS) data

# preprocess gigaword training data
GIGA_DIR=./Gigaword/
GIGA_RAW=$(GIGA_DIR)LDC2012T21/Data/data/xml/
GIGA_ARTICLES=$(GIGA_DIR)articles/

GIGA_GZ_FILES=$(wildcard $(GIGA_RAW)*.xml.gz)
GIGA_ARTICLES_FILES=$(patsubst $(GIGA_RAW)%.xml.gz,$(GIGA_ARTICLES)%.articles,$(GIGA_GZ_FILES))

$(GIGA_ARTICLES)%.articles : $(GIGA_RAW)%.xml.gz ./convert_gigaword.py
	python2.7 $(word 2,$^) $< $@

# this target is defined to preserve *.articles files
gigaword-articles: $(GIGA_ARTICLES_FILES)

%.articles.f: %.articles ./process_gigaword.py
	python3 $(word 2,$^) filter $< 2 120 > $@

%.sumdata: %.splits $(subst .articles,.articles.f,$(GIGA_ARTICLES_FILES))
	cat $< | xargs -I % cat $(GIGA_ARTICLES)%.f > $@

%.vocab: %.sumdata ./process_gigaword.py
	python3 $(word 2,$^) vocab $< > $@

data: $(GIGA_DIR).gigaword-data

$(GIGA_DIR).gigaword-data: $(GIGA_DIR)train.sumdata $(GIGA_DIR)strain.sumdata $(GIGA_DIR)valid.sumdata $(GIGA_DIR)test.sumdata $(GIGA_DIR)train.vocab $(GIGA_DIR)strain.vocab
	mkdir $(GIGA_DIR)data-$(TAG)/
	cp $(GIGA_DIR)*.sumdata $(GIGA_DIR)*.vocab $(GIGA_DIR)data-$(TAG)/
	touch $@

check-tag:
	@echo current tag is \'$(TAG)\'

small-train:
	python3 ./seq2seq_attention.py \
		--mode=train \
		--data_path=./Gigaword/data-0323/strain.data \
		--vocab_path=./Gigaword/data-0323/strain.vocab \
		--log_root=./Gigaword/log-giga0323 --train_dir=./Gigaword/log-giga0323/train

train:
	python3 ./seq2seq_attention.py \
		--mode=train \
		--data_path=$(GIGA_DIR)train.sumdata \
		--vocab_path=$(GIGA_DIR)train.vocab \
		--log_root=$(GIGA_DIR)log-$(TAG) --train_dir=$(GIGA_DIR)log-$(TAG)/train

eval:
	python3 ./seq2seq_attention.py \
		--mode=eval \
		--data_path=$(GIGA_DIR)test.10.sumdata \
		--vocab_path=$(GIGA_DIR)train.vocab \
		--log_root=$(GIGA_DIR)summodel --eval_dir=$(GIGA_DIR)summodel/eval

decode:
	python3 ./seq2seq_attention.py \
		--mode=decode \
		--data_path=$(GIGA_DIR)test.1k.sumdata \
		--vocab_path=$(GIGA_DIR)train.vocab \
		--log_root=$(GIGA_DIR)summodel0329 --decode_dir=$(GIGA_DIR)summodel0329/decode \
		--beam_size=8
