CPUS ?= $(shell nproc)
#MAKEFLAGS += -j24	# doesn't work with make v4.1
TAG = $(shell date +'%Y%m%d%H%M%S')

# preprocess gigaword training data
GIGA_DIR=Gigaword/
GIGA_RAW_DIR=$(GIGA_DIR)LDC2012T21/Data/data/xml/


all:
	$(MAKE) -j$(CPUS) data


clean:
	rm -f $(GIGA_RAW_DIR)*.articles $(GIGA_RAW_DIR)*.articles.f
	rm -f $(GIGA_DIR)*.sumdata $(GIGA_DIR)*.vocab


GIGA_GZ_FILES=$(wildcard $(GIGA_RAW_DIR)*.xml.gz)
GIGA_ARTICLES_FILES=$(subst .xml.gz,.articles,$(GIGA_GZ_FILES))

$(GIGA_RAW_DIR)%.articles : $(GIGA_RAW_DIR)%.xml.gz data_gigaword.py
	python3 $(word 2,$^) $< > $@

# this target is defined to preserve *.articles files
gigaword-articles: $(GIGA_ARTICLES_FILES)

%.articles.f: %.articles data_generic.py
	python3 $(word 2,$^) filter $< 2 120 30 > $@

%.sumdata: %.splits $(subst .articles,.articles.f,$(GIGA_ARTICLES_FILES))
	cat $< | xargs -I % cat $(GIGA_RAW_DIR)%.f > $@

%.vocab: %.sumdata data_generic.py
	python3 $(word 2,$^) vocab $< > $@

data: $(GIGA_DIR).gigaword-data

$(GIGA_DIR).gigaword-data: $(GIGA_DIR)train.sumdata $(GIGA_DIR)strain.sumdata $(GIGA_DIR)valid.sumdata $(GIGA_DIR)test.sumdata $(GIGA_DIR)train.vocab $(GIGA_DIR)strain.vocab
	mkdir -p $(GIGA_DIR)data-$(TAG)
	shuf -n 1000 $(GIGA_DIR)test.sumdata --random-source=$(GIGA_DIR)test.sumdata > $(GIGA_DIR)test.1k.sumdata
	shuf -n 8000 $(GIGA_DIR)test.sumdata --random-source=$(GIGA_DIR)test.sumdata > $(GIGA_DIR)test.8k.sumdata
	cp $(GIGA_DIR)*.sumdata $(GIGA_DIR)*.vocab $(GIGA_DIR)*.splits data_*.py $(GIGA_DIR)data-$(TAG)
	touch $@

TRAINING_ROOT=training_center/run-$(TAG)/

train: data
	cp *.py Makefile $(GIGA_DIR)*.sumdata $(GIGA_DIR)*.vocab $(TRAINING_ROOT)
	(killall -w tensorboard || echo) 2>&1 >/dev/null
	tensorboard --port 2345 --logdir=$(TRAINING_ROOT)model &
	python3 $(TRAINING_ROOT)seq2seq_attention.py \
		--mode=train \
		--data_path=$(TRAINING_ROOT)train.sumdata \
		--vocab_path=$(TRAINING_ROOT)train.vocab \
		--log_root=$(TRAINING_ROOT)model --train_dir=$(TRAINING_ROOT)model/train

decode:
	python3 ./seq2seq_attention.py \
		--mode=decode \
		--data_path=training_center/sa16-0330/test.1k.sumdata \
		--vocab_path=training_center/sa16-0330/train.vocab \
		--log_root=$(GIGA_DIR)log-20180330133216 --decode_dir=$(GIGA_DIR)log-20180330133216/decode \
		--beam_size=8
