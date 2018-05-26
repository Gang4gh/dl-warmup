CPUS ?= $(shell nproc)
#MAKEFLAGS += -j24	# doesn't work with make v4.1
TAG ?= $(shell date +'%Y%m%d%H%M%S')


# preprocess gigaword training data
GIGA_DATA_DIR=/home/gang/LDC2012T21/Data/data/xml/
GIGA_DIR=Gigaword/
GIGA_TMP_DIR=$(GIGA_DIR)articles/


all:
	$(MAKE) -j$(CPUS) data


clean:
	rm -rf $(GIGA_TMP_DIR)
	rm -f $(GIGA_DIR)*.sumdata $(GIGA_DIR)*.vocab

check:
	@echo TAG is '$(TAG)'


# prepare data used for training
GIGA_GZ_FILES=$(wildcard $(GIGA_DATA_DIR)*.xml.gz)
GIGA_ARTICLES_FILES=$(GIGA_GZ_FILES:$(GIGA_DATA_DIR)%.xml.gz=$(GIGA_TMP_DIR)%.articles)

$(GIGA_TMP_DIR)%.articles : $(GIGA_DATA_DIR)%.xml.gz data_gigaword.py
	@mkdir -p $(GIGA_TMP_DIR)
	python3 $(word 2,$^) $< > $@

# this target is defined to preserve *.articles files
gigaword-articles: $(GIGA_ARTICLES_FILES)

%.articles.f: %.articles data_generic.py
	python3 $(word 2,$^) filter $< 2 120 30 > $@

%.sumdata: %.splits $(subst .articles,.articles.f,$(GIGA_ARTICLES_FILES))
	cat $< | xargs -I % cat $(GIGA_TMP_DIR)%.f > $@

%.vocab: %.sumdata data_generic.py
	python3 $(word 2,$^) vocab $< > $@

data: $(GIGA_DIR).gigaword-data

$(GIGA_DIR).gigaword-data: $(GIGA_DIR)train.sumdata $(GIGA_DIR)strain.sumdata $(GIGA_DIR)valid.sumdata $(GIGA_DIR)test.sumdata $(GIGA_DIR)train.vocab $(GIGA_DIR)strain.vocab
	mkdir -p $(GIGA_DIR)data-$(TAG)
	shuf $(GIGA_DIR)train.sumdata --random-source=$(GIGA_DIR)test.sumdata > $(GIGA_DIR)train.shuf.sumdata
	mv -f $(GIGA_DIR)train.shuf.sumdata $(GIGA_DIR)train.sumdata
	touch $(GIGA_DIR)train.vocab
	shuf -n 1000 $(GIGA_DIR)test.sumdata --random-source=$(GIGA_DIR)test.sumdata > $(GIGA_DIR)test.1k.sumdata
	shuf -n 8000 $(GIGA_DIR)test.sumdata --random-source=$(GIGA_DIR)test.sumdata > $(GIGA_DIR)test.8k.sumdata
	cp -p $(GIGA_DIR)*.sumdata $(GIGA_DIR)*.vocab $(GIGA_DIR)*.splits data_*.py $(GIGA_DIR)data-$(TAG)
	touch $@


# training/test actions
tb tensorboard:
	killall -q -w tensorboard ; true
	CUDA_VISIBLE_DEVICES= tensorboard --port 2345 --logdir=model &

tm trainmodel:
	python3 seq2seq_attention.py $(ARGS) \
		--mode=train --batch_size=64 \
		--data_path=$(GIGA_DIR)train.sumdata \
		--vocab_path=$(GIGA_DIR)train.vocab \
		--log_root=model

cleanmodel:
	rm -rf model

train: data cleanmodel tensorboard trainmodel

ctrain: data tensorboard trainmodel

TRAINING_ROOT=training_center/$(TAG)/

tc trainingcenter: data
	mkdir -p $(TRAINING_ROOT)
	cp -p --parents $(GIGA_DIR)*.sumdata $(GIGA_DIR)*.vocab *.py Makefile $(TRAINING_ROOT)
	$(MAKE) -C $(TRAINING_ROOT) tensorboard trainmodel

decode:
	CUDA_VISIBLE_DEVICES= python3 seq2seq_attention.py $(ARGS) \
		--mode=decode \
		--data_path=$(GIGA_DIR)test.1k.sumdata \
		--vocab_path=$(GIGA_DIR)train.vocab \
		--log_root=model \
		--beam_size=8

cdecode:
	watch -n 10800 $(MAKE) decode
