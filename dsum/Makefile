CPUS ?= $(shell nproc)
#MAKEFLAGS += -j24	# doesn't work with make v4.1
TIMESTAMP = $(shell date +'%Y%m%d-%H%M%S')

-include .makeargs
TAG ?= notag
PORT ?= 2345


# preprocess gigaword training data
GIGA_DATA_DIR=/home/gang/LDC2012T21/Data/data/xml/
GIGA_DIR=Gigaword/
GIGA_TMP_DIR=$(GIGA_DIR)articles/


all:
	$(MAKE) -j$(CPUS) data


clean:
	rm -rf $(GIGA_TMP_DIR)
	rm -f $(GIGA_DIR)*.sumdata $(GIGA_DIR)*.vocab $(GIGA_DIR).datatag

check:
	@echo TAG is '$(TAG)'

.DELETE_ON_ERROR:


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

%.raw.sumdata: %.splits $(subst .articles,.articles.f,$(GIGA_ARTICLES_FILES))
	cat $< | xargs -I % cat $(GIGA_TMP_DIR)%.f > $@

%.sumdata: %.raw.sumdata
	shuf $< --random-source=$< > $@

%.vocab: %.sumdata data_generic.py
	python3 $(word 2,$^) vocab $< 100000 > $@

data: $(GIGA_DIR).datatag

$(GIGA_DIR).datatag: $(GIGA_DIR)train.sumdata $(GIGA_DIR)test.sumdata $(GIGA_DIR)valid.sumdata $(GIGA_DIR)train.vocab
	head -n 1000 $(GIGA_DIR)test.sumdata > $(GIGA_DIR)test.1k.sumdata
	mkdir -p $(GIGA_DIR)data-$(TIMESTAMP)
	cp -p $(GIGA_DIR)*.sumdata $(GIGA_DIR)*.vocab $(GIGA_DIR)*.splits data_*.py $(GIGA_DIR)data-$(TIMESTAMP)
	md5sum $(GIGA_DIR)data-$(TIMESTAMP)/* > $@


# training/test actions
tb tensorboard:
	pkill -f tensorboard\ --port\ $(PORT); sleep 1
	CUDA_VISIBLE_DEVICES= tensorboard --port $(PORT) --window_title $(lastword $(subst /, ,$(CURDIR))) --logdir model &
	
tm trainmodel:
	python3 summarization.py $(ARGS) \
		--mode=train --batch_size=64 \
		--data_path=$(GIGA_DIR)train.sumdata \
		--vocab_path=$(GIGA_DIR)train.vocab \
		--log_root=model

cleanmodel:
	rm -rf model

train: data cleanmodel tensorboard trainmodel

ctrain: data tensorboard trainmodel

TRAINING_ROOT=training_center/$(TAG)-$(TIMESTAMP)/

tc trainingcenter: data
	mkdir -p $(TRAINING_ROOT)
	cp -p --parents $(GIGA_DIR)*.sumdata $(GIGA_DIR)*.vocab $(GIGA_DIR).datatag *.py Makefile $(TRAINING_ROOT)
	echo ARGS ?= $(ARGS) > $(TRAINING_ROOT).makeargs
	echo PORT ?= $(PORT) >> $(TRAINING_ROOT).makeargs
	$(MAKE) -C $(TRAINING_ROOT) train

decode:
	python3 summarization.py $(ARGS) \
		--mode=decode --batch_size=25 \
		--data_path=$(GIGA_DIR)test.1k.sumdata \
		--vocab_path=$(GIGA_DIR)train.vocab \
		--log_root=model \
		--beam_size=8

cdecode:
	watch -n 3600 $(MAKE) decode

