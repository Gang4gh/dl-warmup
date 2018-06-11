SHELL = /bin/bash
CPUS = $(shell nproc)
TIMESTAMP = $(shell date +'%Y%m%d-%H%M%S')

-include Makefile.config
DATASOURCES ?= Gigaword
TAG ?= notag
PORT ?= 2345


all: labeled-data

clean:
	rm -f */data.md5 *.articles *.vocab

cleanall: clean
	for ds in $(DATASOURCES); do $(MAKE) -C $$ds clean; done

check:
	@echo TAG is '$(TAG)'

.DELETE_ON_ERROR:


# data pre-processing
%/data.md5:
	$(MAKE) -j$(CPUS) -C $*
	md5sum $*/*.articles > $@
keep-data-md5: $(addsuffix /data.md5, $(DATASOURCES))

labeled-data: training.articles training.vocab test.articles test-sample.articles validation.articles
	@echo labeled data is prepared from \'$(DATASOURCES)\'

%.articles: data_generic.py $(addsuffix /data.md5, $(DATASOURCES))
	cat $(addsuffix /$*.articles, $(DATASOURCES)) \
		| python3 $< filter - 2 120 30 \
		| shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:17 -nosalt </dev/zero 2>/dev/null) \
		> $@

%.vocab: %.articles data_generic.py
	python3 $(word 2,$^) build-vocab $< | sort -k2nr -k1 > $@

test-sample.articles: test.articles
	head -n 4000 $< > $@

# training/test actions
tb tensorboard:
	pkill -f tensorboard\ --port\ $(PORT); sleep 1
	CUDA_VISIBLE_DEVICES= tensorboard --port $(PORT) --window_title $(lastword $(subst /, ,$(CURDIR))) --logdir model &
	
tm trainmodel: labeled-data
	python3 summarization.py $(ARGS) \
		--mode=train --batch_size=64 \
		--data_path=training.articles \
		--vocab_path=training.vocab \
		--log_root=model

cleanmodel:
	rm -rf model

train: labeled-data cleanmodel tensorboard trainmodel

ctrain: labeled-data tensorboard trainmodel

TRAINING_ROOT=running_center/$(TAG)-$(TIMESTAMP)/

tc trainingcenter: labeled-data
	mkdir -p $(TRAINING_ROOT)
	cp -p --parents *.py Makefile *.articles *.vocab */data.md5 $(TRAINING_ROOT)
	echo ARGS ?= $(ARGS) > $(TRAINING_ROOT)Makefile.config
	echo PORT ?= $(PORT) >> $(TRAINING_ROOT)Makefile.config
	$(MAKE) -C $(TRAINING_ROOT) train

decode: labeled-data
	python3 summarization.py $(ARGS) \
		--mode=decode --batch_size=25 \
		--data_path=test-sample.articles \
		--vocab_path=training.vocab \
		--log_root=model \
		--beam_size=8

cdecode: labeled-data
	watch -n 3600 $(MAKE) decode

