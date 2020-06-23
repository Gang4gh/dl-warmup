SHELL = /bin/bash
CPUS = $(shell nproc)
export TF_CPP_MIN_LOG_LEVEL=1

-include Makefile.config
DTITLE_RAW ?= snapshot-0403-small.7z
ifeq (1, $(words $(DTITLE_RAW)))
SPLIT_DIR ?= split_raw/cache_$(notdir $(DTITLE_RAW))/
else
SPLIT_DIR ?= split_raw/cache_$(shell echo -n "$(DTITLE_RAW)" | md5sum | cut -f1 -d' ')/
endif
DTITLE_FILES := $(shell for i in {100..195}; do echo $(SPLIT_DIR)data-$${i:1}.$(TAG).dtitle; done)
TAG ?=
ARGS ?=
VOCAB_FILE ?= data-v3-vocab-24gb

all: $(SPLIT_DIR)all-data.md5
	$(MAKE) -j$(CPUS) $(TAG)-meta.log

split-data: $(SPLIT_DIR)all-data.md5

clean:
	rm -rf $(SPLIT_DIR)

VARIABLES = DTITLE_RAW SPLIT_DIR TAG ARGS VOCAB_FILE
check:
	$(foreach var,$(VARIABLES),$(info $(var) = $($(var))))

.DELETE_ON_ERROR:

.INTERMEDIATE: $(DTITLE_FILES)

$(SPLIT_DIR)all-data.md5: $(DTITLE_RAW)
ifeq (, $(shell which 7z))
	$(error "No 7z in $(PATH), consider apt install p7zip-full")
else
	mkdir -p $(SPLIT_DIR)
	rm -rf $(SPLIT_DIR)*.raw $(SPLIT_DIR)*.raw.gz
	(for fi in $^; do 7z e -so $$fi; done) | split -d -nr/96 --additional-suffix=.raw - $(SPLIT_DIR)data-
	$(MAKE) -j$(CPUS) $(shell for i in {100..195}; do echo $(SPLIT_DIR)data-$${i:1}.raw.gz; done)
	echo "#DTITLE_RAW = $(DTITLE_RAW)" > $@
	md5sum $(SPLIT_DIR)*.raw.gz >> $@
endif

%.raw.gz: %.raw
	gzip $<

%.$(TAG).dtitle: %.raw.gz
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< $(ARGS) \
		| shuf --random-source=$(firstword $(DTITLE_RAW)) > $@

$(TAG)-training.dtitle.gz: $(DTITLE_FILES)
	tail -q -n +1001 $^ | gzip > $@

$(TAG)-test.dtitle: $(DTITLE_FILES)
	head -q -n 1000 $^ > $@

%.dtitle.gz: %.dtitle
	gzip -fk $<

%.subwords: $(SPLIT_DIR)all-data.md5
	python3 process_dtitle_data.py --cmd=build-vocab --input_file=$(SPLIT_DIR)data-*.raw.gz --vocab_file=$* $(ARGS)

$(TAG)-vocab.subwords:
	cp $(VOCAB_FILE).subwords $@

%.dtitle.tokenized.gz: %.dtitle.gz $(TAG)-vocab.subwords
	python3 process_dtitle_data.py --cmd=tokenize-dtitle-v2 --input_file=$< --vocab_file=$(TAG)-vocab $(ARGS)

$(TAG)-meta.log: $(TAG)-training.dtitle.tokenized.gz $(TAG)-test.dtitle $(TAG)-test.dtitle.tokenized.gz $(TAG)-vocab.subwords
	python3 process_dtitle_data.py --cmd=print-flags --vocab_file=$(TAG)-vocab $(ARGS) > $@
	@echo ---------- source code  ---------- >> $@
	cat process_dtitle_data.py >> $@
	@echo ---------- data files md5sum ---------- >> $@
	md5sum $(TAG)-*.* >> $@

TEST_DATA = ~/CaptionData/October_Scraping_joinedData-1204.tsv

%.test.dtitle: $(TEST_DATA)
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< --for_inference $(ARGS) | sort | uniq > $@.tmp
	mv $@.tmp $@
