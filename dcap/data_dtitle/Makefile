SHELL = /bin/bash
CPUS = $(shell nproc)
export TF_CPP_MIN_LOG_LEVEL=1

-include Makefile.local.config
DTITLE_RAW ?= snapshot-0403-small.7z
SHUF_RANDOM_SOURCE ?= snapshot-0403-small.7z
SPLIT_DIR = split_raw/
TAG =
ARGS =
DTITLE_FILES = $(shell for i in {100..195}; do echo $(SPLIT_DIR)data-$${i:1}.$(TAG).dtitle; done)
VOCAB_FILE = data-v3-vocab-24gb

all: $(SPLIT_DIR)all-data.md5
	$(MAKE) -j$(CPUS) $(TAG)-meta.log

split-data: $(SPLIT_DIR)all-data.md5

clean:
	rm -rf $(SPLIT_DIR)

.DELETE_ON_ERROR:

.INTERMEDIATE: $(DTITLE_FILES)

$(SPLIT_DIR)all-data.md5: $(DTITLE_RAW)
ifeq (, $(shell which 7z))
	$(error "No 7z in $(PATH), consider doing apt install p7zip-full")
else
	mkdir -p $(SPLIT_DIR)
	rm -rf $(SPLIT_DIR)*.raw $(SPLIT_DIR)*.raw.gz
	7z e -so $< | split -d -nr/96 --additional-suffix=.raw - $(SPLIT_DIR)data-
	$(MAKE) -j$(CPUS) $(shell for i in {100..195}; do echo $(SPLIT_DIR)data-$${i:1}.raw.gz; done)
	md5sum $(SPLIT_DIR)*.raw.gz > $@
endif

%.raw.gz: %.raw
	gzip $<

%.$(TAG).dtitle: %.raw.gz
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< $(ARGS) \
		| shuf --random-source=$(SHUF_RANDOM_SOURCE) > $@

$(TAG)-training.dtitle.gz: $(DTITLE_FILES)
	tail -q -n +1001 $^ | gzip > $@

$(TAG)-test.dtitle: $(DTITLE_FILES)
	head -q -n 1000 $^ > $@

%.dtitle.gz: %.dtitle
	gzip -fk $<

$(TAG)-vocab-%.subwords: $(TAG)-training.dtitle.gz
	#python3 process_dtitle_data.py --cmd=build-vocab --input_file=$< --vocab_file_prefix=$(TAG)-vocab --target_vocab_size=$* $(ARGS)
	cp -lp $(VOCAB_FILE)-$*.subwords $@

%.tokenized-tfrecord.gz: %.dtitle.gz
	python3 process_dtitle_data.py --cmd=tokenize-dtitle-mp --input_file=$< --vocab_file_prefix=$(VOCAB_FILE) --target_vocab_size=8192 $(ARGS)

$(TAG)-meta.log: $(TAG)-training.tokenized-tfrecord.gz $(TAG)-test.dtitle $(TAG)-test.tokenized-tfrecord.gz $(TAG)-vocab-8192.subwords
	python3 process_dtitle_data.py --cmd=print-flags --vocab_file_prefix=$(TAG)-vocab $(ARGS) > $@
	@echo ---------- source code  ---------- >> $@
	cat process_dtitle_data.py >> $@
	@echo ---------- data files md5sum ---------- >> $@
	md5sum $(TAG)-*.* >> $@

data-v3-vocab-1gb-%.subwords:
	python3 process_dtitle_data.py --cmd=build-vocab --input_file=$(SPLIT_DIR)data-*.raw.gz --vocab_file_prefix=data-v3-vocab-1gb --target_vocab_size=$* --max_corpus_chars=1
data-v3-vocab-4gb-%.subwords:
	python3 process_dtitle_data.py --cmd=build-vocab --input_file=$(SPLIT_DIR)data-*.raw.gz --vocab_file_prefix=data-v3-vocab-4gb --target_vocab_size=$* --max_corpus_chars=4
data-v3-vocab-16gb-%.subwords:
	python3 process_dtitle_data.py --cmd=build-vocab --input_file=$(SPLIT_DIR)data-*.raw.gz --vocab_file_prefix=data-v3-vocab-16gb --target_vocab_size=$* --max_corpus_chars=16
data-v3-vocab-24gb-%.subwords:
	python3 process_dtitle_data.py --cmd=build-vocab --input_file=$(SPLIT_DIR)data-*.raw.gz --vocab_file_prefix=data-v3-vocab-24gb --target_vocab_size=$* --max_corpus_chars=24
data-v3-vocab-32gb-%.subwords:
	python3 process_dtitle_data.py --cmd=build-vocab --input_file=$(SPLIT_DIR)data-*.raw.gz --vocab_file_prefix=data-v3-vocab-32gb --target_vocab_size=$* --max_corpus_chars=32
data-v3-vocab-256gb-%.subwords:
	python3 process_dtitle_data.py --cmd=build-vocab --input_file=$(SPLIT_DIR)data-*.raw.gz --vocab_file_prefix=data-v3-vocab-256gb --target_vocab_size=$* --max_corpus_chars=256

TEST_DATA = ~/CaptionData/October_Scraping_joinedData-1204.tsv
TEST_DATA_SCHEMA = cap_query,cap_url,cap_title,cap_snippet,url,hostname,visual_title,title,html
preprocess-testdata: $(TEST_DATA)
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< --nosuppress_enoughtokens --nosuppress_exactmatch --input_schema=$(TEST_DATA_SCHEMA) $(ARGS) | uniq > $(notdir $(TEST_DATA)).test.dtitle
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< --input_schema=$(TEST_DATA_SCHEMA) $(ARGS) | uniq > $(notdir $(TEST_DATA))-2s.test.dtitle
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< --check_enoughtokens --check_exactmatch --input_schema=$(TEST_DATA_SCHEMA) $(ARGS) | uniq > $(notdir $(TEST_DATA))-2c.test.dtitle

%.test.dtitle: $(TEST_DATA)
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< --nosuppress_enoughtokens --nosuppress_exactmatch --input_schema=$(TEST_DATA_SCHEMA) $(ARGS) | uniq > $@
