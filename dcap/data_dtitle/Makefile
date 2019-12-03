SHELL = /bin/bash
CPUS = $(shell nproc)

DTITLE_RAW = raw-data.7z
SPLIT_DIR = split_raw/
TAG =
ARGS =
DTITLE_FILES = $(shell for i in {100..199}; do echo $(SPLIT_DIR)data-$${i:1}.$(TAG).dtitle; done)

all: $(SPLIT_DIR)all-data.md5
	$(MAKE) -j$(CPUS) $(TAG)-meta.log

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
	7z e -so $< | split -d -nr/100 --additional-suffix=.raw - $(SPLIT_DIR)data-
	$(MAKE) -j$(CPUS) $(shell for i in {100..199}; do echo $(SPLIT_DIR)data-$${i:1}.raw.gz; done)
	md5sum $(SPLIT_DIR)*.raw.gz > $@
endif

%.raw.gz: %.raw
	gzip $<

%.$(TAG).dtitle: %.raw.gz
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< $(ARGS) \
		| shuf --random-source=$(DTITLE_RAW) > $@

$(TAG)-training.dtitle: $(DTITLE_FILES)
	tail -q -n +101 $^ > $@

$(TAG)-test.dtitle: $(DTITLE_FILES)
	head -q -n 100 $^ > $@

$(TAG)-vocab-%.subwords: $(TAG)-training.dtitle
	python3 process_dtitle_data.py --cmd=build-vocab --input_file=$< --vocab_file_prefix=$(TAG)-vocab --target_vocab_size=$* $(ARGS)

$(TAG)-meta.log: $(TAG)-training.dtitle $(TAG)-test.dtitle $(TAG)-vocab-8192.subwords
	python3 process_dtitle_data.py --cmd=print-flags --vocab_file_prefix=$(TAG)-vocab $(ARGS) > $@
	@echo ---------- source code  ---------- >> $@
	cat process_dtitle_data.py >> $@
	@echo ---------- data files md5sum ---------- >> $@
	md5sum $(TAG)-*.* >> $@

TEST_DATA = ~/CaptionData/October_Scraping_UrlSet_joinedData-1125-preprocessed.raw
preprocess-testdata: $(TEST_DATA)
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< --nosuppress_enoughtokens --nosuppress_exactmatch $(ARGS) > test-$(TAG).dtitle
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< $(ARGS) > test-$(TAG)2s.dtitle
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< --check_enoughtokens --check_exactmatch $(ARGS) > test-$(TAG)2c.dtitle

%.test-dtitle: $(TEST_DATA)
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< --nosuppress_enoughtokens --nosuppress_exactmatch $(ARGS) > $@
