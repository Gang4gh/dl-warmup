SHELL = /bin/bash
CPUS = $(shell nproc)

DTITLE_RAW = raw-data.gz
SPLIT_DIR = split_raw/
TAG =
ARGS =
DTITLE_FILES = $(shell for i in {100..199}; do echo $(SPLIT_DIR)data-$${i:1}.$(TAG).dtitle; done)

all: $(SPLIT_DIR)data.md5
	$(MAKE) -j$(CPUS) $(TAG)-meta.log

clean:
	rm -rf $(SPLIT_DIR)

.DELETE_ON_ERROR:

.INTERMEDIATE: $(DTITLE_FILES)

$(SPLIT_DIR)data.md5: $(DTITLE_RAW)
	mkdir -p $(SPLIT_DIR)
	rm -rf $(SPLIT_DIR)*.raw $(SPLIT_DIR)*.raw.gz
	gunzip -c $< | split -d -nr/100 --additional-suffix=.raw - $(SPLIT_DIR)data-
	$(MAKE) -j$(CPUS) $(shell for i in {100..199}; do echo $(SPLIT_DIR)data-$${i:1}.raw.gz; done)
	md5sum $(SPLIT_DIR)*.raw.gz > $@

%.raw.gz: %.raw
	gzip $<

%.$(TAG).dtitle: %.raw.gz
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< $(ARGS) \
		| shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:17 -nosalt </dev/zero 2>/dev/null) > $@

$(TAG)-training.dtitle: $(DTITLE_FILES)
	tail -q -n +101 $^ > $@

$(TAG)-test.dtitle: $(DTITLE_FILES)
	head -q -n 100 $^ > $@

$(TAG)-vocab-%.subwords: $(TAG)-training.dtitle
	python3 process_dtitle_data.py --cmd=build-vocab --input_file=$< --vocab_file_prefix=$(TAG)-vocab --target_vocab_size=$* $(ARGS)

$(TAG)-meta.log: $(TAG)-training.dtitle $(TAG)-test.dtitle $(TAG)-vocab-16384.subwords
	python3 process_dtitle_data.py --cmd=print-flags --vocab_file_prefix=$(TAG)-vocab $(ARGS) > $@
	@echo ---------- CODE ---------- >> $@
	cat process_dtitle_data.py >> $@
	@echo ---------- Data files md5(s) ---------- >> $@
	md5sum $(TAG)-*.* >> $@

