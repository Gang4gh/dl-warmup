SHELL = /bin/bash
CPUS = $(shell nproc)

DTITLE_RAW ?= raw-data.gz
SPLIT_DIR = split_raw/
DTITLE_FILES = $(shell for i in {100..199}; do echo $(SPLIT_DIR)data-$${i:1}.raw.dtitle; done)
TAG ?=
ARGS ?=

all:
	$(MAKE) $(SPLIT_DIR)data.md5
	$(MAKE) -j$(CPUS) $(TAG)meta
	#$(MAKE) clean

clean:
	rm -rf $(SPLIT_DIR)

cleanall: clean
	@echo type the command to clean all data files: rm -rf *.dtitle *.subwords

.DELETE_ON_ERROR:

$(SPLIT_DIR)data.md5: $(DTITLE_RAW)
	mkdir -p $(SPLIT_DIR)
	rm -f $(SPLIT_DIR)*.raw
	#split -d -nr/100 --additional-suffix=.raw $< $(SPLIT_DIR)data-
	gunzip -c $< | split -d -nr/100 --additional-suffix=.raw - $(SPLIT_DIR)data-
	md5sum $(SPLIT_DIR)*.raw > $@

%.raw.dtitle: %.raw
	python3 process_dtitle_data.py --cmd=pre-process --input_file=$< $(ARGS) \
		| shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:17 -nosalt </dev/zero 2>/dev/null) > $@

$(TAG)training.dtitle: $(DTITLE_FILES)
	tail -q -n +1001 $^ > $@

$(TAG)test.dtitle: $(DTITLE_FILES)
	head -q -n 1000 $^ > $@

#vocabs: $(TAG)vocab-4096.subwords $(TAG)vocab-8192.subwords $(TAG)vocab-16384.subwords $(TAG)vocab-32768.subwords $(TAG)vocab-65536.subwords

$(TAG)vocab-%.subwords: $(TAG)training.dtitle
	python3 process_dtitle_data.py --cmd=build-vocab --input_file=$< --vocab_file_prefix=$(TAG)vocab --target_vocab_size=$* $(ARGS)

$(TAG)meta: $(TAG)training.dtitle $(TAG)test.dtitle $(TAG)vocab-16384.subwords
	python3 process_dtitle_data.py --cmd=print-flags --vocab_file_prefix=$(TAG)vocab $(ARGS) > $@
	md5sum $(TAG)*.* >> $@

