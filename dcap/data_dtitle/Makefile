SHELL = /bin/bash
CPUS = $(shell nproc)

DTITLE_RAW ?= raw-data.gz
SPLIT_DIR = split_raw/
DTITLE_FILES = $(shell for i in {100..199}; do echo $(SPLIT_DIR)data-$${i:1}.raw.dtitle; done)
TAG ?= 

all:
	$(MAKE) $(SPLIT_DIR)data.md5
	$(MAKE) -j$(CPUS) $(TAG)training.dtitle $(TAG)test.dtitle $(TAG)vocab-16384.subwords #training.tfe test.tfe
	$(MAKE) clean

clean:
	rm -rf $(SPLIT_DIR)

cleanall: clean
	rm -rf *.dtitle *.subwords *.tfe

.DELETE_ON_ERROR:

$(SPLIT_DIR)data.md5: $(DTITLE_RAW)
	mkdir -p $(SPLIT_DIR)
	rm -f $(SPLIT_DIR)*.raw
	#split -d -nr/100 --additional-suffix=.raw $< $(SPLIT_DIR)data-
	gunzip -c $< | split -d -nr/100 --additional-suffix=.raw - $(SPLIT_DIR)data-
	md5sum $(SPLIT_DIR)*.raw > $@

$(TAG)training.dtitle: $(DTITLE_FILES)
	tail -q -n +1001 $^ > $@

$(TAG)test.dtitle: $(DTITLE_FILES)
	head -q -n 1000 $^ > $@

vocabs: $(TAG)vocab-4096.subwords $(TAG)vocab-8192.subwords $(TAG)vocab-16384.subwords $(TAG)vocab-32768.subwords $(TAG)vocab-65536.subwords

$(TAG)vocab-%.subwords: $(TAG)training.dtitle
	python3 process_dtitle_data.py build-vocab $< $(TAG)vocab $* 16 4

#%.tfe: %.dtitle $(TAG)vocab-8192.subwords
#	python3 process_dtitle_data.py build-tfexample $< $(TAG)vocab-8192 $@

%.raw.dtitle: %.raw
	python3 process_dtitle_data.py pre-process $< $(TAG) \
		| shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:17 -nosalt </dev/zero 2>/dev/null) > $@
