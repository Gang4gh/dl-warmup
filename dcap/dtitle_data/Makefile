SHELL = /bin/bash
CPUS = $(shell nproc)

DTITLE_RAW = /home/gang/CaptionData/retroindex-0930-sample-5percent-cooked20k.txt.gz
SPLIT_DIR = split_raw/
DTITLE_FILES = $(shell for i in {100..199}; do echo data-$${i:1}.raw.dtitle; done)

all: $(SPLIT_DIR)data.md5
	$(MAKE) -j$(CPUS) training.dtitle test.dtitle vocab-8192.subwords #training.tfe test.tfe

clean:
	rm -rf *.dtitle *.subwords *.tfe $(SPLIT_DIR)

.DELETE_ON_ERROR:

.INTERMEDIATE: $(DTITLE_FILES)

$(SPLIT_DIR)data.md5: $(DTITLE_RAW)
	mkdir -p $(SPLIT_DIR)
	rm -f $(SPLIT_DIR)*.raw
	#split -d -nr/100 --additional-suffix=.raw $< $(SPLIT_DIR)data-
	gunzip -c $< | split -d -nr/100 --additional-suffix=.raw - $(SPLIT_DIR)data-
	md5sum $(SPLIT_DIR)*.raw > $@

training.dtitle: $(DTITLE_FILES)
	tail -q -n +1001 $^ > $@

test.dtitle: $(DTITLE_FILES)
	head -q -n 1000 $^ > $@

vocabs: vocab-4096.subwords vocab-8192.subwords vocab-16384.subwords vocab-32768.subwords vocab-65536.subwords

vocab-%.subwords: training.dtitle
	python3 process_dtitle_data.py build-vocab $< vocab $*

#%.tfe: %.dtitle vocab-8192.subwords
#	python3 process_dtitle_data.py build-tfexample $< vocab-8192 $@

%.raw.dtitle: $(SPLIT_DIR)%.raw
	python3 process_dtitle_data.py pre-process $< \
		| shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:17 -nosalt </dev/zero 2>/dev/null) > $@
