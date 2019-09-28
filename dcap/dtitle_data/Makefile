SHELL = /bin/bash
CPUS = $(shell nproc)

DTITLE_RAW = /home/gang/CaptionData/retroindex-0927-sample-5percent-cooked30k.txt
SPLIT_DIR = split_raw/
DTITLE_FILES = $(shell for i in {100..199}; do echo data-$${i:1}.raw.dtitle; done)

all: $(SPLIT_DIR)
	$(MAKE) -j$(CPUS) training.dtitle test.dtitle vocab.subwords training.tfe test.tfe

clean:
	rm -rf *.dtitle *.subwords *.tfe $(SPLIT_DIR)

.DELETE_ON_ERROR:

.INTERMEDIATE: $(DTITLE_FILES)

$(SPLIT_DIR): $(DTITLE_RAW)
	mkdir -p $(SPLIT_DIR)
	rm -f *.raw
	split -d -nl/100 --additional-suffix=.raw $< $(SPLIT_DIR)data-
	md5sum $(SPLIT_DIR)*.raw > $@data.md5

training.dtitle: $(DTITLE_FILES)
	tail -q -n +1001 $^ > $@

test.dtitle: $(DTITLE_FILES)
	head -q -n 1000 $^ > $@

%.subwords: training.dtitle
	python3 process_dtitle_data.py build-vocab $< $*

%.tfe: %.dtitle vocab.subwords
	python3 process_dtitle_data.py build-tfexample $< vocab $@

%.raw.dtitle: $(SPLIT_DIR)%.raw
	python3 process_dtitle_data.py pre-process $< \
		| shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:17 -nosalt </dev/zero 2>/dev/null) > $@

