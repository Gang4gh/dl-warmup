SHELL = /bin/bash
CPUS = $(shell nproc)

RAW_INPUT ?= /home/gang/CaptionData/retroindex-0920-sample-5percent-cut20k.txt
DATA_DIR ?= raw/
RAW_FILES=$(wildcard $(DATA_DIR)data*.raw)
TITLE_FILES=$(RAW_FILES:%.raw=%.titles)

all: $(DATA_DIR)raw-data.md5
	$(MAKE) -j$(CPUS) test.titles training.titles

clean:
	rm -rf $(DATA_DIR) *.md5 *.titles

.DELETE_ON_ERROR:

#.INTERMEDIATE: $(TITLE_FILES) $(RAW_FILES)

$(DATA_DIR)raw-data.md5: $(RAW_INPUT)
	mkdir -p $(DATA_DIR)
	split -d -nl/100 --additional-suffix=.raw $< $(DATA_DIR)data-
	md5sum $(DATA_DIR)*.raw > $@

%.titles: %.raw convert2captitles.py
	python3 $(word 2, $^) $< \
		| shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:17 -nosalt </dev/zero 2>/dev/null) > $@

test.titles: $(TITLE_FILES)
	rm -f $@
	for i in $(TITLE_FILES); do head -n 1000 $$i >> $@; done

training.titles: $(TITLE_FILES)
	rm -f $@
	for i in $(TITLE_FILES); do tail -n +1001 $$i >> $@; done

