SHELL = /bin/bash
CPUS = $(shell nproc)

RAW_INPUT ?= /home/gang/CaptionData/retroindex-0920-sample-5percent-cut20k.txt
DATA_DIR ?= raw/

RAW_FILES=$(wildcard $(DATA_DIR)data*.raw)
TITLE_FILES=$(RAW_FILES:$(DATA_DIR)%.raw=%.titles)

all: $(DATA_DIR).data.md5
	$(MAKE) -j$(CPUS) training.titles test.titles

clean:
	rm -rf $(DATA_DIR) *.md5 *.titles

.DELETE_ON_ERROR:

.INTERMEDIATE: $(TITLE_FILES)

$(DATA_DIR).data.md5: $(RAW_INPUT)
	mkdir -p $(DATA_DIR)
	rm -f $(DATA_DIR)*.raw
	split -d -nl/64 --additional-suffix=.raw $< $(DATA_DIR)data-
	md5sum $(DATA_DIR)*.raw > $(DATA_DIR).data.md5

%.titles: $(DATA_DIR)%.raw convert2captitles.py
	python3 $(word 2, $^) $< \
		| shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:17 -nosalt </dev/zero 2>/dev/null) > $@

test.titles: $(TITLE_FILES)
	head -q -n 100 $(TITLE_FILES) > $@

training.titles: $(TITLE_FILES)
	tail -q -n +101 $(TITLE_FILES) > $@

