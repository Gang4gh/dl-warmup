CPUS ?= $(shell nproc)
#MAKEFLAGS += -j24	# doesn't work at make 4.1

all:
	$(MAKE) -j$(CPUS) gigaword-data

# preprocess gigaword training data
GIGA_DIR=./Gigaword/
GIGA_RAW=$(GIGA_DIR)LDC2012T21/Data/data/xml/
GIGA_ARTICLES=$(GIGA_DIR)articles/

GIGA_GZ_FILES=$(wildcard $(GIGA_RAW)*.xml.gz)
GIGA_ARTICLES_FILES=$(patsubst $(GIGA_RAW)%.xml.gz,$(GIGA_ARTICLES)%.articles,$(GIGA_GZ_FILES))

$(GIGA_ARTICLES)%.articles : $(GIGA_RAW)%.xml.gz ./convert_gigaword.py
	python2.7 $(word 2,$^) $< $@

# this target is defined to preserve *.articles files
gigaword-articles: $(GIGA_ARTICLES_FILES)

%.articles.filtered: %.articles ./process_gigaword.py
	python3 $(word 2,$^) filter $< > $@

%.sumdata: %.splits $(subst .articles,.articles.filtered,$(GIGA_ARTICLES_FILES))
	cat $< | xargs -I % cat $(GIGA_ARTICLES)%.filtered > $@

%.vocab: %.sumdata ./process_gigaword.py
	python3 $(word 2,$^) vocab $< > $@

gigaword-data: $(GIGA_DIR)train.sumdata $(GIGA_DIR)strain.sumdata $(GIGA_DIR)valid.sumdata $(GIGA_DIR)test.sumdata $(GIGA_DIR)train.vocab $(GIGA_DIR)strain.vocab
