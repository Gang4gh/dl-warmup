DATA_DIR=/media/hdd1T/home_backup/gang/LDC2012T21/Data/data/xml/
TMP_DIR=articles/

all: training.articles test.articles validation.articles

clean:
	rm -rf $(TMP_DIR)
	rm -f *.articles

.DELETE_ON_ERROR:

GZ_FILES=$(wildcard $(DATA_DIR)*.xml.gz)
ARTICLES_FILES=$(GZ_FILES:$(DATA_DIR)%.xml.gz=$(TMP_DIR)%.articles)

$(TMP_DIR)%.articles : $(DATA_DIR)%.xml.gz convert_gigaword.py | $(TMP_DIR)
	python3 $(word 2,$^) $< > $@

$(TMP_DIR):
	mkdir -p $(TMP_DIR)

# this target is defined to preserve *.articles files
keep-articles: $(ARTICLES_FILES)

%.articles: %.splits $(ARTICLES_FILES)
	cat $< | xargs -I % cat $(TMP_DIR)% > $@
