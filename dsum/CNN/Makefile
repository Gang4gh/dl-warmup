DATA_DIR ?= /home/gang/CNN-DailyMail/cnn_stories_tokenized/
DATA_URLS_BASE ?= https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/cnn_wayback

all: training.articles test.articles validation.articles

clean:
	rm -f *.articles *.split-urls

.DELETE_ON_ERROR:

%.split-urls :
	wget $(DATA_URLS_BASE)_$*_urls.txt -O $@
keep-urls: training.split-urls test.split-urls validation.split-urls

%.articles: %.split-urls ../CNN/convert_cnndm.py
	python3 $(word 2,$^) $< $(DATA_DIR) > $@
