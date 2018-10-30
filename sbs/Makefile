training: trainingdata.vocab sbsdatacleaned20181008_flips_tokLower.tsv naivedl.py
	python3 naivedl.py

trainingdata.vocab: trainingdata.tsv data_generic.py
	python3 data_generic.py build-vocab $< 5 0,1,2 10 > $@

trainingdata.tsv: sbsdatacleaned20181008_flips_tokLower.tsv sbs_data.py
	python3 sbs_data.py
