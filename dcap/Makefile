SHELL = /bin/bash
CPUS = $(shell nproc)
TIMESTAMP = $(shell date +'%Y%m%d-%H%M%S')

-include Makefile.config
DATASOURCES ?= CapTitle #CNN DailyMail
TAG ?= notag
PORT ?= 2345
MDIR ?= model


all:
	python lt.py --mode=train

clean:
	rm -f */data.md5 *.articles *.vocab

cleanall: clean
	for ds in $(DATASOURCES); do $(MAKE) -C $$ds clean; done

.DELETE_ON_ERROR:


# data pre-processing
%/data.md5:
	$(MAKE) -j$(CPUS) -C $*
	md5sum $*/*.dcaptitles > $@
keep-data-md5: $(addsuffix /data.md5, $(DATASOURCES))

labeled-data: training.dcaptitles test.dcaptitles
	@echo labeled data is prepared from \'$(DATASOURCES)\'

all.dcaptitles: $(addsuffix /data.md5, $(DATASOURCES))
	cat $(addsuffix /$*.dcaptitles, $(DATASOURCES)) \
		| shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:17 -nosalt </dev/zero 2>/dev/null) > $@

training.dcaptitles: all.dcaptitles
	head -n 4000 $< > $@

test.dcaptitles: all.dcaptitles
	tail -n 10000 $< > $@

# training/test actions
tb tensorboard:
	pkill -f tensorboard\ --port\ $(PORT); sleep 1
	CUDA_VISIBLE_DEVICES= tensorboard --port $(PORT) --window_title $(lastword $(subst /, ,$(CURDIR))) --logdir $(MDIR) 2>/dev/null &

tbr tensorboard_root:
	$(MAKE) tb MDIR=. PORT=$(PORT)

tm trainmodel: labeled-data
	python3 summarization.py --mode=train \
		--model_root=model --data_path=training.articles \
		--batch_size=32 --log_rouge_interval=3600 --max_train_step=256000 $(ARGS)

cleanmodel:
	rm -rf model

train: labeled-data cleanmodel tensorboard trainmodel

ctrain: labeled-data tensorboard trainmodel

TRAINING_ROOT=running_center/$(TAG)-$(TIMESTAMP)/

tc trainingcenter: labeled-data
	mkdir -p $(TRAINING_ROOT)
	cp -p --parents *.py Makefile *.articles *.vocab */data.md5 $(TRAINING_ROOT)
	echo ARGS ?= $(ARGS) > $(TRAINING_ROOT)Makefile.config
	echo PORT ?= $(PORT) >> $(TRAINING_ROOT)Makefile.config
	$(MAKE) -C $(TRAINING_ROOT) train

decode:
	python3 summarization.py --mode=decode \
		--model_root=model --data_path=test-sample.articles \
		--batch_size=20 --beam_size=4 --enable_log2file=1 $(ARGS)

cdecode:
	watch -n 3600 $(MAKE) decode

naive: labeled-data
	python3 summarization.py --mode=naive --model_root=model --data_path=test-sample.articles $(ARGS)

cptb:
	@if [ '$(ID)' != '' ]; then \
		for i in $(ID); do \
			mkdir -p running_center/$(TAG)/_$$i; \
			philly-fs -cp -r //philly/eu2/ipgsrch/sys/jobs/application_$$i/models/output-model-path/tb/* running_center/$(TAG)/_$$i; \
		done; \
		echo; \
	else echo 'make cptb: please specify a valid ID'; \
	fi

trainv2:
	python3 dtitlev2.py --data_dir=data_dtitle/1002-training.dtitle --model_dir=model-1002 --vocab_file=data_dtitle/1002-vocab-16384 --param_set=base --train_steps=1000000 --steps_between_evals=10000 --batch_size=8 --max_length=64 --num_gpus=1 --enable_time_history=false

