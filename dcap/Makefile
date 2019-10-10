SHELL = /bin/bash
CPUS = $(shell nproc)
TIMESTAMP = $(shell date +'%Y%m%d-%H%M%S')

-include Makefile.config
DATASOURCES ?= CapTitle #CNN DailyMail
TAG ?= notag
PORT ?= 2345
MDIR ?= model

all:
	#python lt.py --mode=train

clean:
	#rm -f */data.md5 *.articles *.vocab

cleanall: clean
	#for ds in $(DATASOURCES); do $(MAKE) -C $$ds clean; done

.DELETE_ON_ERROR:


# data pre-processing

# training/test actions
tb tensorboard:
	pkill -f tensorboard\ --port\ $(PORT); sleep 1
	CUDA_VISIBLE_DEVICES= tensorboard --port $(PORT) --window_title $(lastword $(subst /, ,$(CURDIR))) --logdir $(MDIR) 2>/dev/null &

tbr tensorboard_root:
	$(MAKE) tb MDIR=. PORT=$(PORT)

cdecode:
	watch -n 3600 $(MAKE) decode

cptb:
	@if [ '$(ID)' != '' ]; then \
		for i in $(ID); do \
			mkdir -p running_center/$(TAG)/_$$i; \
			philly-fs -cp -r //philly/eu2/ipgsrch/sys/jobs/application_$$i/models/output-model-path/tb/* running_center/$(TAG)/_$$i; \
		done; \
		echo; \
	else echo 'make cptb: please specify a valid ID'; \
	fi

DATA_DIR ?= data_dtitle
DTAG ?= d1008-
VOCAB ?= vocab-16384
DATA_FILES = $(DATA_DIR)/$(DTAG)training.dtitle $(DATA_DIR)/$(DTAG)test.dtitle $(DATA_DIR)/$(DTAG)$(VOCAB).subwords
MODEL_FILES = $(filter-out $(wildcard v1_*.py),$(wildcard *.py))

check-params:
	@echo DATA_FILES = $(DATA_FILES)
	@echo MODEL_FILES = $(MODEL_FILES)

cleanmodel:
	rm -rf model

tm trainmodel: $(DATA_FILES)
	python3 dtitle.py --data_dir=$(DATA_DIR)/$(DTAG)training.dtitle --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)$(VOCAB) --param_set=base --train_steps=1000000 --steps_between_evals=10000 --batch_size=8 --max_length=64 --num_gpus=1 --enable_time_history=false $(ARGS)

train: cleanmodel trainmodel

TRAINING_ROOT=running_center/$(TAG)-$(TIMESTAMP)/

tc trainingcenter: $(DATA_FILES)
	mkdir -p $(TRAINING_ROOT)
	cp -p Makefile $(MODEL_FILES) $(TRAINING_ROOT)
	cp -lp --parents $(DATA_FILES) $(TRAINING_ROOT)
	echo ARGS ?= $(ARGS) > $(TRAINING_ROOT)Makefile.config
	$(MAKE) -C $(TRAINING_ROOT) train

predict:
	python3 dtitle.py --mode=predict --data_dir=$(DATA_DIR)/$(DTAG)test.dtitle --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)$(VOCAB) --param_set=base --batch_size=64 --num_gpus=1 $(ARGS)

