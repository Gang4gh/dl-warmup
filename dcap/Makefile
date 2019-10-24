SHELL = /bin/bash
CPUS = $(shell nproc)
TIMESTAMP = $(shell date +'%Y%m%d-%H%M%S')

-include Makefile.config
TAG ?= notag
PORT ?= 2345
MDIR ?= model
DATA_DIR ?= data_dtitle
DTAG ?= 1022-noTH-exact
VOCAB ?= vocab-16384
MODEL_SIZE ?= trial
DATA_FILES = $(DATA_DIR)/$(DTAG)-training.dtitle $(DATA_DIR)/$(DTAG)-test.dtitle $(DATA_DIR)/$(DTAG)-$(VOCAB).subwords
MODEL_FILES = $(filter-out $(wildcard v1_*.py),$(wildcard *.py))

all: check-params

clean:
	#rm -f */data.md5 *.articles *.vocab

.DELETE_ON_ERROR:

check-params:
	@echo TAG ?= $(TAG)
	@echo PORT ?= $(PORT)
	@echo MDIR ?= $(MDIR)
	@echo DATA_DIR ?= $(DATA_DIR)
	@echo DTAG ?= $(DTAG)
	@echo VOCAB ?= $(VOCAB)
	@echo MODEL_SIZE ?= $(MODEL_SIZE)
	@echo DATA_FILES = $(DATA_FILES)
	@echo MODEL_FILES = $(MODEL_FILES)
	@echo ARGS ?= $(ARGS)

cleanmodel:
	rm -rf model

tb tensorboard:
	pkill -f tensorboard\ --port\ $(PORT); sleep 1
	CUDA_VISIBLE_DEVICES= tensorboard --port $(PORT) --bind_all --window_title $(lastword $(subst /, ,$(CURDIR))) --logdir $(MDIR) 2>/dev/null &

tbr tensorboard_root:
	$(MAKE) tb MDIR=. PORT=$(PORT)

tm trainmodel: $(DATA_FILES)
	python3 dtitle.py --data_dir=$(DATA_DIR)/$(DTAG)-training.dtitle --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-$(VOCAB) --param_set=$(MODEL_SIZE) --train_steps=1000000 --steps_between_evals=5000 --batch_size=16 --max_input_length=1024 --max_target_length=48 --num_gpus=-1 --noenable_time_history --enable_metrics_in_training  --enable_tensorboard --validation_example_count=1024 $(ARGS)

train: cleanmodel tensorboard trainmodel

TRAINING_ROOT=running_center/$(TAG)-$(DTAG)-$(TIMESTAMP)/

tc trainingcenter: $(DATA_FILES)
	mkdir -p $(TRAINING_ROOT)
	cp -p Makefile $(MODEL_FILES) $(TRAINING_ROOT)
	cp -lp --parents $(DATA_FILES) $(TRAINING_ROOT)
	$(MAKE) --no-print-directory check-params > $(TRAINING_ROOT)Makefile.config
	$(MAKE) -C $(TRAINING_ROOT) train

eval:
	python3 dtitle.py --mode=eval --data_dir=$(DATA_DIR)/$(DTAG)-test.dtitle --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-$(VOCAB) --param_set=$(MODEL_SIZE) --batch_size=16 --num_gpus=-1 $(ARGS)

predict:
	CUDA_VISIBLE_DEVICES= python3 dtitle.py --mode=predict --data_dir=$(DATA_DIR)/$(DTAG)-test.dtitle --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-$(VOCAB) --param_set=$(MODEL_SIZE) --batch_size=16 --num_gpus=-1 $(ARGS)

predict-cpu:
	CUDA_VISIBLE_DEVICES= $(MAKE) predict

#cdecode:
#	watch -n 3600 $(MAKE) decode
