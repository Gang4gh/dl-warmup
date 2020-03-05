SHELL = /bin/bash
CPUS = $(shell nproc)
TIMESTAMP := $(shell date +'%Y%m%d-%H%M%S')

-include Makefile.config
TAG ?= notag
PORT ?= 2345
ROOT ?= .
MDIR ?= $(subst //,/,$(ROOT)/model)
DATA_DIR ?= $(subst //,/,$(ROOT)/data_dtitle)
DTAG ?= 1217-noTH2s-r0.1
VOCAB ?= vocab-8192
MODEL_SIZE ?= trial
DATA_FILES = $(DATA_DIR)/$(DTAG)-training.tfrecord.gz $(DATA_DIR)/$(DTAG)-test.dtitle $(DATA_DIR)/$(DTAG)-$(VOCAB).subwords $(DATA_DIR)/$(DTAG)-meta.log
CODE_FILES = $(filter-out $(wildcard v1_*.py),$(wildcard Makefile *.py $(DATA_DIR)/Makefile $(DATA_DIR)/*.py))

all: check-params

clean:
	#rm -f */data.md5 *.articles *.vocab

.DELETE_ON_ERROR:

check-params:
	@echo TAG ?= $(TAG)
	@echo PORT ?= $(PORT)
	@echo ROOT ?= $(ROOT)
	@echo MDIR ?= $(MDIR)
	@echo DATA_DIR ?= $(DATA_DIR)
	@echo DTAG ?= $(DTAG)
	@echo VOCAB ?= $(VOCAB)
	@echo MODEL_SIZE ?= $(MODEL_SIZE)
	@echo DATA_FILES = $(DATA_FILES)
	@echo CODE_FILES = $(CODE_FILES)
	@echo ARGS ?= $(ARGS)

cleanmodel:
	rm -rf model

tb tensorboard:
	pkill -f tensorboard\ --port\ $(PORT); sleep 1
	CUDA_VISIBLE_DEVICES= tensorboard --port $(PORT) --bind_all --window_title $(lastword $(subst /, ,$(CURDIR))) --logdir $(MDIR) 2>/dev/null &

tbr tensorboard_root:
	$(MAKE) tb MDIR=. PORT=$(PORT)

%-training.tfrecord.gz: %-training.dtitle.gz %-$(VOCAB).subwords
	CUDA_VISIBLE_DEVICES= python3 dtitle.py --mode=train-prep --data_dir=$< --vocab_file=$(subst -training.dtitle.gz,,$<)-$(VOCAB) --batch_size=16 --max_input_length=1024 --max_target_length=48 --num_gpus=0 $(ARGS)

tm trainmodel: $(DATA_FILES)
	python3 dtitle.py --data_dir=$(DATA_DIR)/$(DTAG)-training.tfrecord.gz --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-$(VOCAB) --param_set=$(MODEL_SIZE) --train_steps=1000000 --steps_between_evals=5000 --batch_size=16 --max_input_length=1024 --max_target_length=48 --num_gpus=-1 --noenable_time_history --enable_metrics_in_training --enable_tensorboard --validation_example_count=1024 --use_reformer $(ARGS)

train-debug: $(DATA_FILES)
	TF_FORCE_GPU_ALLOW_GROWTH=true $(MAKE) train

train: cleanmodel tensorboard trainmodel

TRAINING_ROOT = running_center/$(TAG)-$(DTAG)-$(TIMESTAMP)/

tc trainingcenter: $(DATA_FILES)
	mkdir -p $(TRAINING_ROOT)
	cp -p --parents $(CODE_FILES) $(TRAINING_ROOT)
	cp -lp --parents $(DATA_FILES) $(TRAINING_ROOT)
	$(MAKE) --no-print-directory check-params > $(TRAINING_ROOT)Makefile.config
	$(MAKE) -C $(TRAINING_ROOT) train

eval:
	python3 dtitle.py --mode=eval    --data_dir=$(DATA_DIR)/$(DTAG)-test.dtitle --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-$(VOCAB) --param_set=$(MODEL_SIZE) --batch_size=16 --num_gpus=-1 $(ARGS)

predict:
	python3 dtitle.py --mode=predict --data_dir=$(DATA_DIR)/$(DTAG)-test.dtitle --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-$(VOCAB) --param_set=$(MODEL_SIZE) --batch_size=64 --num_gpus=-1 $(ARGS)

predict-cpu:
	CUDA_VISIBLE_DEVICES= $(MAKE) predict

#cdecode:
#	watch -n 3600 $(MAKE) decode

#INPUT_DATA = $(HOME)/CaptionData/October_Scraping_joinedData-1204.tsv
INPUT_DATA = /data/aether/gluo/63a274d4-33be-4b54-b52d-28487a80465a
PREDICTION = ../pred-unittest/
PRED_ABSPATH = $(abspath $(PREDICTION))
#usage: make predict-test INPUT_DATA=<input_file> PREDICTION=<output_dir>
prediction: $(INPUT_DATA) $(CODE_FILES)
ifdef INPUT_DATA
	$(info INPUT_DATA: $(INPUT_DATA))
else
	$(error INPUT_DATA is not specified)
endif
ifdef PREDICTION
	$(info PREDICTION: $(PREDICTION))
else
	$(error PREDICTION is not specified)
endif
	mkdir -p $(PRED_ABSPATH)
	$(MAKE) -C $(DATA_DIR) $(PRED_ABSPATH)/input-preprocessed.test.dtitle TEST_DATA=$(abspath $(INPUT_DATA)) ARGS=
	#$(MAKE) -C $(DATA_DIR) $(PRED_ABSPATH)/input-preprocessed.test.dtitle TEST_DATA=$(abspath $(INPUT_DATA)) ARGS='--check_enoughtokens --check_exactmatch'
	python3 dtitle.py --mode=predict --data_dir=$(PRED_ABSPATH)/input-preprocessed.test.dtitle --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-$(VOCAB) --param_set=$(MODEL_SIZE) --batch_size=64 --num_gpus=-1 --calc_rouge_scores=0 --prediction_reference_file=$(INPUT_DATA) --prediction_details_file=$(PRED_ABSPATH)/prediction-details.txt --prediction_compact_file=$(PRED_ABSPATH)/prediction-compact.tsv $(ARGS)

DLTS_DIRS=$(wildcard running_center/dlts*)
sync-all:
	for fd in $(DLTS_DIRS); do cp -p --parents $(CODE_FILES) $$fd; done

pred-all: $(patsubst running_center%,running_center/pred-$(TIMESTAMP)%, $(DLTS_DIRS))

running_center/pred-$(TIMESTAMP)/%: running_center/%
	@echo ===== start of '$<' =====
	$(MAKE) -C $< prediction INPUT_DATA=$(abspath $(INPUT_DATA)) PREDICTION=$(abspath $@)-1k ARGS='--max_predict_count=1024'
	#$(MAKE) -C $< prediction INPUT_DATA=$(abspath $(INPUT_DATA)) PREDICTION=$(abspath $@)
