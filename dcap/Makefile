SHELL = /bin/bash
CPUS = $(shell nproc)
TIMESTAMP := $(shell date +'%Y%m%d-%H%M%S')
#export TF_CPP_MIN_LOG_LEVEL=1

-include Makefile.config
TAG ?= notag
DTAG ?= 0611-urltitle-30m-1k
MODEL_SIZE ?= trial
PORT ?= 45678
ROOT ?= .
MDIR ?= $(subst //,/,$(ROOT)/model)
DATA_DIR ?= $(subst //,/,$(ROOT)/data_dtitle)
TRAINING_ROOT ?= .

CODE_FILES = $(filter-out $(wildcard v1_*.py),$(wildcard Makefile *.py data_dtitle/Makefile data_dtitle/*.py))
DATA_FILES = $(DATA_DIR)/$(DTAG)-training.dtitle.tokenized.gz $(DATA_DIR)/$(DTAG)-test.dtitle.tokenized.gz $(DATA_DIR)/$(DTAG)-vocab.subwords $(DATA_DIR)/$(DTAG)-meta.log
DATA_CACHE_DIR = $(subst //,/,$(TRAINING_ROOT)/data_dtitle)
DATA_CACHE_FILES = $(subst $(DATA_DIR),$(DATA_CACHE_DIR),$(DATA_FILES))

#TRAINING_SCHEMA ?= Url:128,Language:4,LanguageAnchor:4,InjHdr_CDG_H:16,InjHdr_CDG_E:16,Wiki_Name:16,ODPTitle:16,CaptionAnchorText=>TargetTitle
TRAINING_SCHEMA ?= Url:128,Language:4,LanguageAnchor:4,InjHdr_CDG_H:32,InjHdr_CDG_E:32,CaptionAnchorText=>TargetTitle

all: check-params

.DELETE_ON_ERROR:

check-params:
	@echo TAG = $(TAG)
	@echo DTAG = $(DTAG)
	@echo MODEL_SIZE = $(MODEL_SIZE)
	@echo PORT = $(PORT)
	@echo ROOT = $(ROOT)
	@echo MDIR = $(MDIR)
	@echo DATA_DIR = $(DATA_DIR)
	@echo TRAINING_ROOT = $(TRAINING_ROOT)
	@echo DATA_CACHE_DIR = $(DATA_CACHE_DIR)
	@echo CODE_FILES = $(CODE_FILES)
	@echo DATA_FILES = $(DATA_FILES)
	@echo DATA_CACHE_FILES = $(DATA_CACHE_FILES)
	@echo TRAINING_SCHEMA = "$(TRAINING_SCHEMA)"
	@echo ARGS = $(ARGS)

ifneq ($(DATA_CACHE_DIR),$(DATA_DIR))
$(DATA_CACHE_DIR)/%: $(DATA_DIR)/%
	mkdir -p $(DATA_CACHE_DIR); cp -p $< $@
endif

tb: kill-tensorboard
	CUDA_VISIBLE_DEVICES= tensorboard --port $(PORT) --bind_all --window_title $(lastword $(subst /, ,$(CURDIR))) --logdir $(MDIR) 2>/dev/null &

ktb kill-tensorboard:
	pkill -f tensorboard\ --port\ $(PORT); sleep 1

tbr tensorboard_root:
	@MDIR=. $(MAKE) --no-print-directory tb

tm train-model: $(DATA_FILES)
	python3 dtitle.py --data_dir=$(DATA_DIR)/$(DTAG)-training.dtitle.tokenized.gz --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-vocab --param_set=$(MODEL_SIZE) --train_steps=640000 --steps_between_evals=5000 --batch_size=16 --num_gpus=-1 --noenable_time_history --enable_metrics_in_training --enable_tensorboard --validation_example_count=2048 --use_reformer=0 --training_schema="$(TRAINING_SCHEMA)" $(ARGS)

clean-training-env:
	@pkill -f tensorboard\ --port\ $(PORT); sleep 1
	rm -rf $(MDIR); sleep 1
	@$(MAKE) --no-print-directory tb

train: clean-training-env
	$(MAKE) train-model

TRAINING_DIR = $(subst //,/,$(TRAINING_ROOT)/running_center/$(TAG)-$(DTAG)-$(TIMESTAMP)/)

tc trainingcenter: $(DATA_CACHE_FILES)
ifeq (,$(wildcard $(TRAINING_DIR)Makefile.config))
	mkdir -p $(TRAINING_DIR)
	cp -p --parents $(CODE_FILES) $(TRAINING_DIR)
	cp -lp $(DATA_CACHE_FILES) $(TRAINING_DIR)data_dtitle
	$(MAKE) --no-print-directory check-params > $(TRAINING_DIR)Makefile.config
	$(MAKE) -C $(TRAINING_DIR) train
else
	$(MAKE) -C $(TRAINING_DIR) train-model
endif

eval: $(DATA_FILES)
	python3 dtitle.py --mode=eval --data_dir=$(DATA_DIR)/$(DTAG)-test.dtitle.tokenized.gz --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-vocab --param_set=$(MODEL_SIZE) --batch_size=64 --validation_example_count=8192 --num_gpus=-1 --enable_metrics_in_training --use_reformer=0 --training_schema="$(TRAINING_SCHEMA)" $(ARGS)

#predict:
#	python3 dtitle.py --mode=predict --data_dir=$(DATA_DIR)/$(DTAG)-test.dtitle.tokenized.gz --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-vocab --param_set=$(MODEL_SIZE) --batch_size=64 --num_gpus=-1 --use_reformer=0 --calc_rouge_scores=0 --test_num_hashes=8 --max_predict_count=1024 --dev_mode --prediction_details_file='#model_dir' --training_schema="$(TRAINING_SCHEMA)" $(ARGS)

predict:
	python3 dtitle.py --mode=predict-express --data_dir=$(DATA_DIR)/$(DTAG)-test.dtitle.tokenized.gz --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-vocab --param_set=$(MODEL_SIZE) --batch_size=64 --num_gpus=-1 --use_reformer=0 --calc_rouge_scores=0 --test_num_hashes=8 --max_predict_count=2048 --dev_mode --prediction_details_file='#model_dir' --training_schema="$(TRAINING_SCHEMA)" $(ARGS)

predict-cpu:
	CUDA_VISIBLE_DEVICES= $(MAKE) predict

#predict-express-cpu:
#	CUDA_VISIBLE_DEVICES= $(MAKE) predict-express

INPUT_DATA = /data/aether/gluo/19ca8a83-5361-4c13-9546-5156bf8076e9
PRED_SCHEMA=Url,Language,LanguageAnchor,InjHdr_CDG_H,InjHdr_CDG_E,Wiki_Name,ODPTitle,CaptionAnchorText,AHtmlTitle
PREDICTION = ./pred-unittest/
#usage: make prediction INPUT_DATA=<input_file> PREDICTION=<output_dir>

PRED_ABSPATH = $(abspath $(PREDICTION))

#prediction: $(INPUT_DATA)
#ifdef INPUT_DATA
#	$(info INPUT_DATA: $(INPUT_DATA))
#else
#	$(error INPUT_DATA is not specified)
#endif
#ifdef PREDICTION
#	$(info PREDICTION: $(PREDICTION))
#else
#	$(error PREDICTION is not specified)
#endif
#	mkdir -p $(PRED_ABSPATH)
#	$(MAKE) -C $(DATA_DIR) $(PRED_ABSPATH)/input-preprocessed.test.dtitle.gz $(PRED_ABSPATH)/input-preprocessed.test.dtitle.tokenized.gz TEST_DATA=$(abspath $(INPUT_DATA)) ARGS='--html_token_limit=1024' TAG=$(DTAG)
#	python3 dtitle.py --mode=predict --data_dir=$(PRED_ABSPATH)/input-preprocessed.test.dtitle.tokenized.gz --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-vocab --param_set=$(MODEL_SIZE) --num_gpus=-1 --use_reformer=0 --calc_rouge_scores=0 --dev_mode --attention_padding_strategy=nopadding --training_schema="$(TRAINING_SCHEMA)" --prediction_reference_file=$(PRED_ABSPATH)/input-preprocessed.test.dtitle.gz --prediction_details_file=$(PRED_ABSPATH)/prediction-details.txt --prediction_compact_file=$(PRED_ABSPATH)/prediction-compact.tsv $(ARGS)

prediction: $(INPUT_DATA)
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
	$(MAKE) $(PRED_ABSPATH)/data.split.done
	for fi in $(PRED_ABSPATH)/split-*.raw.gz; do $(MAKE) $${fi::-7}.predict; done
	head -n 1 $(PRED_ABSPATH)/split-000.predict > $(PRED_ABSPATH)/prediction-compact.tsv
	tail -q -n +2 $(PRED_ABSPATH)/split-*.predict >> $(PRED_ABSPATH)/prediction-compact.tsv
	gzip $(PRED_ABSPATH)/split-*.predict

$(PRED_ABSPATH)/data.split.done: $(INPUT_DATA)
	split -d -l1024000 -a3 --additional-suffix=.raw $< $(PRED_ABSPATH)/split-
	wc $(PRED_ABSPATH)/split-*.raw | tee $@
	gzip $(PRED_ABSPATH)/*.raw

$(PRED_ABSPATH)/%.predict: $(PRED_ABSPATH)/%.raw.gz
	$(MAKE) -C $(DATA_DIR) $(PRED_ABSPATH)/$*.test.dtitle.tokenized.gz TEST_DATA=$(abspath $<) ARGS='--input_schema=$(PRED_SCHEMA) --dtitle_schema=$(PRED_SCHEMA) --html_token_limit=0' TAG=$(DTAG)
	#$(MAKE) -C $(DATA_DIR) $(PRED_ABSPATH)/$*.test.dtitle.tokenized.gz TEST_DATA=$(abspath $<) ARGS='--input_schema=$(PRED_SCHEMA) --dtitle_schema=$(PRED_SCHEMA),TargetTitle --use_lower_case=0' TAG=$(DTAG)
	python3 dtitle.py --mode=predict-express --data_dir=$(PRED_ABSPATH)/$*.test.dtitle.tokenized.gz --model_dir=$(MDIR) --vocab_file=$(DATA_DIR)/$(DTAG)-vocab --param_set=$(MODEL_SIZE) --num_gpus=-1 --use_reformer=0 --calc_rouge_scores=0 --dev_mode --attention_padding_strategy=nopadding --training_schema="$(TRAINING_SCHEMA)" --prediction_compact_file=$@ --dtitle_data_schema=$(PRED_SCHEMA) $(ARGS) --batch_size=256

batch0:
	CUDA_VISIBLE_DEVICES=0 $(MAKE) batch
batch1:
	CUDA_VISIBLE_DEVICES=1 $(MAKE) batch
batch2:
	CUDA_VISIBLE_DEVICES=2 $(MAKE) batch
batch3:
	CUDA_VISIBLE_DEVICES=3 $(MAKE) batch

batch:
	$(MAKE) eval ARGS='--use_full_attention_in_reformer $(ARGS)'
	$(MAKE) eval ARGS='--test_num_hashes=1 $(ARGS)'
	$(MAKE) eval ARGS='--test_num_hashes=2 $(ARGS)'
	$(MAKE) eval ARGS='--test_num_hashes=4 $(ARGS)'
	$(MAKE) eval ARGS='--test_num_hashes=8 $(ARGS)'
	$(MAKE) eval ARGS='--test_num_hashes=16 $(ARGS)'
	$(MAKE) eval ARGS='--test_num_hashes=32 $(ARGS)'
	$(MAKE) eval ARGS='--test_num_hashes=64 $(ARGS)'
