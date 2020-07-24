#!/bin/bash

dtitle_raw_path=/relevance2-nfs/gluo/shared/deepsum-demo/smoketest-dataset-200k.tsv.7z
test_data=/relevance2-nfs/gluo/shared/deepsum-demo/smoketest-dataset-val-10k.tsv
tag=end2end-v20200623
dtag=smoketest

echo ----- run somke test for deepsum-title -----
echo working_dir = $PWD
echo DTITLE_RAW = $dtitle_raw_path
echo TAG = $tag
echo DTAG = $dtag
echo test INPUT_DATA = $test_data
read -n 1 -r -s -p $'Press any key to continue...\n'
echo

set -x

cd data_dtitle
rm -f data-v3-for-$dtag.*
echo ----- preprocess / build vocab -----
make data-v3-for-$dtag.subwords DTITLE_RAW=$dtitle_raw_path ARGS='--max_corpus_chars=0.032'
if [ $? -ne 0 ]; then exit; fi

rm -f $dtag-*
echo ----- preprocess / tokeize training data -----
make TAG=$dtag DTITLE_RAW=$dtitle_raw_path VOCAB_FILE=data-v3-for-$dtag
if [ $? -ne 0 ]; then exit; fi

cd ..
make ktb
rm -rf running_center/$tag-$dtag-*
echo ----- training -----
make tc TAG=$tag DTAG=$dtag TRAINING_SCHEMA='Url:128=>TargetTitle' ARGS='--max_input_length=128 --batch_size=128 --train_steps=300 --steps_between_evals=100'
if [ $? -ne 0 ]; then exit; fi

cd running_center/$tag-$dtag-*
echo ----- inference / predict on validation set -----
make predict
if [ $? -ne 0 ]; then exit; fi

rm -rf pred-unittest-prediction
echo ----- inference / prediction -----
make prediction INPUT_DATA=$test_data PREDICTION=pred-unittest-prediction
if [ $? -ne 0 ]; then exit; fi

rm -rf pred-unittest-express-prediction
echo ----- inference / express-prediction -----
make express-prediction INPUT_DATA=$test_data PREDICTION=pred-unittest-express-prediction
if [ $? -ne 0 ]; then exit; fi

echo ----- clean intermediate files in data_dtitle -----
mv ../../data_dtitle/data-v3-for-$dtag.* data_dtitle
mv ../../data_dtitle/$dtag-* data_dtitle
rm -rf ../../data_dtitle/$dtag-*
