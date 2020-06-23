set -x
dtitle_raw_path=/relevance2-nfs/gluo/shared/retroindex/smalltest-dataset-200k.tsv.7z
tag=smoketest

echo run somketest for deepsum-title
echo working_dir = $PWD
echo DTITLE_RAW = $dtitle_raw_path
echo TAG = $tag
read -n 1 -r -s -p $'Press any key to continue...\n'
echo

echo ----- preprocess / build vocab -----
cd data_dtitle
rm -f data-v3-vocab-24gb-4096.subwords
make data-v3-vocab-24gb-4096.subwords DTITLE_RAW=$dtitle_raw_path ARGS='--max_corpus_chars=0.25'

echo ----- preprocess / tokeize training data -----
rm -f $tag-*
make TAG=$tag DTITLE_RAW=$dtitle_raw_path

echo ----- training -----
cd ..
make train DTAG=$tag TRAINING_SCHEMA='Url:128=>TargetTitle' ARGS='--max_input_length=128 --batch_size=128 --train_steps=1000 --steps_between_evals=500'

echo ----- inference / predict -----
make predict

echo ----- inference / prediction -----
make prediction

echo ----- inference / express-prediction -----
make express-prediction
