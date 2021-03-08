HOME_DIR=~/3rd
cd ${HOME_DIR}

export embed_dir=data/bert-embedding
export data_dir=data/




lr=2e-5
ep=50
b=100
s=256
wp=0.1
run=1

CUDA_VISIBLE_DEVICES=0 python codes/main.py \
  --data_dir ${data_dir} \
  --bert_dir ${embed_dir} \
  --bert_file bert-base-cased \
  --do_train \
  --learning_rate ${lr} \
  --epoch ${ep} \
  --use_cuda \
  --batch_size ${b} \
  --max_seq_length ${s} \
  --warmup_propotion ${wp} \
  --output_dir results/ner_output_lr${lr}_ep${ep}_b${b}_s${s}_wp${wp}_run${run}/ \
  --shell_print shell \
  --suffix last \
  --multi_gpu \
  --do_eval
