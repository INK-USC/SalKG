cd ../;

RUN=$1

if [[ $RUN == "mhgrn" ]]; then
    echo "RUNING MHGRN";
    python main.py --config configs/qa/csqa/mhgrn/bert_base_uncased__quadro-rtx-8000.ini \
        --text_lr 0.0001 --graph_lr 0.001 \
        --freeze_epochs 0 --weight_decay 0.02 \
        --train_batch_size 32 --eval_batch_size 32 --accumulate_grad_batches 2 \
        --seed 2 --train_percentage 100 \
        --encoder_pooler cls --text_encoder_head bos_token_mlp --data ../d 

elif [[ $RUN == "nokg" ]]; then
    echo "RUNING NO KG";
    python main.py --config configs/qa/csqa/no_kg/bert_base_uncased__quadro-rtx-8000.ini \
        --text_lr 0.00002 --cls_lr 0.001 \
        --freeze_epochs 0 --weight_decay 0 \
        --train_batch_size 32 --eval_batch_size 32 --accumulate_grad_batches 2 \
        --seed 1 --train_percentage 100 \
        --encoder_pooler cls --text_encoder_head bos_token_mlp

elif [[ $RUN == "pathgen" ]]; then
    echo "RUNING PATHGEN";
    python main.py --config configs/qa/csqa/pathgen/bert_base_uncased__quadro-rtx-8000.ini \
        --text_lr 0.0001 --graph_lr 0.005 \
        --freeze_epochs 0 --weight_decay 0.01 \
        --train_batch_size 32 --eval_batch_size 32 --accumulate_grad_batches 2 \
        --seed 1 --train_percentage 100 \
        --encoder_pooler cls --text_encoder_head bos_token_mlp

else
    echo "RUNING RN";
    python main.py --config configs/qa/csqa/rn/bert_base_uncased__quadro-rtx-8000.ini \
        --text_lr 0.0001 --graph_lr 0.007 \
        --freeze_epochs 0 --weight_decay 0.002 \
        --train_batch_size 32 --eval_batch_size 32 --accumulate_grad_batches 2 \
        --seed 2 --train_percentage 100 \
        --encoder_pooler cls --text_encoder_head bos_token_mlp
fi;
