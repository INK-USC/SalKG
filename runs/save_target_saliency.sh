# CONFIG='configs/qa/csqa/no_kg/bert_base_uncased__quadro-rtx-8000.ini'
# CKPT_PATH="/home/jiashu/FactSelection/SalKG/save/SAL-15/checkpoints/epoch=6-step=930.ckpt"
CONFIG="configs/qa/csqa/mhgrn/bert_base_uncased__quadro-rtx-8000.ini"
CKPT_PATH="/home/jiashu/FactSelection/SalKG/save/SAL-16/checkpoints/epoch=1-step=265.ckpt"
# coarse occl
echo "loading config ${CONFIG}; ckpt from ${CKPT_PATH}";
python main.py --config $CONFIG \
    --save_saliency \
    --saliency_source target \
    --saliency_mode coarse \
    --saliency_method occl \
    --ckpt_path $CKPT_PATH;
echo "done"