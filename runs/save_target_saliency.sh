cd ../;

# CONFIG='configs/qa/csqa/no_kg/bert_base_uncased__quadro-rtx-8000.ini'
CONFIG="configs/qa/csqa/mhgrn/bert_base_uncased__quadro-rtx-8000.ini"
# CKPT_PATH="/home/jiashu/FactSelection/save2/SAL-9/checkpoints/epoch=6-step=930.ckpt"
CKPT_PATH="/home/jiashu/FactSelection/save2/SAL-12/checkpoints/epoch=1-step=265.ckpt"
# coarse occl
echo "loading config ${CONFIG}; ckpt from ${CKPT_PATH}";
python main.py --config $CONFIG \
    --save_saliency \
    --saliency_source target \
    --saliency_mode coarse \
    --saliency_method occl \
    --ckpt_path $CKPT_PATH;
echo "done"