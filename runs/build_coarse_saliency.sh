nokg=$1
kg=$2
# eg. mhgrn
graph=$3
text="bert-base-uncased"
savedir="./save/"

dataset="csqa"
method="occl"
inhouse=""
if [[ $dataset == "csqa" ]]; then
    inhouse="--inhouse"
fi;

echo -e "building $method coarse dataset"
percentage="100"
echo -e "$nokg + $kg with $graph ($percentage %)"
for s in train valid test; do
    echo $s
    python scripts/build_coarse_saliency.py --split $s --text-encoder $text --graph-encoder $graph \
    --dataset $dataset --saliency-method $method --qa-kg-dir ${savedir}/SAL-${kg}/saliency $inhouse --target-type cls \
    --num-classes 2 --train-percentage $percentage --qa-no-kg-dir ${savedir}/SAL-${nokg}/saliency \
    --threshold 0.01 --root-dir ./data \
    --kg-exp $kg --no-kg-exp $nokg
done;