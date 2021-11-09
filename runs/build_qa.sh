cd ../;


for split in train valid test; do
        python scripts/build_qa.py \
            --model-name bert-base-uncased \
            --dataset csqa --inhouse \
            --split $split \
            --root-dir ../d;

        # do fine occl
        python scripts/build_qa.py \
            --model-name bert-base-uncased \
            --dataset csqa --inhouse --fine-occl \
            --split $split \
            --root-dir ../d;
done;