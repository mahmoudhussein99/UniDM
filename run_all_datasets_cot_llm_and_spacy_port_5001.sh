#SBATCH --export=NONE
# BATCH=$1
# LR=$2
# TEMP=$3
# SIZE=$4
# PRODUCT=$5
# AUG=$6
conda activate unidm-env

DATASETS=("WDC-Computers" )
# DATASETS=("WDC-Shoes" "WDC-Watches" "WDC-Cameras" "MusicBrainz20k" "DBLP-GoogleScholar")
# DATASETS=("WDC-Shoes" )
# "DBLP-GoogleScholar" "MusicBrainz20k"
for DATASET in "${DATASETS[@]}"
do
    export DATASET=$DATASET
    python undersample_test.py
    python inference.py \
    --use_local_model \
    --data_dir dataset/datasets/entity_matching/structured/$DATASET \
    --task entity_resolution \
    --local_model_port 5001 \
    --clean_text \
    --seed 42 \
    --instance_num 3 \
    --context_num 3 \
    --metadata_wise \
    --instance_wise \
    --use_chain_of_thoughts_llm \
    --prompt_engineering
    # --data_parsing \
    # --use_chain_of_thoughts_experimental \
    # --max_tokens 100 \
    # --use_chain_of_thoughts_spacy \

    
done

