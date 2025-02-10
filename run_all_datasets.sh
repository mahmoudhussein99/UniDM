#SBATCH --export=NONE
# BATCH=$1
# LR=$2
# TEMP=$3
# SIZE=$4
# PRODUCT=$5
# AUG=$6
conda activate unidm-env

DATASETS=("WDC-Computers" )
# "WDC-Computers" "WDC-Watches" "WDC-Cameras" "MusicBrainz20k" "DBLP-GoogleScholar")
# "DBLP-GoogleScholar" "MusicBrainz20k"
for DATASET in "${DATASETS[@]}"
do
    python inference.py \
    --use_local_model \
    --data_dir dataset/datasets/entity_matching/structured/$DATASET \
    --task entity_resolution \
    --instance_num 1 \
    --context_num 1 \
    --metadata_wise \
    --instance_wise \
    --use_chain_of_thoughts_llm \
    --prompt_engineering
    # --data_parsing \
    # --max_tokens 100 \
    # --use_chain_of_thoughts_spacy \

    
done

