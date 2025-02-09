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
    --instance_num 3 \
    --context_num 3 \
    --metadata_wise \
    --instance_wise \
    --prompt_engineering
    # --use_chain_of_thoughts \

    # --data_parsing \
    
done

