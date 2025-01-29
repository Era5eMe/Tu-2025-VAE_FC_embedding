#!/bin/bash
SECONDS=0

# MSC
subjs=(01 02 03 04 05 06 07 08 09 10)
parcelcount=(602 567 620 616 633 580 628 710 613 649)
zdim=2
checkpoint="./VAE_Model/Checkpoint/checkpoint49_2024-03-28_Zdim_2_Vae-beta_20.0_Lr_0.0001_Batch-size_128_washu120_subsample10_train100_val10.pth.tar"
for i in "${!subjs[@]}"; do 
    subj="${subjs[$i]}" 
    curr_parcel_count="${parcelcount[$i]}"
    echo $curr_parcel_count
    namestr="sub-MSC${subj}_sub-MSC${subj}Parcel"

    python3 VAE_inference_example.py --data-path ./data/$namestr --zdim $zdim \
        --resume  "${checkpoint}" \
        --z-path './result/latent/'$namestr'_Zdim'$zdim --mode 'encode' --batch-size $curr_parcel_count

    echo "The command took $SECONDS seconds." 
done