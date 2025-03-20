#!/bin/bash
SECONDS=0

# Lynch Priors
namestr="Lynch2024_45subj_Prior_20NetsParcel" 
zdim=2
checkpoint="./VAE_Model/Checkpoint/checkpoint49_2024-03-28_Zdim_2_Vae-beta_20.0_Lr_0.0001_Batch-size_128_washu120_subsample10_train100_val10.pth.tar"
curr_parcel_count = 20 # Not a big deal if you mess this up, this just tells the .mat file to save every n data. I set it to 20 becausee there are 20 images in the h5 file.
python3 VAE_inference_example.py --data-path ./data/$namestr --zdim $zdim \
    --resume  "${checkpoint}" \
    --z-path './result/latent/'$namestr'_Zdim'$zdim --mode 'encode' --batch-size $curr_parcel_count
echo "The command took $SECONDS seconds." 

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