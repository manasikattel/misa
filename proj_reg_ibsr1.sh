export DYLD_LIBRARY_PATH=~/Downloads/elastix-5.0.1-mac/lib:$DYLD_LIBRARY_PATH 
for VARIABLE in $(ls "data/project_data/Training_Set") 
do
    mkdir data/output
    mkdir data/output/reg_bspline_$VARIABLE

    # switch between preprocessed and unpreprocessed image here
    IMG=$VARIABLE/$VARIABLE"_preprocessed_histmatched.nii.gz"
    # IMG=$VARIABLE/$VARIABLE".nii.gz"


    elastix -f data/project_data/Training_Set/IBSR_01/IBSR_01.nii.gz -m data/project_data/Test_Set/$IMG -fmask data/project_data/Training_Set/IBSR_01/IBSR_01_seg.nii.gz -out data/out/test/reg_$VARIABLE -p atlas/par_affine.txt -p  atlas/par_bspline64.txt

done
