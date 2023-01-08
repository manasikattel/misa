export DYLD_LIBRARY_PATH=~/Downloads/elastix-5.0.1-mac/lib:$DYLD_LIBRARY_PATH 

for VARIABLE in $(ls "data/project_data/Training_Set") 
do

    #train-val
    mkdir data/out/Training_Validation_preprocessed
    mkdir data/out/Training_Validation_preprocessed/reg_p1_$VARIABLE

    #test
    # mkdir data/out/Test_preprocessed
    # mkdir data/out/Test_preprocessed/reg_p1_$VARIABLE

    IMG=$VARIABLE/$VARIABLE"_preprocessed_histmatched.nii.gz"
    # IMG=$VARIABLE/$VARIABLE".nii.gz"


    #Training
    elastix -m mean_image_bspline.nii -f data/project_data/Training_Set/$IMG  -out data/out/Training_Validation_preprocessed/reg_p1_$VARIABLE -p atlas/par_affine.txt -p  atlas/par_bspline64.txt
    
    #Test
    # elastix -m mean_image_bspline.nii -f data/project_data/Test_Set/$IMG  -out data/out/Test_preprocessed/reg_p1_$VARIABLE -p atlas/par_affine.txt -p  atlas/par_bspline64.txt



done
