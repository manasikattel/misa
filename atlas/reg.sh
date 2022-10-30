for VARIABLE in $(ls data/training-images) 
do
    mkdir data/out/reg_affine_$VARIABLE
    mkdir data/out/reg_bspline_$VARIABLE
    mkdir data/out/reg_rigid_$VARIABLE

    MASKPATH="${VARIABLE/.nii/_1C.nii}"	
    # echo "data/training-images/$VARIABLE"
    # echo "$MASKPATH"
    
    ~/Downloads/elastix-5.0.0-mac/elastix -f data/training-images/1000.nii -m data/training-images/$VARIABLE -fmask data/training-mask/$MASKPATH -out data/out/reg_affine_$VARIABLE -p par_affine.txt 
    ~/Downloads/elastix-5.0.0-mac/elastix -f data/training-images/1000.nii -m data/training-images/$VARIABLE -fmask data/training-mask/$MASKPATH -out data/out/reg_bspline_$VARIABLE -p par_bspline64.txt 
    ~/Downloads/elastix-5.0.0-mac/elastix -f data/training-images/1000.nii -m data/training-images/$VARIABLE -fmask data/training-mask/$MASKPATH -out data/out/reg_rigid_$VARIABLE -p par_rigid.txt

done
