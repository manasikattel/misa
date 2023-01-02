function [T] = dice_calc(file_name)
    D = 'data';
    S = dir(fullfile(D,'*'));
    N = setdiff({S([S.isdir]).name},{'.','..'});
    T1_GM_dice=[];
    T1_WM_dice=[];
    T1_CSF_dice=[];
    FLAIR_GM_dice=[];
    FLAIR_WM_dice=[];
    FLAIR_CSF_dice=[];
    for ii = 1:numel(N)
        subfolder_name=fullfile(D,N{ii});
        T1_nii_filename = fullfile(subfolder_name,'T1.nii'); % improve by specifying the file extension.
        T2Falir_nii_filename = fullfile(subfolder_name,'T2_FLAIR.nii'); % improve by specifying the file extension.
        GT=niftiread(fullfile(subfolder_name,'LabelsForTesting.nii'));
        
        class_GM_T1= double(niftiread(fullfile(subfolder_name,'c1T1.nii')))/double(255);
        class_WM_T1= double(niftiread(fullfile(subfolder_name,'c2T1.nii')))/double(255);
        class_CSF_T1= double(niftiread(fullfile(subfolder_name,'c3T1.nii')))/double(255);
        class_B_T1= double(niftiread(fullfile(subfolder_name,'c4T1.nii')))/double(255);
        class_ST_T1= double(niftiread(fullfile(subfolder_name,'c5T1.nii')))/double(255);
        class_BG_T1= double(niftiread(fullfile(subfolder_name,'c6T1.nii')))/double(255);
        
        [mx_T1,pmx_T1]=max(cat(4,class_CSF_T1,class_GM_T1,class_WM_T1,class_B_T1,class_ST_T1,class_BG_T1),[],4);
    
        class_GM_Flair= double(niftiread(fullfile(subfolder_name,'c1T2_FLAIR.nii')))/double(255);
        class_WM_Flair= double(niftiread(fullfile(subfolder_name,'c2T2_FLAIR.nii')))/double(255);
        class_CSF_Flair= double(niftiread(fullfile(subfolder_name,'c3T2_FLAIR.nii')))/double(255);    
        class_B_Flair= double(niftiread(fullfile(subfolder_name,'c4T2_FLAIR.nii')))/double(255);
        class_ST_Flair= double(niftiread(fullfile(subfolder_name,'c5T2_FLAIR.nii')))/double(255);
        class_BG_Flair= double(niftiread(fullfile(subfolder_name,'c6T2_FLAIR.nii')))/double(255);
        
        [mx_flair,pmx_flair]=max(cat(4,class_CSF_Flair,class_GM_Flair,class_WM_Flair,class_B_Flair,class_ST_Flair,class_BG_Flair),[],4);

        GT_CSF = (GT==1);
        GT_GM = (GT==2);
        GT_WM = (GT==3);
    
        t1_CSF = (pmx_T1==1);
        t1_GM = (pmx_T1==2);
        t1_WM = (pmx_T1==3);
        
        flair_CSF = (pmx_flair==1);
        flair_GM = (pmx_flair==2);
        flair_WM = (pmx_flair==3);
            
        disp('GM T1:'); 
        T1_GM_dice(end+1)=dice(t1_GM,GT_GM);
        disp(T1_GM_dice(end));
        disp('WM T1:');
        T1_WM_dice(end+1)= dice(t1_WM,GT_WM);
        disp(T1_WM_dice(end));
        disp('CSF T1:');
        T1_CSF_dice(end+1)= dice(t1_CSF,GT_CSF);
        disp(T1_CSF_dice(end));
        
        disp('GM Flair:'); 
        FLAIR_GM_dice(end+1)= dice(flair_GM,GT_GM);
        disp(FLAIR_GM_dice(end));
        disp('WM Flair:');
        FLAIR_WM_dice(end+1)=dice(flair_WM,GT_WM);
        disp(FLAIR_WM_dice(end));
        disp('CSF Flair:');
        FLAIR_CSF_dice(end+1)=dice(flair_CSF,GT_CSF);
        disp(FLAIR_CSF_dice(end));
     
    end
    
    r_mat=cat(1,T1_GM_dice,T1_WM_dice,T1_CSF_dice,FLAIR_GM_dice,FLAIR_WM_dice,FLAIR_CSF_dice)';
    mean_mat=cat(1,mean(T1_GM_dice),mean(T1_WM_dice),mean(T1_CSF_dice),mean(FLAIR_GM_dice),mean(FLAIR_WM_dice),mean(FLAIR_CSF_dice))';
    std_mat=cat(1,std(T1_GM_dice),std(T1_WM_dice),std(T1_CSF_dice),std(FLAIR_GM_dice),std(FLAIR_WM_dice),std(FLAIR_CSF_dice))';
    
    r_mat=cat(1,T1_GM_dice,T1_WM_dice,T1_CSF_dice)';
    mean_mat=cat(1,mean(T1_GM_dice),mean(T1_WM_dice),mean(T1_CSF_dice))';
    std_mat=cat(1,std(T1_GM_dice),std(T1_WM_dice),std(T1_CSF_dice))';
        
    final_mat=cat(1,r_mat,mean_mat,std_mat);
    
    T = array2table(final_mat);
    T.Properties.VariableNames(1:6) = {'T1_GM','T1_WM','T1_CSF','FLAIR_GM','FLAIR_WM','FLAIR_CSF'};
    
    writetable(T,file_name)
    

end


