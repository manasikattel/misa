%%addpath C:\Users\maraw\OneDrive\Documents\MATLAB\spm12\spm12

GT=niftiread('data\1\LabelsForTesting.nii');

class_GM_T1= double(niftiread('data\1\c1T1.nii'))/double(255);
class_WM_T1= double(niftiread('data\1\c2T1.nii'))/double(255);
class_CSF_T1= double(niftiread('data\1\c3T1.nii'))/double(255);
class_B_T1= double(niftiread('data\1\c4T1.nii'))/double(255);
class_ST_T1= double(niftiread('data\1\c5T1.nii'))/double(255);
class_BG_T1= double(niftiread('data\1\c6T1.nii'))/double(255);


[mx,pmx]=max(cat(4,class_CSF_T1,class_GM_T1,class_WM_T1,class_B_T1,class_ST_T1,class_BG_T1),[],4);

t1_CSF = (pmx==1);
t1_GM = (pmx==2);
t1_WM = (pmx==3);

% class_GM_Flair= double(niftiread('data\1\c1T2_FLAIR.nii'))/double(255);
% class_WM_Flair= double(niftiread('data\1\c2T2_FLAIR.nii'))/double(255);
% class_CSF_Flair= double(niftiread('data\1\c3T2_FLAIR.nii'))/double(255);

GT_CSF = (GT==1);
GT_GM = (GT==2);
GT_WM = (GT==3);

disp('GM T1:'); 
disp(dice(t1_GM,GT_GM));
disp('WM T1:');
disp(dice(t1_WM,GT_WM));
disp('CSF T1:');
disp(dice(t1_CSF,GT_CSF));


% disp('GM FLAIR:'); 
% disp(dice((class_GM_Flair>0.5),GT_GM));
% disp('WM FLAIR:'); 
% disp(dice((class_WM_Flair>0.5),GT_WM));
% disp('CSF FLAIR:'); 
% disp(dice((class_CSF_Flair>0.5),GT_CSF));