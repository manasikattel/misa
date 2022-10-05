matlabbatch{1}.spm.spatial.preproc.channel(1).vols = {
                                                      '/Users/manasikattel/misa/Lab1/data/1/T1.nii,1'
                                                      '/Users/manasikattel/misa/Lab1/data/2/T1.nii,1'
                                                      '/Users/manasikattel/misa/Lab1/data/3/T1.nii,1'
                                                      '/Users/manasikattel/misa/Lab1/data/4/T1.nii,1'
                                                      '/Users/manasikattel/misa/Lab1/data/5/T1.nii,1'
                                                      };
matlabbatch{1}.spm.spatial.preproc.channel(1).biasreg = biasreg(br);
matlabbatch{1}.spm.spatial.preproc.channel(1).biasfwhm = biasfwhm(bf);
matlabbatch{1}.spm.spatial.preproc.channel(1).write = [0 1];


matlabbatch{1}.spm.spatial.preproc.channel(2).vols = {
                                                      '/Users/manasikattel/misa/Lab1/data/1/T2_FLAIR.nii,1'
                                                      '/Users/manasikattel/misa/Lab1/data/2/T2_FLAIR.nii,1'
                                                      '/Users/manasikattel/misa/Lab1/data/3/T2_FLAIR.nii,1'
                                                      '/Users/manasikattel/misa/Lab1/data/4/T2_FLAIR.nii,1'
                                                      '/Users/manasikattel/misa/Lab1/data/5/T2_FLAIR.nii,1'
                                                      };
matlabbatch{1}.spm.spatial.preproc.channel(2).biasreg = biasreg(br);
matlabbatch{1}.spm.spatial.preproc.channel(2).biasfwhm = biasfwhm(bf);
matlabbatch{1}.spm.spatial.preproc.channel(2).write = [0 1];



matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {'/Users/manasikattel/spm12/tpm/TPM.nii,1'};
matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {'/Users/manasikattel/spm12/tpm/TPM.nii,2'};
matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {'/Users/manasikattel/spm12/tpm/TPM.nii,3'};
matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {'/Users/manasikattel/spm12/tpm/TPM.nii,4'};
matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {'/Users/manasikattel/spm12/tpm/TPM.nii,5'};
matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {'/Users/manasikattel/spm12/tpm/TPM.nii,6'};
matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.warp.mrf = mrf(mr);
matlabbatch{1}.spm.spatial.preproc.warp.cleanup = cleanup(cu);
matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];
matlabbatch{1}.spm.spatial.preproc.warp.vox = NaN;
matlabbatch{1}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                          NaN NaN NaN];
