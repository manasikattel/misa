biasreg = [0.001, 0.01, 0.1, 1, 10];
biasfwhm = [40,60,80,100,120, 140];
cleanup = [0,1,2];
mrf = [0,1];
tissue1_ngauss = [2];
tissue2_ngauss = [2];
tissue3_ngauss = [2];
tissue4_ngauss = [3];
tissue5_ngauss = [4];
tissue6_ngauss = [2];

n    = length(biasreg) * length(biasfwhm) * length(cleanup) * length(mrf);
% H    = waitbar(0, 'Please wait...');
i=1;

nrun = 1;
jobfile = {'job2.m'};

for br=1:length(biasreg)
    for bf=1:length(biasfwhm)
        for cu=1:length(cleanup)
            for mr=1:length(mrf)
                    
                    filename = strcat("biasreg_",string(biasreg(br)), "_fwhm_",string(biasfwhm(bf)), "_cu_",string(cleanup(cu)), "_mrf_",string(mrf(mr)),".csv");

                    str_1=strcat("sed -i '' 's/biasreg(br)/",string(biasreg(br)),"/g' job.m");
                    str_2=strcat("sed -i '' 's/biasfwhm(bf)/",string(biasfwhm(bf)),"/g' job.m");
                    str_3=strcat("sed -i '' 's/cleanup(cu)/",string(cleanup(cu)),"/g' job.m");
                    str_4=strcat("sed -i '' 's/mrf(mr)/",string(mrf(mr)),"/g' job.m");

                    system(str_1);
                    system(str_2);
                    system(str_3);
                    system(str_4);


                    jobs = repmat(jobfile, 1, nrun);
                    inputs = cell(0, nrun);
                    for crun = 1:nrun
                    end
                    spm('defaults', 'FMRI');
                    spm_jobman('run', jobs, inputs{:});
                    
%                     dice_calc(filename);
%                                             waitbar(i/n, H, sprintf('%d of %d', i, n));
%                     i = i+1;

                                   
            end
        end
    end
end