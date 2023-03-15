function normalize_human(Data_path, suffix)
addpath utils\natsort
% turn full frames sequence into 60 or 90 frames sequences

data_path = [Data_path '\3D\Skeletons' suffix]; % csv file folder
save_path = [Data_path '\3D\Skeletons_reduced' suffix]; % save folder
mean_path = [Data_path '\3D\MEANS' suffix];
norm_path = [Data_path '\3D\NORMS' suffix];
center_path = [Data_path '\3D\CENTERS' suffix];

joint_ignore =  [5,6,10,11,12,17,21,22,23,24,25,29,30,31,32]; %17 joints remaining

fprintf('normalisation in progress\n')
data_folder=dir(data_path);
data_folder=data_folder(3:end);
for i=1:length(data_folder)
    subj_folders=dir([data_path '\' data_folder(i).name]);
    subj_folders=subj_folders(3:end);
    mkdir([save_path '\' data_folder(i).name]);
    mkdir([mean_path '\' data_folder(i).name]);
    mkdir([norm_path '\' data_folder(i).name]);
    mkdir([center_path '\' data_folder(i).name]);
    for j=1:length(subj_folders)
        subk_folders=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name]);
        subk_folders=subk_folders(3:end);
        mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name]);
        mkdir([mean_path '\' data_folder(i).name '\' subj_folders(j).name]);
        mkdir([norm_path '\' data_folder(i).name '\' subj_folders(j).name]);
        mkdir([center_path '\' data_folder(i).name '\' subj_folders(j).name]);
        nb_seq = 1;
        for k=1:length(subk_folders)
            expr_files=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name]);
            expr_files=expr_files(3:end);
            expr_files=natsort({expr_files.name});
            mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name]);
            mkdir([mean_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name]);
            mkdir([norm_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name]);
            mkdir([center_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name]);
            for w=1:length(expr_files)
                
                   name=char(expr_files(w));
                   data = [data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\' name];
                   Save = [save_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\' name];
                   Mean = [mean_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\mean.mat' ];
                   Norm = [norm_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\norm.mat' ];
                   Center = [center_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\center.mat' ];

                   pts=csvread(data);
                   pts(joint_ignore,:) = [];
                       if w ==1
                           m1 = mean(pts(:,1));
                           m2 = mean(pts(:,2));
                           m3 = mean(pts(:,3));
                           pts1 = pts - [m1,m2,m3];
                           normFro = norm( pts1 ,'fro');
                           pts1 = pts1 / normFro;
                           xc = pts1(12,1);
                           yc = pts1(12,2);
                           zc = pts1(12,3);
                           x = pts1(:,1) - xc;
                           y = pts1(:,2) - yc;
                           z = pts1(:,3) - zc;
                           ptsc = [x y z];
                           centering = [xc yc zc];
                           norms = normFro;
                           means = [m1, m2, m3];
                           save(Mean,'means');
                           save(Norm,'norms');
                           save(Center,'centering');
                       else
                           pts1 = pts - [m1,m2,m3];
                           pts1 = pts1 / normFro;
                           x = pts1(:,1) - xc;
                           y = pts1(:,2) - yc;
                           z = pts1(:,3) - zc;
                           ptsc = [x y z];
                       end
                   %save data
                   writematrix(ptsc,Save);
            end
        end
    end
end
fprintf('normalisation done \n')
end

