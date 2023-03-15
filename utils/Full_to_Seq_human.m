function  Full_to_Seq_human(Data_path,nb_target_frame,downsampling,jump,suffix)
addpath utils\natsort
% turn full frames sequence into sequence of nb_target_frame
%downsampling = perform downsampling by 2 default = true for human3.6M
%jump = difference of frame between the start of a sequence and the next if
%jump => nb_target_frame then all sequences will contains unique frames

data_path = [Data_path '\expmap\expmap_full']; % csv file folder
save_path = [Data_path '\expmap\expmap_sequences' suffix]; % save folder

joint_ignore = [];

fprintf('creating sequences \n')
data_folder=dir(data_path);
data_folder=data_folder(3:end);
for i=1:length(data_folder)
    subj_folders=dir([data_path '\' data_folder(i).name]);
    subj_folders=subj_folders(3:end);
    mkdir([save_path '\' data_folder(i).name]);
    for j=1:length(subj_folders)
        subk_folders=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name]);
        subk_folders=subk_folders(3:end);
        mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name]);
        nb_seq = 1;
        for k=1:length(subk_folders)
            expr_files=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name]);
            expr_files=expr_files(3:end);
            expr_files=natsort({expr_files.name});
            nb_frames=length(expr_files);
            check = ones(nb_frames,1);
            max_seq = fix(nb_frames/nb_target_frame)*fix((nb_target_frame/jump));
            % remove one frame out of two
            if downsampling
                max_seq = fix(fix(nb_frames/2)/nb_target_frame)*fix((nb_target_frame/jump));
                for z = 1:nb_frames
                    if ~rem(z,2)
                        check(z)=0;
                    end
                end
            end
            nb_fram_in_seq=0;
            nb=0;
			nb_2 = 0;
            num_seq_done = 0;
            mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name '\' sprintf('%03d',nb_seq)]);
            while nb<nb_frames && num_seq_done<max_seq
                nb=nb+1;
                if check(nb)
                   nb_fram_in_seq = nb_fram_in_seq+1;
                   data = [data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\' expr_files{nb}];
                   save = [save_path '\' data_folder(i).name '\' subj_folders(j).name '\' sprintf('%03d',nb_seq) '\skeleton_pos_' sprintf('%03d',nb_fram_in_seq) '.csv' ];
                   pts=csvread(data);
                   pts(joint_ignore,:) = [];
                   ptsc = pts;
                   %save data
                   
                   ptsc=single(ptsc);
                   
                   writematrix(ptsc,save);
                   %change floder when max_size reached
                   if nb_fram_in_seq ==nb_target_frame
                       nb_seq = nb_seq+1;
                       num_seq_done = num_seq_done+1;
                       nb_fram_in_seq =0;
					   nb_2 = nb_2+jump;
                       nb= nb_2-1;
                       if num_seq_done~=max_seq
                           mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name '\' sprintf('%03d',nb_seq)]);
                       end
                   end
                end
            end
        end
    end
end
fprintf('sequences created \n\n')
end

