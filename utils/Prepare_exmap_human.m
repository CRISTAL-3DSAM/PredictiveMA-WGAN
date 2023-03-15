function Prepare_exmap_human(Data_path,nb_fr,suffix)
% reformat SBU skeleton files to one file per frame, one point per line (x,y,z)
addpath utils\natsort
data_path=[Data_path '\expmap\expmap_sequences' suffix];
save_path=[Data_path '\expmap\expmap_Check' suffix];
fprintf('preparing expmaps \n')

N_joints= 33;

data_folder=dir(data_path);
data_folder=data_folder(3:end);
for i=1:length(data_folder)
    subj_folders=dir([data_path '\' data_folder(i).name]);
    subj_folders=subj_folders(3:end);
    mkdir([save_path '\' data_folder(i).name]);
    for j=1:length(subj_folders)
        if subj_folders(j).isdir
            mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name])
            subk_folders=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name]);
            subk_folders=subk_folders(3:end);
            for k=1:length(subk_folders)
                if subk_folders(k).isdir
                    expr_files=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name]);
                    expr_files=expr_files(3:end);
                    expr_files=natsort({expr_files.name});
                    N_frames=length(expr_files);
                    if N_frames~=nb_fr
                        continue
                    end
                    mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name])
                    % iterate over frames
                    x=zeros(N_frames,(N_joints)*3);
                    for w=1:N_frames
                        if w < 10
                            str = ['skeleton_pos_00' num2str(w) '.csv'];
                        elseif w<100
                            str = ['skeleton_pos_0' num2str(w) '.csv'];
                        else
                            str = ['skeleton_pos_' num2str(w) '.csv'];
                        end
                        y = csvread([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\' str]);
                        for f =1:N_joints
                            x(w,(f*3)-2:f*3)=[y(f,1),y(f,2),y(f,3)];
                        end
                    end
                    
                    if isempty(x)
                        continue
                    end

                  
                    curve_B = single(x);

                    
                    save([save_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\skeleton.mat'],'curve_B');
                end
            end
        end
    end
end
fprintf('expmap prepared \n\n')
end

