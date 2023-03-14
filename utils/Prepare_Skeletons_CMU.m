function Prepare_Skeletons_CMU(Data_path,joints_17)
% reformat SBU skeleton files to one file per frame, one point per line (x,y,z)

addpath utils\natsort
data_path= [Data_path '\3D\Skeletons_reduced'];
save_path= [Data_path '\3D\Skeletons_Check'];

fprintf('preparing skeletons\n');

if joints_17
    N_joints= 17;
else 
    N_joints= 25;
end

data_folder=dir(data_path);
data_folder=data_folder(3:end);
for i=1:length(data_folder)
    subj_folders=dir([data_path '\' data_folder(i).name]);
    subj_folders=subj_folders(3:end);
    mkdir([save_path '\' data_folder(i).name]);
    fprintf("%d\n",i);
    for j=1:length(subj_folders)
        if subj_folders(j).isdir
            mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name])
            subk_folders=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name]);
            subk_folders=subk_folders(3:end);
            for k=1:length(subk_folders)
                if subk_folders(k).isdir
                    mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name])
                    expr_files=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name]);
                    expr_files=expr_files(3:end);
                    expr_files=natsort({expr_files.name});
                    N_frames=length(expr_files);
                    % iterate over frames
                    x=zeros(N_joints, 3,N_frames);
                    for w=1:N_frames
                        if w < 10
                            str = ['skeleton_pos_00' num2str(w) '.csv'];
                        elseif w<100
                            str = ['skeleton_pos_0' num2str(w) '.csv'];
                        else
                            str = ['skeleton_pos_' num2str(w) '.csv'];
                        end
                        x(:,:,w) = csvread([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\' str]);
                    end
                    
                    if isempty(x)
                        continue
                    end
                    
                    X = [x(:,1,1);x(:,2,1);x(:,3,1)];

                    for w=2:N_frames
                        t = [x(:,1,w);x(:,2,w);x(:,3,w)];
                        X = [X,t];
                    end
                   
                    curve_B = X; % pour compat avec python

                    
                    save([save_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\skeleton.mat'],'curve_B');
                end
            end
        end
    end
end
fprintf('Done\n\n');
end

