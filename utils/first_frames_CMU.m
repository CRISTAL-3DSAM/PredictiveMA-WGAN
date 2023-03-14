function first_frames_CMU(Data_path,Use_17_joints_CMU)
addpath utils\natsort
data_path= [Data_path '\3D\Skeletons_Check'];
save_path= [Data_path '\3D\First_Frames'];

frame_number = 50;

fprintf('copying first frames \n')
data_folder=dir(data_path);
data_folder=data_folder(3:end);
for i=1:length(data_folder)
    mkdir([save_path '\' data_folder(i).name]);
    subj_folders=dir([data_path '\' data_folder(i).name]);
    subj_folders=subj_folders(3:end);
    for j=1:length(subj_folders)
        mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name])
        subk_folders=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name]);
        subk_folders=subk_folders(3:end);
        for k=1:length(subk_folders)
            mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name])
            A = load([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\skeleton.mat']);
            A = A.curve_B;
            if Use_17_joints_CMU
                ff = [A(1:17,frame_number),A(18:34,frame_number),A(35:51,frame_number)];
            else
                ff = [A(1:25,frame_number),A(26:50,frame_number),A(51:75,frame_number)];
            end
            writematrix(ff,[save_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\skeleton_pos_050.csv']);
        end
    end
end
fprintf('done \n')
end

