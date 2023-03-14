function create_text_file_CMU(Data_path)

fprintf('creating text files for training and testing\n')
save_path=[Data_path '\3D\Data_train.txt'];
save_path_test = [Data_path '\3D\Data_test.txt'];


data_path=[Data_path '\3D\SRVF_next'];
data_path_partial = [Data_path '\3D\SRVF_prior'];
skeleton_path = [Data_path '\3D\Skeletons_Check'];

data_folder=dir(data_path);
data_folder=data_folder(3:end);
output = {};
output2={};
t=1;
u=1;

for i=1:length(data_folder)
    subj_folders=dir([data_path '\' data_folder(i).name]);
    subj_folders=subj_folders(3:end);
    for j=1:length(subj_folders)
        if subj_folders(j).isdir
            expr_files=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name]);
            expr_files=expr_files(3:end);
            expr_files=natsort({expr_files.name});
            N_frames=length(expr_files);
            for w=1:N_frames
                name=char(expr_files(w));
                A = load([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' name]);
                a_part = load([data_path_partial '\' data_folder(i).name '\' subj_folders(j).name '\' name]);
                skel = load([skeleton_path '\' data_folder(i).name '\' subj_folders(j).name '\' name(1:3) '\skeleton.mat']);
                if ~isnan(sum(sum(A.q2n))) && ~isnan(sum(sum(a_part.q2n))) && ~isnan(sum(sum(skel.curve_B)))
                    if  strcmp(data_folder(i).name,'test')
                        output2(u,1) = {['SRVF_prior/' data_folder(i).name '/' subj_folders(j).name   '/' name]};
                        output2(u,2) = {['SRVF_next/' data_folder(i).name '/' subj_folders(j).name  '/' name]};
                        output2(u,3) = {['First_Frames/' data_folder(i).name '/' subj_folders(j).name '/' name(1:3) '/skeleton_pos_050.csv']};
                        u=u+1;
                    else
                        output(t,1) = {['SRVF_prior/' data_folder(i).name '/' subj_folders(j).name  '/' name]};
                        output(t,2) = {['SRVF_next/' data_folder(i).name '/' subj_folders(j).name  '/' name]};
                        output(t,3) = {['First_Frames/' data_folder(i).name '/' subj_folders(j).name  '/' name(1:3) '/skeleton_pos_050.csv']};
                        output(t,4) = {['Skeletons_Rebuilt/' data_folder(i).name '/' subj_folders(j).name  '/' name(1:3) '/skeleton.mat']};
                        t=t+1;
                    end
                end
            end
        end
    end
end

writecell(output,save_path,'Delimiter',' ');
writecell(output2,save_path_test,'Delimiter',' ');
fprintf('Done \n\n')
end

