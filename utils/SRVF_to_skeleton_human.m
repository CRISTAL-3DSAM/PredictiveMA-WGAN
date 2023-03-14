function SRVF_to_skeleton_human(Data_path)
%reconstruct skeletons from SRVF
addpath utils
data_path= [Data_path '\3D\SRVF_next']; %SRVF path
skeleton_path = [Data_path '\3D\Skeletons_Check'];%skeleton_path .mat format with curve_B variable name
save_path=[Data_path '\3D\Skeletons_Rebuilt'];

frame =50; % frame de départ
L=17; % nombre de joint

fprintf('reconstructing skeletons from SRVF \n')

data_folder=dir(data_path);
data_folder=data_folder(3:end);
for i=1:length(data_folder)
    subj_folders=dir([data_path '\' data_folder(i).name]);
    subj_folders=subj_folders(3:end);
    mkdir([save_path '\' data_folder(i).name]);
    for j=1:length(subj_folders)
        if subj_folders(j).isdir
            mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name]);
            expr_files=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name]);
            expr_files=expr_files(3:end);
            expr_files=natsort({expr_files.name});
            if isempty(expr_files)
                continue
            end
            N_files=length(expr_files);
            for w=1:N_files
                name=char(expr_files(w));
                nameidx = findstr(name,'_');
                names = extractBefore(name,nameidx);
                SRVF = load([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' name]);
                skeleton = load([skeleton_path '\' data_folder(i).name '\' subj_folders(j).name '\' names '\skeleton.mat']);
                mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name '\' names])
                SRVF = SRVF.q2n;
                skeleton = skeleton.curve_B;
                skeleton = skeleton(:,frame);
                curve_B = q_to_curve(SRVF);
                for h=1:size(curve_B,2)
                    curve_B(:,h)=curve_B(:,h)+squeeze(skeleton);
                end
                save([save_path '\' data_folder(i).name '\' subj_folders(j).name '\' names '\skeleton.mat'],'curve_B');
            end
            
        end
    end
end
fprintf('Done \n\n')
end

