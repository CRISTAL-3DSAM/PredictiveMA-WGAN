function Process_dataset_human(data_path,Save_path)

% transform the database to create single frame files

addpath utils\natsort
save_path = [Save_path '\expmap\expmap_full'];

fprintf('processing original database \n')
data_folder=dir(data_path);
data_folder=data_folder(3:end);
for i=1:length(data_folder)
    subj_folders=dir([data_path '\' data_folder(i).name]);
    subj_folders=subj_folders(3:end);
    mkdir([save_path '\' data_folder(i).name]);
    for j=1:length(subj_folders)
        if subj_folders(j).isdir  
            mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name])
            expr_files=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name]);
            expr_files=expr_files(3:end);
            expr_files=natsort({expr_files.name});
            num_seq=length(expr_files);
            % iterate over sequence
            for w=1:num_seq
                num_fold = sprintf('%02d',w);
                mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name '\' num_fold]);
                name=char(expr_files(w));
                CFD = importdata([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' name]);
                len = size(CFD,1);
                for k =1:len
                    X = zeros(33,1);
                    Y = zeros(33,1);
                    Z = zeros(33,1);
                    L = CFD(k,:);
                    for q =1:32
                        X(q)= L((q*3)-2);
                        Y(q)= L((q*3)-1);
                        Z(q)= L((q*3));
                    end
                    
                    pts=[X,Y,Z];
                    writematrix(pts,[save_path '\' data_folder(i).name '\' subj_folders(j).name '\' num_fold '\skeleton_pos_' num2str(k,'%04d') '.csv']);
                end
            end
        end
    end
end
fprintf('original database processed \n\n')
end

