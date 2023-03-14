function SRVF_mean_CMU(Data_path,nb_frames_predict,joints_17)
% compute the mean SRVF used for the tangent space
addpath utils\SRVFcodes\curveframework/
addpath utils\natsort

if joints_17
    L=17;
else
    L=25;
end
len=nb_frames_predict+1;%32;  %% target video lenght
start = 49; %start frame-1
q_samples_action={};
x_samples_action={};

fprintf('processing mean SRVF \n');

%%%%%% SRVF %%%%%%%
data_path= [Data_path '\3D\Skeletons_reduced'];
save_path=[Data_path '\3D'];
data_folder=dir(data_path);
data_folder=data_folder(3:end);
length(data_folder)
for i=1:length(data_folder)
    subj_folders=dir([data_path '\' data_folder(i).name]);
    subj_folders=subj_folders(3:end);
    for j=1:length(subj_folders)
        if subj_folders(j).isdir
            subk_folders=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name]);
            subk_folders=subk_folders(3:end);
            for k=1:length(subk_folders)
                if subk_folders(k).isdir
                    expr_files=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name ]);
                    expr_files=expr_files(3:end);
                    expr_files=natsort({expr_files.name});
                    % iterate over frames
                    Landmarks=zeros(len, 3*L);
                    for w=1:len
                        name=char(expr_files(w+start));
                        pts=csvread([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name  '\' name(1:end-4) '.csv']);
                        pts=[pts(:,1);pts(:,2);pts(:,3)];
                        Landmarks(w,:)= reshape(pts, 1, 3*L);
                    end
                    Landmarks=Landmarks'; 
                    
                    
                    X2 = Landmarks;
                    x_samples_action{end+1}=X2;
                    q_samples_action{end+1}=curve_to_q(X2);
                    clear Landmarks;  clear pts;  clear X2; clear X2_2
                end
                
            end
        end
    end
end
q_mean_action=Karcher_Mean( q_samples_action, x_samples_action, 100,0.9);
q_mean = q_mean_action;
save([save_path '\'  'q_mean_data.mat'], 'q_mean');

fprintf('Done \n');
end

