function  Create_SRVF_human(Data_path,nb_frames,nb_frames_prior,nb_frames_predict)
%create SRVF from skeletons
addpath utils\SRVFcodes/curveframework/
addpath utils\natsort

fprintf('building SRVFs \n')
L=17; %number of joint
l=nb_frames; % number of SRVF frames
start = 0; % start frame number 
%% here set the path to save SRVF
data_path=[Data_path '\3D\Skeletons_reduced'];
path_SRVF_50 = [Data_path, '\3D\SRVF_prior'];
path_SRVF_10 = [Data_path, '\3D\SRVF_next'];
path_intensity_50 = [Data_path '\3D\SRVF_intensity_prior'];
path_intensity_10 = [Data_path '\3D\SRVF_intensity_next'];


%% path to the skeleton data
 

data_folder=dir(data_path);
data_folder=data_folder(3:end);
for i=1:length(data_folder)
    subj_folders=dir([data_path '\' data_folder(i).name]);
    subj_folders=subj_folders(3:end); 
    mkdir([path_SRVF_50 '\' data_folder(i).name]);mkdir([path_SRVF_10 '\' data_folder(i).name]);
    mkdir([path_intensity_50 '\' data_folder(i).name]);mkdir([path_intensity_10 '\' data_folder(i).name]);
    
    for j=1:length(subj_folders)
        if subj_folders(j).isdir           
            subk_folders=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name]);
            subk_folders=subk_folders(3:end);
            
            mkdir([path_SRVF_50 '\' data_folder(i).name '\' subj_folders(j).name]);mkdir([path_SRVF_10 '\' data_folder(i).name '\' subj_folders(j).name]);
            mkdir([path_intensity_50 '\' data_folder(i).name '\' subj_folders(j).name]);mkdir([path_intensity_10 '\' data_folder(i).name '\' subj_folders(j).name]);
            
            for k=1:length(subk_folders)
                if subk_folders(k).isdir
                    expr_files=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name]);
                    expr_files=expr_files(3:end);
                    expr_files=natsort({expr_files.name});                
                    if isempty(expr_files)
                        continue
                    end
                    % iterate over frames
                    Landmarks=zeros(l, 3*L);
                    for w=1:l
                        % start after base frame
                        name=char(expr_files(w+start));
                        pts=csvread([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '\' name(1:end-4) '.csv']);
                        pts=[pts(:,1);pts(:,2);pts(:,3)];
                        Landmarks(w,:)= reshape(pts, 1, 3*L);
                    end

                    
                    Landmarks=Landmarks';
                   
                    L1 = Landmarks(:,(50-nb_frames_prior+1):50);
                    L2 = Landmarks(:,50:50+nb_frames_predict);
                    X2_1 = L1;
                    
                    [q2_1,intensity_1]=curve_to_q(X2_1);
                    
                    
                    X2_2 = L2;
                    [q2_2,intensity_2]=curve_to_q(X2_2);

                    q2n=q2_1;
                    intensity = intensity_1;
                    save([path_SRVF_50 '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '_SRVF.mat'],'q2n');
                    save([path_intensity_50 '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '_SRVF_intensity.mat'],'intensity');
                    q2n=q2_2;
                    intensity = intensity_2;
                    save([path_SRVF_10 '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '_SRVF.mat'],'q2n');
                    save([path_intensity_10 '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '_SRVF_intensity.mat'],'intensity');
                  
                   
                    %%%% save global SRVF
                    %save([path_intensity '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '_SRVF_intensity.mat'],'intensity');
                    %save([path_SRVF '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '_SRVF.mat'],'q2n');
                    %save([path_X2 '\' data_folder(i).name '\' subj_folders(j).name '\' subk_folders(k).name '_X2.mat'],'X2n');
                    clear Landmarks; clear q_sample; clear pts; clear q_mean; clear q2n; clear gamI; clear gam; clear X2, clear q2;
                end
            end
        end
        
    end
    
end
fprintf('SRVFs built \n\n')
end

