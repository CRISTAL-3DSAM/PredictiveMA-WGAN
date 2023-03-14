clear
addpath utils

%%PARAMETERS
Database = 'CMU'; % Database used 'HUMAN' or 'CMU'
DATA_PATH = 'CMU_MoCaP'; % path to the database
SAVE_PATH = 'CMU_17_15_60_frames_test'; %path to save the processed database
sequence_length = 60; %length of each motion sequences
prior_length = 10; % length of the prior sequence <=50
predict_length = 10; % length of the prdicted sequence sequence_length-predict_length <=50
frames_jump = 15; % frame difference between the start frames of 2 sequences
downsampling =true; % perform downsampling by 2. 50fps->25fps motions
Use_17_joints_CMU=true; % use 17 joints representation default is 25 joints

%% CODE
if strcmp(Database,'HUMAN')
    mkdir(SAVE_PATH);
    Process_dataset_human(DATA_PATH,SAVE_PATH);
    Full_to_Seq_human(SAVE_PATH,sequence_length,downsampling,frames_jump);
    Prepare_exmap_human(SAVE_PATH,sequence_length);
    expmap_to_3D_human(SAVE_PATH,sequence_length);
    normalize_human(SAVE_PATH);
    Prepare_Skeletons_human(SAVE_PATH);
    Create_SRVF_human(SAVE_PATH,sequence_length,prior_length,predict_length);
    SRVF_mean_human(SAVE_PATH,predict_length);
    SRVF_to_skeleton_human(SAVE_PATH);
    first_frames_human(SAVE_PATH);
    create_text_file_human(SAVE_PATH);
    fprintf('dataset succesfully created\n\n');
elseif strcmp(Database,'CMU')
    Process_dataset_cmu(DATA_PATH,SAVE_PATH);
    Full_to_Seq_cmu(SAVE_PATH,sequence_length,downsampling,frames_jump);
    Prepare_exmap_cmu(SAVE_PATH,sequence_length);
    expmap_to_3D_cmu(SAVE_PATH,sequence_length);
    normalize_CMU(SAVE_PATH,Use_17_joints_CMU);
    Prepare_Skeletons_CMU(SAVE_PATH,Use_17_joints_CMU);
    Create_SRVF_CMU(SAVE_PATH,sequence_length,prior_length,predict_length,Use_17_joints_CMU);
    SRVF_mean_CMU(SAVE_PATH,predict_length,Use_17_joints_CMU)
    SRVF_to_skeleton_CMU(SAVE_PATH,Use_17_joints_CMU);
    first_frames_CMU(SAVE_PATH);
    create_text_file_CMU(SAVE_PATH);
    fprintf('dataset succesfully created\n\n');
else
    fprintf('Wrong dataset only HUMAN and CMU allowed \n\n');
end