clear
addpath utils

DB_choice = 'CMU_long'; %choose between 'CMU_short', 'CMU_long', 'HM36_short', 'HM36_long'

%%PARAMETERS
if DB_choice=='CMU_short'
    Database = 'CMU'; % Database used 'HUMAN' or 'CMU'
    DATA_PATH = 'CMU_MoCaP'; % path to the database
    SAVE_PATH = 'CMU_short'; %path to save the processed database
    sequence_length = 60; %length of each motion sequences
    prior_length = 10; % length of the prior sequence <=50
    predict_length = 10; % length of the prdicted sequence sequence_length-predict_length <=50
    frames_jump = 15; % frame difference between the start frames of 2 sequences
    downsampling =true; % perform downsampling by 2. 50fps->25fps motions
    Use_17_joints_CMU=true; % use 17 joints representation default is 25 joints
    suffix='_short';
elseif DB_choice=='CMU_long'
    Database = 'CMU'; % Database used 'HUMAN' or 'CMU'
    DATA_PATH = 'CMU_MoCaP'; % path to the database
    SAVE_PATH = 'CMU_long'; %path to save the processed database
    sequence_length = 75; %length of each motion sequences
    prior_length = 25; % length of the prior sequence <=50
    predict_length = 25; % length of the prdicted sequence sequence_length-predict_length <=50
    frames_jump = 15; % frame difference between the start frames of 2 sequences
    downsampling =true; % perform downsampling by 2. 50fps->25fps motions
    Use_17_joints_CMU=true; % use 17 joints representation default is 25 joints
    suffix='_long';
elseif DB_choice=='HM36_short'
    Database = 'HUMAN'; % Database used 'HUMAN' or 'CMU'
    DATA_PATH = 'Human3.6M'; % path to the database
    SAVE_PATH = 'HM36_short'; %path to save the processed database
    sequence_length = 60; %length of each motion sequences
    prior_length = 10; % length of the prior sequence <=50
    predict_length = 10; % length of the prdicted sequence sequence_length-predict_length <=50
    frames_jump = 60; % frame difference between the start frames of 2 sequences
    downsampling =true; % perform downsampling by 2. 50fps->25fps motions
    Use_17_joints_CMU=true; % use 17 joints representation default is 25 joints
    suffix='_short';
elseif DB_choice=='HM36_long'
    Database = 'HUMAN'; % Database used 'HUMAN' or 'CMU'
    DATA_PATH = 'Human3.6M'; % path to the database
    SAVE_PATH = 'HM36_long'; %path to save the processed database
    sequence_length = 75; %length of each motion sequences
    prior_length = 25; % length of the prior sequence <=50
    predict_length = 25; % length of the prdicted sequence sequence_length-predict_length <=50
    frames_jump = 75; % frame difference between the start frames of 2 sequences
    downsampling =true; % perform downsampling by 2. 50fps->25fps motions
    Use_17_joints_CMU=true; % use 17 joints representation default is 25 joints
    suffix='_long';
else
    print('not an existing setup')
end

%% CODE
if strcmp(Database,'HUMAN')
    mkdir(SAVE_PATH);
    Process_dataset_human(DATA_PATH,SAVE_PATH);
    Full_to_Seq_human(SAVE_PATH,sequence_length,downsampling,frames_jump,suffix);
    Prepare_exmap_human(SAVE_PATH,sequence_length,suffix);
    expmap_to_3D_human(SAVE_PATH,sequence_length,suffix);
    normalize_human(SAVE_PATH,suffix);
    Prepare_Skeletons_human(SAVE_PATH,suffix);
    Create_SRVF_human(SAVE_PATH,sequence_length,prior_length,predict_length,suffix);
    SRVF_mean_human(SAVE_PATH,predict_length,suffix);
    SRVF_to_skeleton_human(SAVE_PATH,suffix);
    first_frames_human(SAVE_PATH,suffix);
    create_text_file_human(SAVE_PATH,suffix);
    movefile([SAVE_PATH '/3D/Skeletons_Check' suffix], [SAVE_PATH '/3D/Skeletons_for_visu' suffix])
    fprintf('dataset succesfully created\n\n');
elseif strcmp(Database,'CMU')
    Process_dataset_cmu(DATA_PATH,SAVE_PATH);
    Full_to_Seq_cmu(SAVE_PATH,sequence_length,downsampling,frames_jump,suffix);
    Prepare_exmap_cmu(SAVE_PATH,sequence_length,suffix);
    expmap_to_3D_cmu(SAVE_PATH,sequence_length,suffix);
    normalize_CMU(SAVE_PATH,Use_17_joints_CMU,suffix);
    Prepare_Skeletons_CMU(SAVE_PATH,Use_17_joints_CMU,suffix);
    Create_SRVF_CMU(SAVE_PATH,sequence_length,prior_length,predict_length,Use_17_joints_CMU,suffix);
    SRVF_mean_CMU(SAVE_PATH,predict_length,Use_17_joints_CMU,suffix)
    SRVF_to_skeleton_CMU(SAVE_PATH,Use_17_joints_CMU,suffix);
    first_frames_CMU(SAVE_PATH,Use_17_joints_CMU,suffix);
    create_text_file_CMU(SAVE_PATH,suffix);
    movefile([SAVE_PATH '/3D/Skeletons_Check' suffix], [SAVE_PATH '/3D/Skeletons_for_visu' suffix])
    fprintf('dataset succesfully created\n\n');
else
    fprintf('Wrong dataset only HUMAN and CMU allowed \n\n');
end