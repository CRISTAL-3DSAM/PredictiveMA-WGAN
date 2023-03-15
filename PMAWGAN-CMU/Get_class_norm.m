function [norm_mean,norm_std] = Get_class_norm(class,Short_Term)
%addpath SRVFcodes/curveframework/

if Short_Term
    fr = 11;
    suffix= 'short';
    prior=10;
    next=10;
else
    fr = 26;
    suffix= 'long';
    prior=25;
    next=25;
end

L=17;

%% parameters
test_file = ['Data_skeleton\Data_test_' suffix '.txt'];

N_samples=8; % 8 = number of sample in literrature
number_repeat = 100;


norm_pos_tot = zeros(number_repeat,fr);

%% randomly select samples

Test = importdata(test_file);
% keep only files from class
idx = [];
files = [];
for i=1:length(Test)
    line = Test{i};
    if contains(line,['/' class '/'])
        idx = [idx;i];
        files = [files;line];
    end
end
N_samples=min(N_samples,length(idx));
for repeat = 1 :number_repeat
    selected = randperm(length(idx),N_samples);
    selected_idx = idx(selected);
    selected_lines = files(selected,:);
    
    %% process samples
    norm_pos = zeros(N_samples,fr);
    
    for i=1:N_samples
        % load data
        SRVF_path = ['generated_samples_' suffix '\test_',num2str(selected_idx(i)), '.mat'];
        strstart = strfind(selected_lines(i,:),class);
        strend = strfind(selected_lines(i,:),'_SRVF');
        num = extractBetween(selected_lines(i,:),strstart(1)+length(class)+1,strend(1)-1);
        
        skeleton_path = ['Data_skeleton\Skeletons_for_visu_' suffix '\test\', class,...
            '\' num{1}, '\skeleton.mat'];
        norm_path = ['Data_skeleton\SRVF_Intensity_' suffix '\test\', class,...
            '\' num{1}, '_SRVF_intensity.mat'];
        
        path_mean = ['Data_skeleton\MEANS_' suffix '\test\', class,...
            '\' num{1}];
        path_fronorm = ['Data_skeleton\NORMS_' suffix '\test\', class,...
            '\' num{1}];
        path_center = ['Data_skeleton\CENTERS_' suffix '\test\', class,...
            '\' num{1}];
        
        
        Data= load(skeleton_path);
        SRVF = load(SRVF_path);
        Landmarks= Data.curve_B(:,50:end);
        intensity_data = load(norm_path);
        intensity = intensity_data.intensity;
        ff= Data.curve_B(:,50);
        q2 = SRVF.x_test;
        
        %get curve and sequence
        curve = q_to_curve(q2);
        
        curve = curve*intensity/(prior/next);
        for h=1:size(curve,2)
            curve(:,h)=curve(:,h)+squeeze(ff);
        end
        
        U= Landmarks;
        norm_cumul = zeros(fr,1);
        for t=1:fr
            
            p_mean = [path_mean '\mean.mat'];
            p_norm = [path_fronorm '\norm.mat'];
            p_center = [path_center '\center.mat'];
            
            x=U(1:L,t);
            y=U(L+1:L*2,t);
            z=U((L*2)+1:3*L,t);
            %denormalization
            skel = denormalize([x,y,z],p_mean,p_norm,p_center);
            x = skel(:,1);
            y= skel(:,2);
            z= skel(:,3);
            
            T= (reshape(curve(:,t),L,3));%remplacer par t pour vrai norme1
            T = denormalize(T,p_mean,p_norm,p_center);
            
            norm_joints = zeros(L,1);
            for joint=1:L
                GT = [x(joint),y(joint),z(joint)];
                norm_joints(joint) = norm(GT-T(joint,:)).^2;
            end
            
            
            frame_norm = sum(norm_joints);
            norm_cumul(t) =frame_norm;
            if t==1
                norm_pos(i,t)= frame_norm;
            else
                temp = sum(norm_cumul(2:t)); % first frame is the last frame of the prior sequence, we ignore it
                temp = temp./L;
                temp1 = temp./(t-1); %first frame not counted because norm = 0
                % norm on joint position Learning Dynamic Relationships for 3D
                % Human Motion Prediction version
                temp1 = sqrt(temp1);
                norm_pos(i,t)=temp1;
            end
            
            T=[];
            
        end
    end
    
    
    norm_pos_tot(repeat,:) = mean(norm_pos,1);
end

norm_mean = mean(norm_pos_tot,1);
norm_std = std(norm_pos_tot,1);
