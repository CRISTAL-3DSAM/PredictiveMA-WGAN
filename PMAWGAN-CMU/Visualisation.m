function Visualisation(class,N_samples,Short_Term)
%addpath SRVFcodes/curveframework/

LABELS={'basketball','basketball_signal','directing_traffic','jumping','running','soccer','walking','washwindow','walking_extra'};
if ~any(strcmp(LABELS,class))
    fprintf('Wrong class name please use one of the correct class names\n\n');
    fprintf('''basketball'',''basketball_signal'',''directing_traffic'',''jumping'',''running'',''soccer'',''walking'',''washwindow'',''walking_extra''\n')
    return
end

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
Links_alt_17 ={[1,2],[1,5],[1,8],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[10,11],[9,12],[12,13],[13,14],[9,15],[15,16],[16,17]};
Links_alt = Links_alt_17;

%% parameters
test_file = ['Data_skeleton\Data_test_' suffix '.txt'];

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
selected = randperm(length(idx),N_samples);
selected_idx = idx(selected);
selected_lines = files(selected,:);

%% process samples
total_nb = 1;
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
    Condition = Data.curve_B(:,1:50);
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
    
    %plot figures
    figure('units','normalized','outerposition',[0 0 1 1])
    ax1 = nexttile;
    title('Ground Truth')
    set(gca,'XColor', 'none','YColor','none','ZColor','none')
    ax2 = nexttile;
    title('PredictiveMAWGAN')
    set(gca,'XColor', 'none','YColor','none','ZColor','none')
    for t=36:50
        p_mean = [path_mean '\mean.mat'];
        p_norm = [path_fronorm '\norm.mat'];
        p_center = [path_center '\center.mat'];
        U= Condition;
        x=U(1:L,t);
        y=U(L+1:L*2,t);
        z=U((L*2)+1:3*L,t);
        skel = denormalize([x,y,z],p_mean,p_norm,p_center);
        x = skel(:,1);
        y= skel(:,2);
        z= skel(:,3);
        cla(ax1);
        hold(ax1,'on');
        plot3(ax1,x,y,z, 'bo','MarkerSize',8,'MarkerFaceColor', 'b')
        for j = 1:size(Links_alt,2)
            xt=[x(Links_alt{j}(1)),x(Links_alt{j}(2))];
            yt=[y(Links_alt{j}(1)),y(Links_alt{j}(2))];
            zt=[z(Links_alt{j}(1)),z(Links_alt{j}(2))];
            p1=plot3(ax1,xt,yt,zt,'b','LineWidth',5);
            p1.Color(4)=0.5;
        end
        drawnow;
        hold(ax1,'off');
        axis(ax1,[-1000 1000 -1500 1500 -1300 1000]);
        view(ax1,-45,0)
        cla(ax2);
        hold (ax2,'on')
        plot3(ax2,x,y,z, 'bo','MarkerSize',8,'MarkerFaceColor', 'b')
        for j = 1:size(Links_alt,2)
            xt=[x(Links_alt{j}(1)),x(Links_alt{j}(2))];
            yt=[y(Links_alt{j}(1)),y(Links_alt{j}(2))];
            zt=[z(Links_alt{j}(1)),z(Links_alt{j}(2))];
            p2=plot3(ax2,xt,yt,zt,'b','LineWidth',5);
            p2.Color(4)=0.5;
        end
        drawnow;
        hold(ax2,'off');
        axis(ax2,[-1000 1000 -1500 1500 -1300 1000]);
        view(ax2,-45,0)
        F(total_nb) = getframe(gcf) ;
        total_nb = total_nb+1;
    end
    for t=2:fr
        p_mean = [path_mean '\mean.mat'];
        p_norm = [path_fronorm '\norm.mat'];
        p_center = [path_center '\center.mat'];
        U= Landmarks;
        x=U(1:L,t);
        y=U(L+1:L*2,t);
        z=U((L*2)+1:3*L,t);
        skel = denormalize([x,y,z],p_mean,p_norm,p_center);
        x = skel(:,1);
        y= skel(:,2);
        z= skel(:,3);
        cla(ax1);
        hold(ax1,'on');
        plot3(ax1,x,y,z, 'bo','MarkerSize',8,'MarkerFaceColor', 'b')
        for j = 1:size(Links_alt,2)
            xt=[x(Links_alt{j}(1)),x(Links_alt{j}(2))];
            yt=[y(Links_alt{j}(1)),y(Links_alt{j}(2))];
            zt=[z(Links_alt{j}(1)),z(Links_alt{j}(2))];
            p1=plot3(ax1,xt,yt,zt,'b','LineWidth',5);
            p1.Color(4)=0.5;
        end
        drawnow;
        hold(ax1,'off');
        axis(ax1,[-1000 1000 -1500 1500 -1300 1000]);
        view(ax1,-45,0)
        T= (reshape(curve(:,t),L,3));
        T = denormalize(T,p_mean,p_norm,p_center);
        cla(ax2);
        hold (ax2,'on')
        plot3(ax2,T(:,1),T(:,2),T(:,3),'ro','MarkerSize',8,'MarkerFaceColor', 'r');
        for k = 1:size(Links_alt,2)
            xt=[T(Links_alt{k}(1),1),T(Links_alt{k}(2),1)];
            yt=[T(Links_alt{k}(1),2),T(Links_alt{k}(2),2)];
            zt=[T(Links_alt{k}(1),3),T(Links_alt{k}(2),3)];
            p2=plot3(ax2,xt,yt,zt,'r','LineWidth',5);
            p2.Color(4)=0.5;
                
        end
        drawnow;
        hold(ax2,'off');
        axis(ax2,[-1000 1000 -1500 1500 -1300 1000]);
        view(ax2,-45,0)
        T=[];
        F(total_nb) = getframe(gcf) ;
        total_nb = total_nb+1;
    end
    close
end


%save video

writerObj = VideoWriter([class '_' suffix '_term.avi']);
writerObj.FrameRate = 15;
% set the seconds per image
% open the video writer
open(writerObj);
% write the frames to the video
for i=1:length(F)
    % convert the image to a frame
    frame = F(i) ;
    try
        writeVideo(writerObj, frame);
    catch e
        aaaa=1
    end
end
% close the writer object
close(writerObj);
number = num{1};
end

