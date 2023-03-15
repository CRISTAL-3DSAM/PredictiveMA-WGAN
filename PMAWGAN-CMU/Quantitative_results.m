function results = Quantitative_results(Short_Term)

LABELS={'basketball','basketball_signal','directing_traffic','jumping','running','soccer','walking','washwindow','walking_extra'};

if Short_Term
    nb_frames = 11;
else
    nb_frames = 26;
end

MEAN =zeros(9,nb_frames);
STD =zeros(9,nb_frames);
for i = 1:size(LABELS,2)
    perc= int16(round((i/size(LABELS,2))*100,0));  
    [means,stds] = Get_class_norm(LABELS{i},Short_Term);
    MEAN(i,:)=means;
    STD(i,:)=stds;
    fprintf('%d%% Done\n',perc);
end

if nb_frames == 26 
    col = 5;
    MEAN_tab= round([MEAN(:,3),MEAN(:,5),MEAN(:,9),MEAN(:,11),MEAN(:,26)],1);
    STD_tab = round([STD(:,3),STD(:,5),STD(:,9),STD(:,11),STD(:,26)],1);
elseif nb_frames == 11
    col =4;
    MEAN_tab= round([MEAN(:,3),MEAN(:,5),MEAN(:,9),MEAN(:,11)],1);
    STD_tab = round([STD(:,3),STD(:,5),STD(:,9),STD(:,11)],1);
end
Average= mean(MEAN_tab(1:8,:),1);
results=strings(size(LABELS,2)+2,col+1);
for i = 1:size(LABELS,2)
    for j =1:col
        results(i+1,j+1) = string([char(string(MEAN_tab(i,j))) ' ± ' char(string(STD_tab(i,j)))]);     
    end
end
if Short_Term
    results(1,2:end)=["80","160","320","400"];
else
    results(1,2:end)=["80","160","320","400","1000"];
end
results(:,1)=["milliseconds";"basketball";"basketball_signal";"directing_traffic";"jumping";"running";"soccer";"walking";"washwindow";"walking_extra";"Average (no walking_extra)"];
results(end,2:end) = string(round(Average,1));
end
