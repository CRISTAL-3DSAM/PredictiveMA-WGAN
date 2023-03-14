function expmap_to_3D_cmu(Data_path,fr)
%transform expmap to 3D coordinates
addpath utils\natsort
data_path= [Data_path '\expmap\expmap_Check']; %SRVF path
save_path=[Data_path '\3D\Skeletons'];

mkdir([Data_path '\3D']);

fprintf('transform expmap to 3D coordinates \n')
parent = [0, 1, 2, 3, 4, 5, 6,1, 8, 9,10, 11,12, 1, 14,15,16,17,18,19, 16,21,22,23,24,25,26,24,28,16,30,31,32,33,34,35,33,37];


rotInd = [6,5,4;9,8,7;12,11,10;15,14,13;18,17,16;21,20,19;-1,-1,-1;24,23,22;27,26,25;30,29,28;33,32,31;36,35,34;-1,-1,-1;39,38,37;42,41,40;45,44,43;48,47,46;51,50,49;54,53,52;-1,-1,-1;57,56,55;60,59,58;63,62,61;66,65,64;69,68,67;72,71,70;-1,-1,-1;75,74,73;-1,-1,-1;78,77,76;81,80,79;84,83,82;87,86,85;90,89,88;93,92,91;-1,-1,-1;96,95,94;-1,-1,-1];

posInd=[];
for ii =1:38
  if ii==1
      posInd=[posInd;1,2,3];
  else
      posInd=[posInd;-1,-1,-1];
  end
end

offset = 70*[0,0	,0	,0,	0,	0,	1.65674000000000,	-1.80282000000000,	0.624770000000000,	2.59720000000000,	-7.13576000000000,	0,	2.49236000000000,	-6.84770000000000,	0,	0.197040000000000,	-0.541360000000000,	2.14581000000000,	0,	0,	1.11249000000000,	0,	0,	0,	-1.61070000000000,	-1.80282000000000,	0.624760000000000,	-2.59502000000000,	-7.12977000000000,	0,	-2.46780000000000,	-6.78024000000000,	0,	-0.230240000000000,	-0.632580000000000,	2.13368000000000,	0,	0,	1.11569000000000,	0,	0,	0,	0.0196100000000000,	2.05450000000000,	-0.141120000000000,	0.0102100000000000,	2.06436000000000,	-0.0592100000000000,	0,	0,0,	0.00713000000000000,	1.56711000000000,	0.149680000000000,	0.0342900000000000,	1.56041000000000,	-0.100060000000000,	0.0130500000000000,	1.62560000000000,	-0.0526500000000000,	0,	0,	0,	3.54205000000000,	0.904360000000000,	-0.173640000000000,	4.86513000000000,	0,	0,	3.35554000000000,	0,	0	,0	,0	,0	,0.661170000000000,	0,	0,	0.533060000000000,	0,	0	,0	,0	,0	,0.541200000000000	,0	,0.541200000000000,	0	,0	,0	,-3.49802000000000,	0.759940000000000,	-0.326160000000000,	-5.02649000000000	,0	,0,	-3.36431000000000,	0,0,	0,	0	,0	,-0.730410000000000,	0,	0	,-0.588870000000000,0	,0,	0,	0	,0	,-0.597860000000000	,0	,0.597860000000000];
offset = reshape(offset,[3,38])';

expmapInd = linspace(4,117,114);
expmapInd = reshape(expmapInd,[3,38])';


L=38; % nombre de joint
Lr = 38;% nombre joint reduced

data_folder=dir(data_path);
data_folder=data_folder(3:end);
for i=1:length(data_folder)
    subj_folders=dir([data_path '\' data_folder(i).name]);
    subj_folders=subj_folders(3:end);
    mkdir([save_path '\' data_folder(i).name]);
    for j=1:length(subj_folders)
        if subj_folders(j).isdir
            mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name]);
            subq_folders=dir([data_path '\' data_folder(i).name '\' subj_folders(j).name]);
            subq_folders=subq_folders(3:end);
            for q=1:length(subq_folders)
                mkdir([save_path '\' data_folder(i).name '\' subj_folders(j).name '\' subq_folders(q).name]);
                skeleton = load([data_path '\' data_folder(i).name '\' subj_folders(j).name '\' subq_folders(q).name '\skeleton.mat']);
                skeleton = skeleton.curve_B;
                for t=1:fr
                    angles = skeleton(t,:);
                    xyz =[];
                    for o=1:L
                        if posInd(o,1)==-1
                            xangle=0;
                            yangle=0;
                            zangle=0;
                        else
                            xangle = angles(posInd(o,3));
                            yangle = angles(posInd(o,2));
                            zangle = angles(posInd(o,1));
                        end
                        r = angles(expmapInd(o,:));
                        theta = norm(r);
                        r0 = rdivide(r, theta + eps);
                        r0x = [0, -r0(3), r0(2), 0, 0, -r0(1), 0, 0, 0];
                        r0x = reshape(r0x,[3, 3]);
                        r0x = r0x - r0x';
                        R = eye(3, 3) + sin(theta) * r0x + (1 - cos(theta)) * (r0x*r0x);
                        thisRotation = R';
                        thisPosition = [xangle, yangle, zangle];
                        if parent(o) == 0
                            xyzStruct(o).rotation = thisRotation;
                            xyzStruct(o).xyz = reshape(offset(o ,:), [1 ,3]) + thisPosition;
                            xyz = [xyz;xyzStruct(o).xyz];
                        else
                            xyzStruct(o).xyz = (offset(o ,:) + thisPosition)*xyzStruct(parent(o)).rotation + xyzStruct(parent(o)).xyz;
                            xyzStruct(o).rotation = thisRotation*xyzStruct(parent(o)).rotation;
                            xyz = [xyz;xyzStruct(o).xyz];
                        end
                    end
                    xyz = squeeze(xyz);
                    xyz = [xyz(:,1),xyz(:,3),xyz(:,2)];
                    writematrix(xyz,[save_path '\' data_folder(i).name '\' subj_folders(j).name '\' subq_folders(q).name '\skeleton_pos_' sprintf('%03d',t) '.csv' ])
                end
            end
            
        end
    end
end
fprintf('Done \n\n')
end

