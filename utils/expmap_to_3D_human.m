function expmap_to_3D_human(Data_path,fr)
% angle based representation to 3D representation

addpath utils\natsort
data_path= [Data_path '\expmap\expmap_Check']; %SRVF path
save_path= [Data_path '\3D\Skeletons'];
mkdir([Data_path '\3D']);

fprintf('creating 3D coordinates \n')
parent = [0, 1, 2, 3, 4, 5, 1, 7, 8, 9,10, 1,12,13,14,15,13, 17,18,19,20,21,20,23,13,25,26,27,28,29,28,31];
offset = [0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000];
offset = reshape(offset,[3,32])';
rotInd_old = [5, 6, 4;8, 9, 7;11, 12, 10;14, 15, 13;17, 18, 16;-1,-1,-1;20, 21, 19;23, 24, 22;26, 27, 25;29, 30, 28;-1,-1,-1;32, 33, 31;35, 36, 34;38, 39, 37;41, 42, 40;-1,-1,-1;44, 45, 43;47, 48, 46;50, 51, 49;53, 54, 52;56, 57, 55;-1,-1,-1;59, 60, 58;-1,-1,-1;62, 63, 61;65, 66, 64;68, 69, 67;71, 72, 70;74, 75, 73;-1,-1,-1;77, 78, 76;-1,-1,-1];

rotInd = [5, 6, 4;8, 9, 7;11, 12, 10;14, 15, 13;17, 18, 16;20, 21, 19;23, 24, 22;26, 27, 25;29, 30, 28;32, 33, 31;35, 36, 34;38, 39, 37;41, 42, 40;44, 45, 43;47, 48, 46;50, 51, 49;53, 54, 52;56, 57, 55;59, 60, 58;62, 63, 61;65, 66, 64;68, 69, 67;71, 72, 70;74, 75, 73;77, 78, 76;80,81,79;83,84,82;86,87,85;89,90,88;92,93,91;95,96,94;98,99,97];


expmapInd = linspace(4,99,96);
expmapInd = reshape(expmapInd,[3,32])';



L=33; % nombre de joint
Lr = 33;% nombre joint reduced

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
                    for o=1:32
                        if rotInd(o,1)==-1
                            xangle=0;
                            yangle=0;
                            zangle=0;
                        else
                            xangle = angles(rotInd(o,1));
                            yangle = angles(rotInd(o,2));
                            zangle = angles(rotInd(o,3));
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
fprintf('3D coordinates created \n \n')
end

