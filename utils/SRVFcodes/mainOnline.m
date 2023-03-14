clear all
clc

% Exemple de main pour appel online

addpath curveframework/
% 1: Object, 2:RightIndex, 3:RightThumb, 4:RightHand, 5:RightWristL
% 6: RightWristR, 7:RightElbow, 8:RightShoulder, 9:LeftShoulder, 10:Head


marqueur=6;


load distrib
load distriba
load distribb

% Test
% data get from online

[ classe, score ] = ClassifySequenceFull( data, marqueur, segment, distrib );
