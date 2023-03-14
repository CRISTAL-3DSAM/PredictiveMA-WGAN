clear all
clc
addpath curveframework/
% 1: Object, 2:RightIndex, 3:RightThumb, 4:RightHand, 5:RightWristL
% 6: RightWristR, 7:RightElbow, 8:RightShoulder, 9:LeftShoulder, 10:Head


marqueur=6;
npca=15;

% Learning
[distrib, distriba,distribb] = LearningStep('data/Train/', marqueur,npca);

% Test
id=1;
[classe,score,classeA,scoreA,classeB,scoreB]=TestStep('data/Test/',id,marqueur,distrib,distriba,distribb );
