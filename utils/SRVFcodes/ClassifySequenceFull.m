
function [ classe, likelihood ] = ClassifySequenceFull( data, marqueur, segment, distrib )
%CLASSIFYSEQUENCEFULL Summary of this function goes here
%   Detailed explanation goes here

% Classify a full sequence corresponding to the whole movement (reach, put
% and return.
% INPUT1: data corresponding to the trajectory (3xF) of length F
% INPUT2: marqueur to use
% INPUT3: segment value: 1 corresponding to first segment (reach), 2
% corresponding to second segment (put), 3 corresponding to both segments
% INPUT4: distribution corresponding to the segment (1,2 or 3)
% OUTPUT1: predicted classe, 1 for personnal, 2 for social
% OUTPUT2: likelihood corresponding to the prediction

% Get data from corresponding marqueur
c=((marqueur-1)*3)+1;
ndata=data(:,c:c+2);
data=ndata;

% Removing nan values if any
data(isnan(data(:,1)),:)=[];

% Resample Trajectory
X1=ReSampleCurve(data',400);

% Detection of minimas and segmentation
[maxtab, mintab]=peakdet(X1(3,:), 0.9);
mintab(mintab(:,1)<50,:)=[];
while (mintab(2,1)-mintab(1,1))<50
    mintab(2,:)=[];
end
X1(:,mintab(2,1)+1:end)=[];

% Selection of corresponding segment to evaluate
switch segment
    case 3 % Both segments
        X2=X1;
        X=ReSampleCurve(X2,200);     
    case 1 % First segment
        X2=X1(:,1:mintab(1,1));
        X=ReSampleCurve(X2,100);   
    case 2 % Second segment
        X2=X1(:,mintab(1,1)+1:end);
        X=ReSampleCurve(X2,100);   
end

% Classification
for d=1:length(distrib)
    [ maxlikelihood ] = EvalMaxLikelihoodV( X, distrib{d} );
    likelihoods(d)=maxlikelihood;
end
[val, ind]=max(likelihoods);
classe=distrib{ind}.classe;
likelihood=likelihoods(ind);
end

