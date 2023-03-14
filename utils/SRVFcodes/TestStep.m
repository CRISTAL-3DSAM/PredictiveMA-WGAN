function [ classe, score, classeA, scoreA, classeB, scoreB ] = TestStep( directory, id, marqueur, distrib, distriba, distribb )
%TESTSTEP Summary of this function goes here
%   Detailed explanation goes here

%INPUT 1: directory of the test data
%INPUT 2: id of the sample
%INPUT 3: marqueur to use
%INPUT 4, 5, 6: distributions of full, segA et segB
%OUTPUTS: classe and score of each portion (FULL,A,B)

filename=sprintf('Raw_play_%.2i.txt',id);
f=fopen([directory filename]);
if f>0
    fclose(f);
    data=load([directory filename]);
    %data du marqueur
    c=((marqueur-1)*3)+1;
    ndata=data(:,c:c+2);
    ndata(isnan(ndata(:,1)),:)=[];
    X1=ReSampleCurve(ndata',400);
    %segmentation
    [maxtab, mintab]=peakdet(X1(3,:), 0.9);
    mintab(mintab(:,1)<50,:)=[];
    while (mintab(2,1)-mintab(1,1))<50
        mintab(2,:)=[];
    end
    mini1=[X1(:,mintab(1,1)) X1(:,mintab(2,1))];
    X1(:,mintab(2,1)+1:end)=[];
    Xa=X1(:,1:mintab(1,1));
    Xb=X1(:,mintab(1,1)+1:end);
    Xab=ReSampleCurve(X1,200);
    X1a=ReSampleCurve(Xa,100);
    X1b=ReSampleCurve(Xb,100);
    
    %Full
    for d=1:length(distrib)
        [ maxlikelihood ] = EvalMaxLikelihoodV( Xab, distrib{d} );
        likelihoods(d)=maxlikelihood;
    end
    [score ind]=max(likelihoods);
    classe = distrib{ind}.classe;

    %Fisrt segment
    for d=1:length(distriba)
        [ maxlikelihooda ] = EvalMaxLikelihoodV( X1a, distriba{d} );
        likelihoodsa(d)=maxlikelihooda;
    end
    [scoreA ind]=max(likelihoodsa);
    classeA=distriba{ind}.classe;
    
    %Second segment
    for d=1:length(distribb)
        [ maxlikelihoodb ] = EvalMaxLikelihoodV( X1b, distribb{d} );
        likelihoodsb(d)=maxlikelihoodb;
    end
    [scoreB ind]=max(likelihoodsb);
    classeB=distribb{ind}.classe;

end

end

