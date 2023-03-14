function [ distrib, distriba, distribb ] = LearningStep( directory, marqueur, npca )
%LEARNINGSTEP Summary of this function goes here
%   Detailed explanation goes here

% Process the learning step
% INPUT 1: directory of the training data
% INPUT 2: marqueur to use
% INPUT 3: number of principal components
% OUTPUT 1: distribution of full movement
% OUTPUT 1: distribution of first segment
% OUTPUT 1: distribution of second segment


% Learning
dir=directory;
cptTr1=0;
cptTr2=0;
for trial=1:40

    filename=sprintf('Raw_play_%.2i.txt',trial);
    f=fopen([dir filename]);
    if f>0
        fclose(f);
        data=load([dir filename]);

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

        if trial<=20
            cptTr1=cptTr1+1;
            %Full
            XsamplesTrain1{cptTr1}=Xab;
            Xab = Xab - repmat(mean(Xab')',1,size(Xab,2));
            [qab] = curve_to_q(Xab);
            QsamplesTrain1{cptTr1}=qab;
            %First segment
            XsamplesTraina1{cptTr1}=X1a;
            X1a = X1a - repmat(mean(X1a')',1,size(X1a,2));
            [q1a] = curve_to_q(X1a);
            QsamplesTraina1{cptTr1}=q1a;
            %Second segment
            XsamplesTrainb1{cptTr1}=X1b;
            X1b = X1b - repmat(mean(X1b')',1,size(X1b,2));
            [q1b] = curve_to_q(X1b);
            QsamplesTrainb1{cptTr1}=q1b;
        else
            cptTr2=cptTr2+1;
            %Full
            XsamplesTrain2{cptTr2}=Xab;
            Xab = Xab - repmat(mean(Xab')',1,size(Xab,2));
            [qab] = curve_to_q(Xab);
            QsamplesTrain2{cptTr2}=qab;
            %First segment
            XsamplesTraina2{cptTr2}=X1a;
            X1a = X1a - repmat(mean(X1a')',1,size(X1a,2));
            [q1a] = curve_to_q(X1a);
            QsamplesTraina2{cptTr2}=q1a;
            %Second segment
            XsamplesTrainb2{cptTr2}=X1b;
            X1b = X1b - repmat(mean(X1b')',1,size(X1b,2));
            [q1b] = curve_to_q(X1b);
            QsamplesTrainb2{cptTr2}=q1b;
        end
    end
end


%Full
[ Distribution1 ] = learnDistributionV( XsamplesTrain1,QsamplesTrain1, npca );
Distribution1.classe=1;
[ Distribution2 ] = learnDistributionV( XsamplesTrain2,QsamplesTrain2, npca);
Distribution2.classe=2;
distrib{1}=Distribution1;
distrib{2}=Distribution2;
%Fisrt segment
[ Distribution1a ] = learnDistributionV( XsamplesTraina1,QsamplesTraina1, npca );
Distribution1a.classe=1;
[ Distribution2a ] = learnDistributionV( XsamplesTraina2,QsamplesTraina2, npca );
Distribution2a.classe=2;
distriba{1}=Distribution1a;
distriba{2}=Distribution2a;
%Second segment
[ Distribution1b ] = learnDistributionV( XsamplesTrainb1,QsamplesTrainb1, npca );
Distribution1b.classe=1;
[ Distribution2b ] = learnDistributionV( XsamplesTrainb2,QsamplesTrainb2, npca );
Distribution2b.classe=2;
distribb{1}=Distribution1b;
distribb{2}=Distribution2b;


end

