function [dist,q2n]=Max_mygeodWithoutSVD(q1,X2)
 
%input: two curves X1 and X2 as Nxn (ND curve)
%output: dist=distance, also will display when you run the program; X2n:
%optimally registered curve X2; q2n: same as X2n except in q-fun form; X1:
%normalized curve X1; q1: same as X1 except in q-fun form;

% Load some parameters, no need to change this
    lam = 0;  

%Center curves, not really needed but good for display purposes
%     X1 = X1 - repmat(mean(X1')',1,size(X1,2));
    X2 = X2 - repmat(mean(X2')',1,size(X2,2));    
    
% Form the q function for representing curves and find best rotation
    [q2] = curve_to_q(X2);
    [n] = size(q1,1);


% Applying optimal re-parameterization to the second curve
    %[gam] = DynamicProgrammingQ(q1/sqrt(InnerProd_Q(q1,q1)),q2/sqrt(InnerProd_Q(q2,q2)),lam,0);
    %gamI = invertGamma(gam);
    %gamI = (gamI-gamI(1))/(gamI(end)-gamI(1));
    %X2n = Group_Action_by_Gamma_Coord(X2,gamI);
    %q2n = curve_to_q(X2n);
    q2n=q2;

% Computing geodesic distance between the registered curves
N = size(q1,2);
dist = acos(sum(sum(q1.*q2n))/N);
%sprintf('The distance between the two curves is %0.3f',dist)