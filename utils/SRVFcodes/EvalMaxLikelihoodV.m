function [ Loglikelihood ] = EvalMaxLikelihoodV( XT, distrib )
%EVALMAXLIKELIHOOD Summary of this function goes here
%   Detailed explanation goes here

q_mean=distrib.q_mean;
ncov=distrib.prinCov;
nU=distrib.prinBasis;
xU=distrib.excluBasis;

[dist,qTn]=Max_mygeodWithoutSVD(q_mean,XT);
vT = sphere_to_tangent( q_mean, qTn );
vT_t=reshape(vT',1,size(vT,1)*size(vT,2));
nvT=vT_t*nU;
xvT=vT_t*xU;

% Densite de proba
tmp=0;
eps=0.00001;
tmp=tmp+(nvT*inv(ncov)*nvT');
lxv=sqrt(sum(sum(xvT.*xvT))/size(xvT,2));
Loglikelihood= -tmp -(lxv^2)/(2*eps) -log(det(ncov));


