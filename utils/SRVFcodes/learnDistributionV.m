function [ Distribution ] = learnDistributionV( Xsamples,Qsamples,numComp )
%LEARNDISTRIBUTION Summary of this function goes here
%   Detailed explanation goes here
%%
% Learn a multivariate normal distribution on samples of trajectories on
% the Shape space.
% INPUT:
% Xsamples: Set of trajectories in the Action space
% Qsamples: Set of trajectories projected in the Shape space
% numComp: Number of principal components to use in PCA
% OUTPUT:
% q_mean: Average trajectory in the Shape space
% prinCov: Covariance matrix of the principal basis
% prinBasis: Principal basis build from 'numComp' principal vectors
% excluBasis: Not principal basis from vectors 'numComp'+1 to M.

%% Compute the mean
[ q_mean ] = Karcher_Mean( Qsamples, Xsamples, 50, 0.9 );

%% Project the q elements into the tangent space of mean to obtain Vsamples
for s=1:length(Xsamples)
    [dist,qn]=Max_mygeodWithoutSVD(q_mean,Xsamples{s});
    v = sphere_to_tangent( q_mean, qn );
    v_t=reshape(v',1,size(v,1)*size(v,2));
    Vsamples{s}=v_t;
end

%% Compute Covariance Matrix and apply SVD (equivalent to PCA)
cov=0;
for s=1:length(Vsamples)
    v=Vsamples{s};
    cov=cov+(v'*v);
end
cov=cov./length(Vsamples);
[U,S,V]=svd(cov);

%% Compute principal basis and exclusion basis
% principal base
prinBasis=U(:,1:numComp);
% exclusion base
excluBasis=U(:,numComp+1:end);

%% Project vectors into principal basis and compute covariance matrix
prinCov=0;
for s=1:length(Vsamples)
    v=Vsamples{s};
    nv=v*prinBasis;
    prinCov=prinCov+(nv'*nv);
end
prinCov=prinCov/(length(Vsamples));

%% Put parameters into Distribution
Distribution.q_mean=q_mean;
Distribution.prinCov=prinCov;
Distribution.prinBasis=prinBasis;
Distribution.excluBasis=excluBasis;