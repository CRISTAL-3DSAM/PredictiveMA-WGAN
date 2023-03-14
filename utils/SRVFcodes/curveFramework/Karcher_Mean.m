function [ q_mean ] = Karcher_Mean( q_samples, x_samples, iterKM, epsilon )
%KARCHER_MEAN Summary of this function goes here
%   Detailed explanation goes here

% Seul modif le seuil pour l'arrêt du Karcher mean
iterKarcherMean=iterKM;
nbre_traj = length(q_samples);
q_mean = q_samples{1};    
%%%%%%%%%%%%%%%%% karcher mean between curves
%cost = 0;
for j=1:iterKarcherMean%% nbre iteration de karcher mean juste pour converger                            
        v_mean = 0;
        distance = 0;
        for i=1:nbre_traj % boucle sur nbre objets a moyenner                                
                        q1 = q_mean;
                        X2 = x_samples{i};
                        [dist,q2n]=Max_mygeodWithoutSVD(q1,X2);                        
                        if(dist>0)  
                                vn=sphere_to_tangent(q_mean,q2n);
                                v_mean =  v_mean + vn / nbre_traj;                                                     

                        end                             
        end   
        Err=sqrt(sum(sum(v_mean.*v_mean))/size(v_mean,2));
        Error(j)=real(Err);
        qt = Geodesic_Flow(q_mean,real(v_mean),epsilon); 
        q_mean = qt;
        if Error(j)<0.01
            break;
        end
end