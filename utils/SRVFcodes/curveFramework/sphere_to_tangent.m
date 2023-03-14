function [ v2 ] = sphere_to_tangent( q1, q2 )
%SPHERE_TO_TANGENT Summary of this function goes here
%   Project a point q2 to the tangent space of q1 resulting in v2
%   Logarithmic map
%   Formule: log_q1(q2) = u/sqrt(inner_prod(u)) * acos(inner_prod(q1,q2)),
%   with u = q2 - inner_prod(q1,q2)*q1

[n,N]=size(q1);
prod = sum(sum(q1.*q2))/N;
u = q2 - q1.*prod;
lu = sqrt(sum(sum(u.*u))/N);
theta = acos(prod);
if lu==0
    v2=u.*0;
else
    v2 = u.*(theta/lu);
end
v2=real(v2);

u2= q2 - q1.*cos(theta);






