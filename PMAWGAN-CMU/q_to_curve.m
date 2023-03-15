function [ curve ] = q_to_curve( q )
%Q_TO_CURVE Summary of this function goes here
%   Detailed explanation goes here


[L,F]=size(q);
s=linspace(0,1,F);
qnorm=zeros(F,1);
for i=1:F
    qnorm(i)=norm(q(:,i));
end



for i=1:L
    temp=q(i,:).*qnorm';
    curve(i, :)=cumtrapz(s,temp);

end
% curve(:, end)
% size(curve)


end



