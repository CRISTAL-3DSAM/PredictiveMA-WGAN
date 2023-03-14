function qt = Geodesic_Flow(q1,w,delta)

[n,T] = size(q1);
qt = q1;


lw = sum(sum(w.*w))/T;
lw = sqrt(lw);%InnerProd_Q(w,w));
if(lw < 0.001)
    display('small_tg_vector');
    return;
end

  qt  = q1 .* (cos(delta*lw)) + w .* (  (sin(delta*lw))/(lw)  );

return;