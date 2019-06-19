function [F] = model_mb(params,data)

fx = nan(1,7);
ip = [2 4 6 7];
fx(ip) = params;
fx(1) = -inf; % i.e alpha1=0
fx(3) = -inf; % lambda=0
fx(5) = inf;    % w=1

[F] = model_hybrid(fx,data);
end