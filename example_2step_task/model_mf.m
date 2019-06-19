function [F] = model_mf(params,data)

fx = nan(1,7);
ip = [1 2 3 4 6 7];
fx(ip) = params;
fx(5) = -inf;    % w=0

[F] = model_hybrid(fx,data);
end