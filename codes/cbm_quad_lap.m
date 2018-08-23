function [loglik,m,A,G,flag,kopt] = cbm_quad_lap(data,model,prior,config,fid,mode)
% quadratic approximation using laplace approximation
% 
% implemented by Payam Piray, Aug 2018
%==========================================================================

switch mode
    case {'LAP'}
        [loglik,m,A,G,flag,kopt] = quad_map(data,model,prior,config,fid);
    otherwise
        error('mode %s is not recognized!',mode);
end
end

function [loglik,x,A,G,flag,kopt] = quad_map(data,model,prior,config,fid)
pconfig = config;
pconfig.chance_v = prior.precision^-1;
numinit  = pconfig.numinit;
init0    = pconfig.inits;
rng      = pconfig.range;

Gconf =  pconfig.gradient;
Hconf =  pconfig.hessian;

init  = [prior.mean'; init0];
hfunc = @(theta)(neglogloggaussian(theta,model,prior,data,Gconf,Hconf));
[tx,negloglik,H,G,flag,kopt] = cbm_optim(hfunc,pconfig,rng,numinit,init,fid);
if flag == 0
    d  = length(prior.mean);
    tx = nan(1,d);
    H  = nan(d,d);
    G  = nan(d,1);
end
loglik = -negloglik;
x      = tx';
A      = H;
G      = G';
end

function [F,G,H] = neglogloggaussian(theta,model,prior,data,Gconf,Hconf)
[F,G,H] = cbm_loggaussian(theta,model,prior,data,Gconf,Hconf);
F = -F;
G = -G;
H = -H;
end
