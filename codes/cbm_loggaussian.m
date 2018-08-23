function [F,G,H] = cbm_loggaussian(theta,model,prior,data,Gconf,Hconf)
% computes log_gaussian of a joint model and prior
% 
% implemented by Payam Piray, Aug 2018
%==========================================================================
if nargin<5, Gconf = 'off'; end
if nargin<6, Hconf = 'off'; end

mode = 0;
if strcmpi(Gconf,'off') && strcmpi(Hconf,'off'), mode = 1; end
if strcmpi(Gconf,'on')  && strcmpi(Hconf,'off'), mode = 2; end
if strcmpi(Gconf,'on')  && strcmpi(Hconf,'on'), mode = 3; end
if mode==0
    error('Bad config of gradient and hessian');
end

d  = length(theta);
mu = prior.mean;

t  = theta';
T  = prior.precision;

logdetT      = 2*sum(log(diag(chol(T)))); % =log(det(T))
prior.logp   = -d/2*log(2*pi) +.5*logdetT -.5*(t-mu)'*T*(t-mu);
prior.dlogp  = -T*(t-mu);
prior.ddlogp = -T;

switch mode
    case 1
        [F] = model(theta,data); % returing logpdf(X|theta)
        G   = nan;
        H   = nan;
    case 2
        [F,G]   = model(theta,data); % returing logpdf(X|theta)
        H       = nan;
    case 3
        [F,G,H] = model(theta,data); % returing logpdf(X|theta)
end

F   = F + prior.logp;
G   = G + prior.dlogp';
H   = H + prior.ddlogp;

end