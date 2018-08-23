function optconfig = cbm_optim_config(d,optconfig)
% this function configures optimization algorithm in cbm_lap and cbm_hbi
% 
% implemented by Payam Piray, Aug 2018
%==========================================================================

if nargin<2, optconfig = []; end
if isempty(optconfig), optconfig = struct('verbose',1); end

p = inputParser;
p.addParameter('verbose',1);
p.addParameter('algorithm','quasi-newton',@(arg)strcmpi(arg,'trust-region')||strcmpi(arg,'quasi-newton') );
p.addParameter('gradient','off',@(arg)strcmpi(arg,'on')||strcmpi(arg,'off'));
p.addParameter('hessian','off',@(arg)strcmpi(arg,'on')||strcmpi(arg,'off'));
p.addParameter('ObjectiveLimit',-10^-10); % minimum possible

p.addParameter('range',[-5*ones(1,d);5*ones(1,d)],@(arg)valid_rng(d,arg));
p.addParameter('tolgrad',.001001,@(arg)isscalar(arg));
p.addParameter('tolgrad_liberal',.1,@(arg)(isvector(arg) || isempty(arg)));
p.addParameter('inits',[],@(arg)(ismatrix(arg) && (size(arg,2)==d) ));    
p.addParameter('numinit',min(7*d,100),@(arg)isscalar(arg));
p.addParameter('numinit_med',70,@(arg)isscalar(arg));
p.addParameter('numinit_up',100,@(arg)isscalar(arg));        
p.addParameter('prior_for_bads',1);

p.parse(optconfig);
optconfig    = p.Results;

if strcmpi(optconfig.algorithm,'trust-region') && strcmpi(optconfig.gradient,'off')
    error('For trust-region algorithm, gradient should be on, otherwise use quasi-newton algorithm');
end

end


function valid = valid_rng(d,arg)
valid = size(arg,1)==2 && (size(arg,2)==d || size(arg,2)==1);
end
