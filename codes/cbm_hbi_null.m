function [cbm,cbm0] = cbm_hbi_null(data,fname_cbm)
% cbm_hbi_null implements hierarchical Bayesian inference (HBI) under the
% null hypothesis
%       cbm = cbm_hbi_null(data,fname_cbm)
% 1st input: data for all subjects
% 2nd input: the fitted cbm (or its file-address) by cbm_hbi
% 
% implemented by Payam Piray, Aug 2018
%==========================================================================

inputisfile = 0;
if ischar(fname_cbm), inputisfile = 1; fname=fname_cbm; fcbm = load(fname); cbm = fcbm.cbm; end

if inputisfile
[fdir,fname0] = fileparts(fname);
fname0 = fullfile(fdir,sprintf('%s_null.mat',fname0));
end

models       = cbm.input.models;
fcbm_maps    = cbm.input.fcbm_maps;
config       = cbm.input.config;
optimconfigs = cbm.input.optimconfigs;
isnull       = 1;

if isfield(config,'flog')
    if ischar(config.flog)
        [fdir,flog,flogext]=fileparts(config.flog);
        flog = fullfile(fdir,sprintf('%s_null%s',flog,flogext));
        config.flog = flog;
    end
end

if isfield(config,'fname_prog')
    if ischar(config.fname_prog)
        [fdir,fname_prog]=fileparts(config.fname_prog);
        fname_prog = fullfile(fdir,sprintf('%s_null.mat',fname_prog));
        config.fname_prog = fname_prog;
    end
end

%*********

%--------------------------------------------------------------------------
% save the input structure
user_input = struct('models',{models},'fcbm_maps',{fcbm_maps},'fname',fname0,...
                    'config',config,'optimconfigs',optimconfigs);

%--------------------------------------------------------------------------
% hyper (prior) parameters
b = 1; v = 0.5; s = 0.01;
% Note: a0 is the same as the prior mean used in each fcbm_map
hyper = struct('b',b,'v',v,'s',s);
isnull = 1;

config = cbm_hbi_config(config);
[inits,priors]= cbm_hbi_init(fcbm_maps,hyper,isnull,config.initialize);

%--------------------------------------------------------------------------
% run HBI
cbm0 = cbm_hbi_hbi(data,user_input,inits,priors);
%*********

% use cbm0 to compute protected exceedance probability
alpha = cbm.math.qm.alpha;
L = cbm.math.bound.bound.L;
L0 = cbm0.math.bound.bound.L;
cbm.exceedance  = cbm_hbi_exceedance(alpha,L,L0);

% update cbm
cbm.output.protected_exceedance_prob = cbm.exceedance.pxp;
if inputisfile, save(fname,'cbm'); end
end