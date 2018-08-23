function [cbm,cbm0] = cbm_hbi_null(data,fname_cbm)
% cbm_hbi_null implements hierarchical Bayesian inference (HBI) under the
% null hypothesis
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

cbm0 = cbm_hbi(data,models,fcbm_maps,fname0,config,optimconfigs,isnull);

% use cbm0 to compute protected exceedance probability
cbm.exceedance  = cbm_hbi_exceedance(cbm.math,cbm0.math);

% update cbm
cbm.output.protected_exceedance_prob = cbm.exceedance.pxp;
if inputisfile, save(fname,'cbm'); end
end