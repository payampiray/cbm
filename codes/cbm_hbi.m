function cbm = cbm_hbi(data,models,fcbm_maps,fname,config,optimconfigs)
% cbm_hbi implements hierarchical Bayesian inference (HBI)
%       cbm = cbm_hbi(data,models,fcbm_maps,fname,config,optimconfigs)
% 1st input: data for all subjects
% 2nd input: a cell input containing function handle to models
% 3rd input: another cell input containing file-address to files saved by cbm_lap
% 4th input: a file address for saving the output (optional)
% 5th input: is a struct, which configures hbi algorithm (optional)
%   see cbm_hbi_config    
% 6th input: is another struct, which configures optimization algorithm (optional)
%   see cbm_optim_config
% output: cbm struct containing the output of HBI
%   cbm.methid is 'hbi'
%   cbm.input contains inputs to the cbm_hbi (but not the data)
%   cbm.profile contains info about how HBI is configured
%   cbm.math contains all details (the notation is the same as the Appendix
%       of the HBI paper cited below)
%   cbm.output contains all useful output variables
%       cbm.output.parameters contains individual parameters for each
%       subject and model
%       cbm.output.responsibility is the probability that each model is
%       responsible for generating corresponding individual dataset
%       cbm.output.group_mean is the group mean parameters
%       cbm.output.group_hierarchical_errorbar is the errorbar of the group
%       mean parameters
%       cbm.output.model_frequency is the estimate of how much each model 
%       is expressed across the group (normalized). It sums to 1 across all
%       models). 
%       cbm.output.exceedance_prob is the exceedance probability that each
%       model is the most likely model across the group
%       cbm.output.protected_exceedance_prob is the protected exceedance 
%       probability that each model is the most likely model across the 
%       group taking into account the null hypothesis that no model is more
%       likely across the group (this is NaN until running cbm_hbi_null)
% 
% Reference:
% Piray et al. (2018), Hierarchical Bayesian inference for concurrent model 
% fitting and comparison for group studies
% https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007043
% 
% implemented by Payam Piray, April 2019 
%==========================================================================

if nargin<4, fname= []; end
if nargin<5, config= []; end
if nargin<6, optimconfigs = []; end

%--------------------------------------------------------------------------
% save the input structure
user_input = struct('models',{models},'fcbm_maps',{fcbm_maps},'fname',fname,...
                    'config',config,'optimconfigs',optimconfigs);

%--------------------------------------------------------------------------
% hyper (prior) parameters
b = 1; v = 0.5; s = 0.01;
% Note: a0 is the same as the prior mean used in each fcbm_map
hyper = struct('b',b,'v',v,'s',s);
isnull = 0;

config = cbm_hbi_config(config);
[inits,priors]= cbm_hbi_init(fcbm_maps,hyper,isnull,config.initialize);

%--------------------------------------------------------------------------
% run HBI
cbm = cbm_hbi_hbi(data,user_input,inits,priors);

end