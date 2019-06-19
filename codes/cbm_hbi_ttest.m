function [p,stats] = cbm_hbi_ttest(cbm,k,m,d)
% uses the hierarchical errorbars computed by cbm_hbi to make an inference 
% about a parameter at the population level.
% 
% 1st input: the fitted cbm by cbm_hbi
% 2nd input: the index of the model of interest in the cbm file
% 3rd input: the test will be done compared with this value 
% (i.e. this value indicates the null hypothesis)
% 4th input: the index of the parameter of interest 
% 1st output: p-value of test
% 2nd output: stats contaning t-statistics, p-value and degrees of freedom
% 
% implemented by Payam Piray, Aug 2018
%==========================================================================
if ischar(cbm), fhbi=cbm; fhbi = load(fhbi); cbm = fhbi.cbm; end

qmutau  = cbm.math.qmutau(k);
a       = qmutau.a';

e       = qmutau.he';
nk      = qmutau.nk;
Dk      = length(a);

if nargin<4
    d = 1:Dk;
end
if length(m)~=length(d)
    error('the number of elements in m should be matched with the dimension of the corresponding model');    
end

t       = (a(d)-m)./e(d);
p       = 2 * tcdf(-abs(t), nk);

stats   = struct('tstat',t,'pval',p,'df',nk);
end