function cbm = cbm_lap_aggregate(fname_subjs,fname_agg)
% aggregates individual cbms, each fitted by cbm_lap
% 
% implemented by Payam Piray, Aug 2018
%==========================================================================

save_agg = 0;
if nargin>1, save_agg = 1; end

N         = length(fname_subjs);
data      = cell(N,1);

flags     = nan(1,N);
telapsed  = nan(1,N);

loglik    = nan(1,N);
theta     = cell(1,N);
A         = cell(1,N);
lme       = nan(1,N); % log-model-evidence
Ainvdiag  = cell(1,N);
logdetA   = nan(1,N);

for n=1:N
    cbm = load(fname_subjs{n}); cbm = cbm.cbm;
    % input field
    if ~isempty(cbm.input.data)
        data(n)  = cbm.input.data;
    end
    if n==1
        model          = cbm.input.model;
%         functionname   = cbm.input.functionname;
        prior          = cbm.input.prior;
        config         = cbm.input.config;
        fname          = [];
    end
    % profile field
    flags(n)     = cbm.profile.optim.flag;
    gradient(:,n)= cbm.profile.optim.gradient; %#ok<AGROW>
    telapsed(n)  = cbm.profile.optim.telapsed;
    if n==1
        numinit  = cbm.profile.optim.numinit;
        range    = cbm.profile.optim.range;
    end
    
    % math field
    loglik(n)    = cbm.math.loglik;
    theta(n)     = cbm.math.theta;
    A(n)         = cbm.math.A;
    lme(n)       = cbm.math.lme;
    Ainvdiag(n)  = cbm.math.Ainvdiag;
    logdetA(n)   = cbm.math.logdetA;
    
    % output field
    lme(n) = cbm.output.log_evidence;    
end

input      = struct('data',{data},'model',model,...
                    'prior',prior,'config',config,'fname',fname);
math       = struct('loglik',loglik,'theta',{theta},'A',{A},'lme',lme,'Ainvdiag',{Ainvdiag},'logdetA',{logdetA});                
optim      = struct('numinit',numinit,'range',range,'telapsed',telapsed,'flag',flags,'gradient',gradient);
profile    = struct('datetime',datestr(now),'filename',mfilename,'optim',optim);
output     = struct('parameters',cell2mat(theta)','loglik',loglik','log_evidence',lme');
cbm        = struct('method',mfilename,...
                    'input',input,...
                    'profile',profile,...
                    'math',math,...                    
                    'output',output);
if save_agg
    save(fname_agg,'cbm');
end
fprintf('Aggregation is done over %d subjects :]\n',N);
end