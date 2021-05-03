function [cbm, success] = cbm_lap(data, model, prior, fname, pconfig)
% laplace approximation
% [CBM, SUCCESS] = cbm_lap(DATA, MODEL, PRIOR, FNAME, PCONFIG)
% DATA:  data (Nx1 cell) where N is number of samples
% MODEL: a function-handle to the model computing log-likelihood of DATA 
% given some paramete 
% PRIOR: a struct with two fields: mean and variance, as the gaussian prior
%   PRIOR.mu is a d-by-1 vector, where d is the number of parameters
%   (dimension)
%   PRIOR.variance is scaler or a vector or matrix indicating prior
%   variance. If it is scaler, prior variance is diagonal with 
%   PRIOR.variance as diagonal elements. If it is a vector, prior variance
%   is diag(PRIOR.variance).
% FNAME: filename for saving the output (leave it empty for not saving)
% PCONFIG: a struct for configuration (optional), 
% see below for more info
% CBM: the main output-file with following fields:
%   CBM.output is output field. 
%       CBM.output.parameters is N-by-d matrix containing fitted parameters.
%       CBM.output.log_evidence is N-by-1 vector containing (laplace
%       approximation) of log-model evidence for each subject. 
%   CBM.input contains inputs to cbm_lap
%   CBM.profile contains info about optimization
%   CBM.math contains all relevant variables 
% 
% Important configurations (see cbm_optim_config)
% numinit: number of random initialization, default is min(7*d,100)
% prior_for_bads: a 0 or 1 scaler indicating what cbm_lap does with bad 
% subjects that no good gradient found for them. If it is 1, prior values
% will be saved as individual parameters; if it is 0, the algorithm stops. 
% default is 1.
% numinit_med is the (increased) number of random initializations for bad
% subjects; default is 100.
% numinit_up is the (maximum) number of random initializations for bad
% subjects; default is 1000.
% tolgrad: tolerance of gradient at optimal value (theoretically should be
% zero, in practice it is usually close to zero), default is 0.001.
% tolgrad_liberal: is a vector indicating the values of liberal tolerance 
% of gradient for bad subjects; default is 0.1
% range: a 2-by-d matrix representing the range of parameters. random seeds
% for initializations will be picked in this range; default [-5*ones(1,d);5*ones(1,d)]
% verbose: whether to write on screen (or a log-file), default is 1
% 
% dependecies:
% this function needs MATLAB optimization toolbox (calls fminunc)
% cbm_optim_config, cbm_check_input, cbm_quad_lap, cbm_optim
% 
% implemented by Payam Piray, Aug 2018
%==========================================================================
                
% evaluation if prior
mu = prior.mean;
v  = prior.variance;
d  = length(mu); % number of parameters
if numel(v)==1, v = v*eye(d); end
if isvector(v), v = diag(v); end
if isrow(mu), mu = mu'; end
a = inv(v);
prior = struct('mean',mu,'precision',a);
N   = length(data); % number of samples (subjects)

%--------------------------------------------------------------------------
% configuration using cbm_optim_config
if nargin<3, error('Not enough inputs'); end
if nargin<4, fname   = []; end
if nargin<5, pconfig =[]; end
pconfig     = cbm_optim_config(d,pconfig);
rng         = pconfig.range;
numinit     = pconfig.numinit;
verbose     = pconfig.verbose;

fid  = 1;

%--------------------------------------------------------------------------
% Initial report
tic;
if verbose, fprintf(fid,'%-40s%30s\n',mfilename,datestr(now)); end
if verbose, fprintf(fid,'%-70s\n',repmat('=',1,70)); end

% dimensions
if verbose, fprintf(fid,'Number of samples: %d\n',N); end
if verbose, fprintf(fid,'Number of parameters: %d\n\n',d); end
if verbose, fprintf(fid,'Number of initializations: %d\n',numinit); end
if verbose, fprintf(fid,'%-70s\n',repmat('-',1,70)); end

%----------------------------------
% test the model
ok = cbm_check_input(mu',model,data,fname);
if ~ok
else
    if fid~=1, fprintf(fid,'Model is no good at the mean prior! Have to stop here, sorry!\n'); end
end

%--------------------------------------------------------------------------
% The main algorithm

flags     = nan(1,N);
loglik    = nan(1,N);
theta     = cell(1,N);
A         = cell(1,N);
G         = nan(d,N);
lme       = nan(1,N); % log-model-evidence
Ainvdiag  = cell(1,N);
logdetA   = nan(1,N); % log-model-evidence

success   = 1;

tic;        
for n=1:N
    if verbose, fprintf(fid,'Subject: %02d\n',n); end
    dat = data{n};    
    [loglik_n,theta_n,A_n,G_n,flag_n] = cbm_quad_lap(dat,model,prior,pconfig,0,'LAP');
    
    if flag_n~=1 && ~isempty(pconfig.tolgrad_liberal)
        l = 1;
        theta_n_l{l}   = theta_n; %#ok<AGROW>
        loglik_n_l{l}  = loglik_n;  %#ok<AGROW>
        A_n_l{l}       = A_n;  %#ok<AGROW>
        G_n_l{l}       = G_n;  %#ok<AGROW>
        flag_n_l(l)    = flag_n; %#ok<AGROW>
        
        for tg=1:length(pconfig.tolgrad_liberal)
            if flag_n~=1
                l = l+1;
                tolgrad = pconfig.tolgrad_liberal(tg);
                if verbose, fprintf(fid,'\nbad subject %02d ... use liberal tolgrad %0.4f\n',n, tolgrad); end
                ptconfig = pconfig;
                ptconfig.tolgrad = tolgrad;
                [loglik_n,theta_n,A_n,G_n,flag_n] = cbm_quad_lap(dat,model,prior,ptconfig,0,'LAP');                             
                theta_n_l{l} = theta_n;
                loglik_n_l{l} = loglik_n;
                A_n_l{l}  = A_n;
                G_n_l{l}  = G_n;
                flag_n_l(l) = flag_n;
            end
        end
        
        l = flag_n_l>0;
        l = find(l, 1 );
        if isempty(l) % i.e. failed
            l = 1; 
        end
        theta_n      = theta_n_l{l};
        loglik_n     = loglik_n_l{l};
        G_n          = G_n_l{l};
        A_n          = A_n_l{l};
        flag_n       = flag_n_l(l);                
    end
    
    if(~flag_n)
        if verbose, fprintf(fid,'No minimum found for subject %02d\n',n); end
        if pconfig.prior_for_bads
            if verbose, fprintf(fid,'No minimum found, use prior values as individual parameters\n'); end
            theta_n   = mu;
            [loglik_n]  = cbm_loggaussian(theta_n',model,prior,dat);
            A_n       = a;
            G_n       = nan(1,d);
        else
            cbm     = sprintf(fid,'No minimum found for subject %02d\n',n);
            success = 0;
            return;
        end
    end
    flags(n)     = flag_n;
    theta{n}     = theta_n;
    loglik(n)    = loglik_n;
    A{n}         = A_n;
    Ainvdiag{n}  = diag(A_n^-1);
    G(:,n)       = G_n';    
    logdetAn     = 2*sum(log(diag(chol(A_n)))); % = log(det(A_n));
    logdetA(n)   = logdetAn;
    
    lme(n)       = loglik_n +.5*d*log(2*pi) -.5*logdetAn;
end

telapsed = toc;

%--------------------------------------------------------------------------
% output

sdata      = {}; %if pconfig.save_data, sdata = data; end
input      = struct('data',{sdata},'model',func2str(model),...
                    'prior',prior,'config',pconfig,'fname',fname);
math       = struct('loglik',loglik,'theta',{theta},'A',{A},'lme',lme,'Ainvdiag',{Ainvdiag},'logdetA',{logdetA});
optim      = struct('numinit',numinit,'range',rng,'telapsed',telapsed,'flag',flags,'gradient',G);
profile    = struct('datetime',datestr(now),'filename',mfilename,'optim',optim);
output     = struct('parameters',cell2mat(theta)','log_evidence',lme');
cbm        = struct('method',mfilename,...
                    'input',input,...
                    'profile',profile,...
                    'math',math,...                    
                    'output',output);
if ~isempty(fname), save(fname,'cbm'); end
if verbose, fprintf(fid,'done :]\n'); end
if any(fid~=1), fclose(fid); end
end

