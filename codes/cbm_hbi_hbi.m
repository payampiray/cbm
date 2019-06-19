function cbm = cbm_hbi_hbi(data,user_input,inits,priors)

models     = user_input.models;
fcbm_maps  = user_input.fcbm_maps;
fname      = user_input.fname;
config     = user_input.config;
optconfigs = user_input.optimconfigs;


K = length(models);
N = length(data);

%--------------------------------------------------------------------------
% initialize using fcbm_maps

qhquad = inits.qh;
r      = inits.r;
bound  = inits.bound;

hyper  = priors.hyper;
pmutau = priors.pmutau;
pm     = priors.pm;

isnull = pm.limInf;
%--------------------------------------------------------------------------
% configuration, see cbm_hbi_config for more information

config      = cbm_hbi_config(config);
flog        = config.flog; % log file
fname_prog  = config.fname_prog; % fname for saving all iterations
save_prog   = config.save_prog;  % saves in fname_prog if this is 1
verbose     = config.verbose;
maxiter     = config.maxiter;
tolx        = config.tolx;
opt_parallel= config.parallel;

% configuration of the log-file

% if no log-file address is passed, but fname is not empty, create a
% log-file address using fname
% NOTE: if flog is set to -1, no log-file will be created anyway
if isempty(flog) && ~isempty(fname)
    [fdir,fn] = fileparts(fname);
    flog = fullfile(fdir,sprintf('%s.log',fn));
end

fid = 1; 
if flog~=-1, if ~isempty(flog), fid = fopen(flog,'w'); end; end

verbose_multiK = verbose && (K>1) && ~isnull;
fid_multiK     = 0;
if K>1  && ~isnull, fid_multiK = fid; end

%--------------------------------------------------------------------------
% configuration of optimization procedure according to optimconfigs input, 
% see cbm_optim_config for more information

if length(optconfigs)==1, optconfigs = repmat(optconfigs,K,1); end
if ~isempty(optconfigs) && length(optconfigs)~=K
    logging(verbose,fid,sprintf('optimconfigs input is not matched with the number of models (and it is not 1)\n'));                
    error('optimconfigs input is not matched with the number of models (and it is not 1)\n');
end

for k=1:K
    d     = length(pmutau(k).a);
    optfigk = [];
    if ~isempty(optconfigs)
        optfigk = optconfigs(k);
    end
    if ~isfield(optfigk,'numinit'), optfigk.numinit = 0; end
    if ~isfield(optfigk,'verbose'), optfigk.verbose = 0; end
    optconfigk = cbm_optim_config(d,optfigk);
    config.optconfigs(k,1) = optconfigk;
end

%--------------------------------------------------------------------------
% Initial report
logging(verbose,fid,sprintf('%-40s%30s\n',mfilename,datestr(now)));
ss = sprintf('Running hierarchical bayesian inference (HBI)');
if isnull, ss = sprintf('%s- null mode',ss); end
ss = sprintf('%s...',ss);
logging(verbose,fid,sprintf('%s\n\n',ss));

logging(verbose,fid,sprintf('HBI has been initialized according to\n'));
for k=1:K
    logging(verbose,fid,sprintf('\t%s [for model %d]\n',fcbm_maps{k},k));
end
logging(verbose,fid,sprintf(' \n'));

logging(verbose,fid,sprintf('Number of samples: %d\n',N));
logging(verbose,fid,sprintf('Number of models: %d\n',K));
logging(verbose,fid,sprintf('%-70s\n',repmat('=',1,70)));

%--------------------------------------------------------------------------
% HBI algorithm

% for monitoring progress and termination criteria
prog   = struct('L',bound.bound.L,'alpha',pm.alpha,'x',nan);

terminate  = 0;
iter = 0;
while ~terminate && iter<=maxiter
    iter = iter + 1;

    logging(verbose,fid,sprintf('Iteration %02d\n',iter));
    
    % 1: Calculate the summary statistic
    [Nbar, thetabar, Sdiag] = hbi_sumstats(r,qhquad);
    
    % 2: Update parameters of q(mu,tau,m)
    [qmutau,bound.qmutau] = hbi_qmutau(pmutau,Nbar,thetabar,Sdiag);
    [bound] = hbi_bound(bound,'qmutau');
    [qm,bound.qm] = hbi_qm(pm,Nbar);
    [bound] = hbi_bound(bound,'qm');    

    % 3: Update individual posteriors
    [qhquad] = hbi_qhquad(models,data,config.optconfigs,qmutau,qhquad,fid,opt_parallel);
    
    % 4: Update responsibilities
    [r,bound.qHZ] = hbi_qHZ(qmutau,qm,qhquad,thetabar,Sdiag);    
    [bound] = hbi_bound(bound,'qHZ');
        
    % 5: check convergence criteria
    [dprog,prog] = hbi_prog(prog,bound.bound.L,qm.alpha,thetabar,Sdiag);
    if dprog.dx<tolx
        terminate = 1;        
    end
    
    % 5: report summary and save 
    if iter>1
        
        % print model frequencies
        logging(verbose_multiK,fid_multiK,sprintf('\tmodel frequencies (percent)'));
        ss = '';        
        for k=1:K, ss = sprintf('%smodel %d: %2.1f| ',ss,k,Nbar(k)/N*100); end
        logging(verbose_multiK,fid_multiK,sprintf('\n\t%s\n',ss));

        % print change in the lower bound
        logging(verbose,fid,sprintf('%-40s%30s\n',' ',sprintf('dL: %7.2f',dprog.dL)));
        
        % print change in the alpha (i.e model frequency) in percent
        logging(verbose_multiK,fid_multiK,sprintf('%-40s%30s\n',' ',sprintf('dm: %7.2f',dprog.dalpha/N*100)));
        
        % print change in the parameters
        logging(verbose,fid,sprintf('%-40s%30s\n',' ',sprintf('dx: %7.2f',dprog.dx)));
        
        if terminate
            logging(verbose,fid,sprintf('%-40s%30s\n',' ','Converged :]'));
        end
    end    
    math(iter)   = struct('qhquad',qhquad,...
                          'r',r,...
                          'Nbar',Nbar,'thetabar',{thetabar},'Sdiag',{Sdiag},...
                          'pm',pm,'pmutau',pmutau,'qm',qm,'qmutau',qmutau,...                          
                          'bound',bound,'prog',prog,'dprog',dprog,...
                          'input',user_input,'hyper',hyper); %#ok<AGROW>
    if save_prog, save(fname_prog,'math'); end
    
end

math = math(end);

%--------------------------------------------------------------------------
% calculate hierarchical errorbars

qmutau = math.qmutau;
for k=1:K
    nu    = qmutau(k).nu;
    beta  = qmutau(k).beta;
    sigma = qmutau(k).sigma;
    s2    = 2*sigma/beta;
    nk    = 2*nu;
    qmutau(k).nk = nk;
    qmutau(k).he = sqrt(s2/nk);
end
math.qmutau = qmutau;

%--------------------------------------------------------------------------
% calculate exceedance probabilities
[math.exceedance] = cbm_hbi_exceedance(math.qm.alpha);

%--------------------------------------------------------------------------
% create an easy-to-understand output struct
theta = math.qhquad.theta;
Nbar  = math.Nbar';
r     = math.r';

K     = length(math.qmutau);
a     = cell(1,K);
he    = cell(1,K);
nk    = nan(1,K);
for k=1:K
    theta{k} = theta{k}';    
    a{k}     = math.qmutau(k).a';
    he{k}    = math.qmutau(k).he'; 
    nk(k)    = math.qmutau(k).nk';
end

xp  = [];
pxp = [];
if isfield(math,'exceedance')
    xp = math.exceedance.xp;
    if isfield(math.exceedance,'pxp')
        pxp = math.exceedance.pxp;
    else
        pxp = nan(size(xp));
    end
end

output     = struct('parameters',{theta},...
                    'responsibility',r,...
                    'group_mean',{a},...
                    'group_hierarchical_errorbar',{he},...
                    'model_frequency',Nbar/N,... % maximum of 1
                    'exceedance_prob',xp,...
                    'protected_exceedance_prob',pxp);

%--------------------------------------------------------------------------
% wrap up and save
hyper      = math.hyper;
profile    = struct('datetime',datestr(now),'filename',mfilename,...
                    'config',config,'optimconfigs',config.optconfigs,'hyper',hyper);    

cbm        = struct('method','hbi',...
                    'input',user_input,...
                    'profile',profile,...
                    'math',math,...
                    'output',output);
                
if ~isempty(fname), save(fname,'cbm'); end
if fid~=1, fclose(fid); end
end

%==========================================================================
function logging(verbose,fid,str)
% this function prints str on the screen AND on the fid (log file) using 
% fprintf if verbose is 1
if verbose, fprintf(str); end
if fid>1, fprintf(fid,str); end
end

%==========================================================================
function [Nbar, thetabar, Sdiag] = hbi_sumstats(r,qh)
% calculates summary statistics 

theta    = qh.theta;
Ainvdiag = qh.Ainvdiag;

[K]      = size(theta,1);
thetabar = cell(K,1);
Sdiag    = cell(K,1);
Nbar     = nan(K,1);

for k=1:K
    Nk          = sum(r(k,:));    
    Nbar(k)     = Nk;    
    thetabar{k} = sum(bsxfun(@times,theta{k},r(k,:)),2)/Nk;
    Sdiag{k}    = sum(bsxfun(@times,theta{k}.^2+Ainvdiag{k} ,r(k,:)),2)/Nk -thetabar{k}.^2;
end
end

%==========================================================================
function [qmutau,bound] = hbi_qmutau(pmutau,Nbar,thetabar,Sdiag)
% updates q(mu,tau)

K       = size(Nbar,1);
a       = cell(K,1);
beta    = cell(K,1);
sigma   = cell(K,1);
nu      = cell(K,1);
Etau    = cell(K,1);
Elogtau = cell(K,1);
logG    = cell(K,1);

ElogpH     = nan(K,1);
Elogpmu    = nan(K,1);
Elogqmu    = nan(K,1);
Elogptau   = nan(K,1);
Elogqtau   = nan(K,1);

for k=1:K
    a0k     = pmutau(k).a;
    beta0k  = pmutau(k).beta;
    nu0k    = pmutau(k).nu;
    sigma0k = pmutau(k).sigma;
    
    Nk            = Nbar(k);
    
    beta{k}       = beta0k  + Nk;
    a{k}          = (beta0k*a0k + Nk*thetabar{k})/beta{k};
    nu{k}         = nu0k + .5*Nk;
    sigma{k}      = sigma0k + .5*( Nk*Sdiag{k} + Nk*beta0k/(Nk+beta0k)*(thetabar{k}-a0k).^2 );
    
    
    Elogtau{k}    = psi(nu{k})-log(sigma{k});
    Etau{k}       = nu{k}./sigma{k};
    logG{k}       = sum(-gammaln(nu{k}) + nu{k}*log(sigma{k}));
    
    
    %--
    % the lower bound
    logG0         = sum(-gammaln(nu0k) + nu0k.*log(sigma0k));
    Dk            = length(a0k);
    
    ElogdetT      = sum(Elogtau{k});    
    ET            = diag(Etau{k});    
        
    Elogpmu(k)    = -Dk/2*log(2*pi) +.5*Dk*log(beta0k) + .5*ElogdetT -.5*(a{k}-a0k)'*(beta0k*ET)*(a{k}-a0k) -Dk/2*beta0k/beta{k};
    Elogptau(k)   = +(nu0k-1)*ElogdetT -sum(sigma0k.*diag(ET)) + logG0;
    
    Elogqmu(k)    = -Dk/2*log(2*pi) +.5*Dk*log(beta{k}) + .5*ElogdetT + -Dk/2;
    Elogqtau(k)   = +(nu{k}-1)*ElogdetT - Dk*nu{k} + logG{k};
    
    ElogpH(k)     = +.5*Nk*ElogdetT -.5*Nk*Dk*log(2*pi) -.5*Nk*Dk/beta{k} +...
                    -.5*sum(Etau{k}.*( Nk*Sdiag{k}+Nk*(thetabar{k}-a{k}).^2 ) );    
end
qmutau   = struct('name','GaussianGamma','a',a,'beta',beta,'sigma',sigma,'nu',nu,'Etau',Etau,'Elogtau',Elogtau,'logG',logG);
bound   = struct('module','qmutau','ElogpH',ElogpH,'Elogpmu',Elogpmu,'Elogptau',Elogptau,'Elogqmu',Elogqmu,'Elogqtau',Elogqtau);
end

%==========================================================================
function [qm,bound] = hbi_qm(pm,Nbar)
% updates q(m)

limInf     = pm.limInf;
logC0      = pm.logC;
alpha0     = pm.alpha;

alpha      = alpha0 + Nbar;
alpha_star = sum(alpha);
Elogm      = psi(alpha) - psi(alpha_star);
loggamma   = gammaln(alpha);
logC       = gammaln(alpha_star)-sum(loggamma);

Elogpm     = logC0 + sum((alpha0-1).*Elogm);
Elogqm     = logC  + sum((alpha-1).*Elogm);
ElogpZ     = Nbar.*Elogm;

if limInf
    K        = length(alpha);
    pm.alpha = inf(K,1);
    Elogm    = log(ones(K,1)/K);
    logC     = inf;
    Elogpm   = nan;
    Elogqm   = nan;
    ElogpZ   = Nbar.*Elogm;
end

qm         = struct('name','Dirichlet','limInf',limInf,'alpha',alpha,'Elogm',Elogm,'logC',logC);
bound      = struct('module','qm','ElogpZ',ElogpZ,'Elogpm',Elogpm,'Elogqm',Elogqm);

end

%==========================================================================
function [r,bound] = hbi_qHZ(qmutau,qm,qh,thetabar,Sdiag)
% updates q(H,Z)

qmlimInf  = qm.limInf;

logf      = qh.logf;
% theta     = cm.theta;
logdetA   = qh.logdetA;
% Ainvdiag  = cm.Ainvdiag;

[K,N]    = size(logf);
r        = nan(K,N);
% thetabar = cell(K,1);
% Sdiag    = cell(K,1);
% Nbar     = nan(K,1);
ElogpH   = nan(K,1);
ElogpZ   = nan(K,1);
ElogpX   = nan(K,1);
ElogqH   = nan(K,1);
ElogqZ   = nan(K,1);

D        = arrayfun(@(k)length(qmutau(k).a),(1:K)');
ElogdetT = arrayfun(@(k)sum(qmutau(k).Elogtau),(1:K)');
logdetET = arrayfun(@(k)sum(log(qmutau(k).Etau)),(1:K)');
beta     = arrayfun(@(k)qmutau(k).beta,(1:K)');

lambda   = .5*ElogdetT -.5*logdetET -.5*D./beta;
% this is equal
% lambda(k) = D(k)/2*(psi(qmutau(k).nu) -log(qmutau(k).nu) - qmutau(k).beta^-1 );

logrho   = logf -.5*logdetA;
logrho   = bsxfun(@plus,logrho,.5*D*log(2*pi) +lambda +qm.Elogm);

if qmlimInf, r = ones(K,N)/K; end

logeps      = exp(log1p(-1+eps));
for k=1:K
    if ~qmlimInf
    rarg        = bsxfun(@minus,logrho,logrho(k,:));
    r(k,:)      = 1./sum(exp(rarg),1);    
    end
    
    Nk          = sum(r(k,:));
    Dk          = D(k);
    
    ElogpH(k)   = +.5*Nk*ElogdetT(k) -.5*Nk*Dk*log(2*pi) -.5*Nk*Dk/qmutau(k).beta +...
                  -.5*Nk*sum(qmutau(k).Etau.*( Sdiag{k}+(thetabar{k}-qmutau(k).a).^2 ) );
    ElogpZ(k)   = Nk*qm.Elogm(k);
    
    ElogpXH     = sum( r(k,:).*(logf(k,:) -.5*Dk + lambda(k)) );
    ElogpX(k)   = ElogpXH - ElogpH(k);
    
    rlogr       = r(k,:).*(log1p(-1+r(k,:))); % =r(k,:).*log(r(k,:))
    rlogr(r(k,:)<logeps) = 0;
    ElogqH(k)   = sum( r(k,:).*(-Dk/2-Dk/2*log(2*pi)+.5*logdetA(k,:) ) );
    ElogqZ(k)   = sum( rlogr);  
end

bound = struct('module','qHZ','ElogpX',ElogpX,'ElogpH',ElogpH,'ElogpZ',ElogpZ,'ElogqH',ElogqH,'ElogqZ',ElogqZ);
end

%==========================================================================
function [qh_new] = hbi_qhquad(models,data,pconfig,qmutau,qh,fid,opt_paral)
% updates individual posterior q(H|Z)

[N] = length(data);
[K] = length(models);

mu  = cell(K,1);
precision = cell(K,1);
priors = struct('mean',mu,'precision',precision);
D = nan(K,1);

verbose = nan(1,K);
for k=1:K
    D(k)      = length(qmutau(k).a);
    priors(k) = struct('mean',qmutau(k).a,'precision',diag(qmutau(k).Etau));
    for n=1:N
        inits = qh.theta{k}(:,n)';
        allconfigs(k,n) = pconfig(k); %#ok<AGROW>
        allconfigs(k,n).inits = inits; %#ok<AGROW>
    end
    verbose(k) = pconfig(k).verbose;
end

% if verbose is 0 for optimization, do not write optimization messages 
% on the log-file too
if ~all(verbose>0), fid=0; end

% for parallel computing on clusters
if opt_paral
    try
    [logf,theta,A,~,flag] = cbm_loop_quad_lap(data,models,priors,allconfigs,fid,'LAP');
    catch
        save('log_cm.mat','logf','theta','A','flag');
        error('error: %s\n',msg.message);        
    end
end

if ~opt_paral
    theta = cell(K,1);
%     theta = repmat({nan(D(k),N)},K,1);
    logf  = nan(K,N);
%     A     = repmat({nan(D(k),D(k))},K,N);    
    A     = cell(K,N);
    flag  = nan(K,N);
    for k=1:K
        for n=1:N
            [logf(k,n),theta{k}(:,n),A{k,n},~,flag(k,n)] = cbm_quad_lap(data{n},models{k},priors(k),pconfig(k),fid,'LAP'); 
        end
    end
end

Ainvdiag = cell(K,1);
logdetA  = nan(K,N);
for n=1:N
    for k=1:K
        if flag(k,n)<1 % if no good gradient found (i.e. optimization failed), use the prior values for this subject-model
            theta_n      = priors(k).mean;
            [logf(k,n)]  = cbm_loggaussian(theta_n',models{k},priors(k),data{n});    
            A{k,n}       = priors(k).precision;
            theta{k}(:,n)= theta_n;
        end
        logdetA_kn       = 2*sum(log(diag(chol(A{k,n})))); % =log(det(A{k,n}))
        Ainvdiag{k}(:,n) = diag(A{k,n}^-1);
        logdetA(k,n)     = logdetA_kn;        
    end
end
qh_new   = struct('logf',logf,'theta',{theta},'Ainvdiag',{Ainvdiag},'logdetA',logdetA);
end

%==========================================================================
function [dprog,prog] = hbi_prog(prog,L,alpha,thetabar,Sdiag)
% calculates parameters for testing convergence

i         = length(prog);
Lpre      = prog(i).L;
alphapre  = prog(i).alpha;
xpre      = prog(i).x;

thetabar = cell2mat(thetabar);
Sdiag    = cell2mat(Sdiag);
x        = thetabar./sqrt(Sdiag);

dx            = sqrt(mean((x-xpre).^2));
dL            = L-Lpre;

[~,ibest]     = max(alpha);
dalpha        = abs(alpha(ibest)-alphapre(ibest));
dprog         = struct('dL',dL,'dalpha',dalpha,'dx',dx);
prog(i+1)     = struct('L',L,'alpha',alpha,'x',x);
end

%==========================================================================
function [bound,dL] = hbi_bound(bound,lastmodule)
% updates the variational lower bound after each step

bb            = bound.bound;
pmlimInf      = bb.pmlimInf;
Elogpm_Elogqm = bb.Elogpm - bb.Elogqm;
if pmlimInf
    Elogpm_Elogqm = 0;
end
Lpre     = + bb.ElogpX  + bb.ElogpH   + bb.ElogpZ + ...
           + bb.Elogpmu + bb.Elogptau + ...
           - bb.ElogqH  - bb.ElogqZ +...
           - bb.Elogqmu - bb.Elogqtau + ...
           + Elogpm_Elogqm;
       
switch lastmodule
    case 'qHZ'
        bb.ElogpX    = sum(bound.qHZ.ElogpX);
        bb.ElogpH    = sum(bound.qHZ.ElogpH);
        bb.ElogpZ    = sum(bound.qHZ.ElogpZ);
        bb.ElogqH    = sum(bound.qHZ.ElogqH);
        bb.ElogqZ    = sum(bound.qHZ.ElogqZ);        
        
    case 'qmutau'
        bb.ElogpH    = sum(bound.qmutau.ElogpH);
        bb.Elogpmu   = sum(bound.qmutau.Elogpmu);
        bb.Elogptau  = sum(bound.qmutau.Elogptau);
        bb.Elogqmu   = sum(bound.qmutau.Elogqmu);
        bb.Elogqtau  = sum(bound.qmutau.Elogqtau);
        
    case 'qm'
        bb.ElogpZ    = sum(bound.qm.ElogpZ);
        bb.Elogpm    = bound.qm.Elogpm;
        bb.Elogqm    = bound.qm.Elogqm;
end
Elogpm_Elogqm = bb.Elogpm - bb.Elogqm;
if pmlimInf
    Elogpm_Elogqm = 0;
end
L        = + bb.ElogpX  + bb.ElogpH   + bb.ElogpZ + ...
           + bb.Elogpmu + bb.Elogptau + ...
           - bb.ElogqH  - bb.ElogqZ +...
           - bb.Elogqmu - bb.Elogqtau + ...
           + Elogpm_Elogqm;              
dL       = + L - Lpre;


bb.lastmodule = lastmodule;
bb.L          = L;
bb.dL         = dL;
bound.bound   = bb;
bound.(lastmodule).bound = bb;

end
