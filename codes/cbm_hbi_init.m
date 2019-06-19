function [inits,priors]= cbm_hbi_init(flap,hyper,limInf,initialize_r,families)
% initializes HBI algorithm
% implemented by Payam Piray, Aug 2018
%==========================================================================

if nargin<3, limInf = 0; end
if nargin<4, initialize_r = 'all_r_1'; end
if nargin<5, families = []; end

%--------------------------------------------------------------------------

b = hyper.b;
v = hyper.v;
s = hyper.s;
K = length(flap);

% initialize using fcbm_maps
cbm_maps = cell(K,1);
allfiles_map = true;
for k=1:K
    fcbm_map = flap{k};
    if ischar(fcbm_map) % address of a cbm saved by cbm_lap?
        allfiles_map = allfiles_map && 1;
        fcbm_map  = load(fcbm_map); cbm_maps{k} = fcbm_map.cbm;
    elseif isstruct(fcbm_map) % or itself a cbm struct?
        allfiles_map = allfiles_map && 0;
        cbm_maps{k}  = fcbm_map;
    else % or something is wrong
        error('fcbm_map input has not properly been specified for model %d!',k);
    end
end
%--------------------------------------------------------------------------

bb        = struct('ElogpX',nan,...
                   'ElogpH',nan,'ElogpZ',nan,'Elogpmu',nan,'Elogptau',nan,'Elogpm',0,...
                   'ElogqH',nan,'ElogqZ',nan,'Elogqmu',nan,'Elogqtau',nan,'Elogqm',0,...
                   'pmlimInf',limInf,'lastmodule','','L',nan);


%--------------------------------------------------------------------------
% initializing of G and H parameters
K        = length(flap);
logrho   = cell(K,1);
theta    = cell(K,1);
Ainvdiag = cell(K,1);
logdetA  = cell(K,1);
logf     = cell(K,1);
D        = nan(K,1);
a0       = cell(K,1);
N        = length(cbm_maps{1}.output.parameters);
for k    = 1:K
    cbm_map     = cbm_maps{k};
    logrho{k}   = cbm_map.math.lme;
    logf{k}     = cbm_map.math.loglik;
    a0{k}       = cbm_map.input.prior.mean;
    theta{k}    = cell2mat(cbm_map.math.theta);
    Ainvdiag{k} = cell2mat(cbm_map.math.Ainvdiag);
    logdetA{k}  = cbm_map.math.logdetA;    
    D(k) = size(theta{k},1);    
end
logf    = cell2mat(logf);
logdetA = cell2mat(logdetA);
qh      = struct('logf',logf,'theta',{theta},'Ainvdiag',{Ainvdiag},'logdetA',logdetA);

%--------------------------------------------------------------------------
% p(mu,tau)
a       = cell(K,1);
beta    = cell(K,1);
sigma   = cell(K,1);
nu      = cell(K,1);
alpha0  = ones(K,1);
for k  = 1:K
    a{k}     = a0{k};
    beta{k}  = b;
    sigma{k} = s*ones(size(a{k}));%V0{k};
    nu{k}    = v;
end

if ~isempty(families)
    alpha0 = nan(K,1);
    uf = unique(families);
    for i=1:length(uf)
        f = uf(i);
        nf = sum(families==f);        
        alpha0(families==f) = 1/nf;
    end
end

pmutau   = struct('name','GaussianGamma','a',a,'beta',beta,'nu',nu,'sigma',sigma);
pm       = struct('name','Dirichlet','limInf',limInf,'alpha',alpha0);

% complete pmutau (ref: Bishop 2006)
for k=1:K
    a         = pmutau(k).a;
    beta      = pmutau(k).beta;    
    nu        = pmutau(k).nu;
    sigma     = pmutau(k).sigma;    
    
    
    % B.30    
    Elogtau   = psi(nu)-log(sigma);
    
    % B.27
    Etau      = nu./sigma; 
    
    % logG
    logG      = sum(-gammaln(nu) + nu.*log(sigma));
    
    % update
    pmutau(k).a         = a;
    pmutau(k).beta      = beta;    
    pmutau(k).Etau      = Etau;
    pmutau(k).Elogtau   = Elogtau;
    pmutau(k).logG      = logG;    
end
alpha      = pm.alpha;
alpha_star = sum(alpha);
Elogm      = psi(alpha) - psi(alpha_star);

loggamma1  = gammaln(alpha);
logC       = gammaln(alpha_star)-sum(loggamma1);

if pm.limInf
    pm.alpha = inf(size(alpha));
    Elogm = inf(size(alpha));
    logC  = 0;
end

pm.Elogm   = Elogm;
pm.logC    = logC;

%--------------------------------------------------------------------------
lme = cell2mat(logrho)';
switch initialize_r
    case 'all_r_1'
        % initialize with r=1 (i.e. assuming as if zkn = 1)
        r = ones(K,N);
    case 'cluster_r'
        r = init_cluster_r(lme,pm.alpha');
end

%--------------------------------------------------------------------------
bound     = struct('bound',bb,'qHZ',[],'qmutau',[],'qm',[]);
inits     = struct('qh',qh,'r',r,'bound',bound);
priors    = struct('hyper',hyper,'pmutau',pmutau,'pm',pm);

end


function r = init_cluster_r(lme,alpha0)
Ni      = size(lme,1);  % number of subjects
Nk      = size(lme,2);  % number of models
c       = 1;
tolc    = 10e-4;

% prior observations
alpha   = alpha0;

% iterative VB estimation
while c > tolc

    % compute posterior belief g(i,k)=q(m_i=k|y_i) that model k generated
    % the data for the i-th subject
    for i = 1:Ni
        for k = 1:Nk
            % integrate out prior probabilities of models (in log space)
            log_u(i,k) = lme(i,k) + psi(alpha(k))- psi(sum(alpha));
        end
        
        % exponentiate (to get back to non-log representation)
        u(i,:)  = exp(log_u(i,:)-max(log_u(i,:)));
        
        % normalisation: sum across all models for i-th subject
        u_i     = sum(u(i,:));
        g(i,:)  = u(i,:)/u_i;
    end
            
    % expected number of subjects whose data we believe to have been 
    % generated by model k
    for k = 1:Nk
        beta(k) = sum(g(:,k));
    end

    % update alpha
    prev  = alpha;
    for k = 1:Nk
        alpha(k) = alpha0(k) + beta(k);
    end
    
    % convergence?
    c = norm(alpha - prev);

end
r = g';
end