function [exceedance,xp,pxp,bor] = cbm_hbi_exceedance(math,math0)
% This function is called by cbm_hbi and cbm_hbi_null
% 1st input: math computed by cbm_hbi
% 2nd input: math computed by cbm_hbi_null
% 1st output: a struct contaning all exceedance info
% 2nd output: exceedance probability
% 3rd output: protected exceedance probability
% 
% implemented by Payam Piray, Aug 2018
%==========================================================================

if nargin<2, math0 = []; end

alpha = math.qm.alpha;
L     = math.bound.bound.L;
L0    = nan;
if isempty(math0)
    if nargout>2
        error('For computing pxp and bor, cbm0 is required!');
    end
else
    L0    = math0.bound.bound.L;
end
Nsamp = 1e6;
[xp,pxp,bor] = compute_exceedance(alpha,L,L0,Nsamp);

exceedance = struct('xp',xp,'pxp',pxp,'bor',bor,'alpha',alpha,'L',L,'L0',L0,'math0',math0);
end

function [xp,pxp,bor] = compute_exceedance(alpha,L,L0,Nsamp)
K     = length(alpha);
if K == 2
    % comparison of 2 models
    xp(1) = spm_Bcdf(0.5,alpha(2),alpha(1));
    xp(2) = spm_Bcdf(0.5,alpha(1),alpha(2));
else
    % comparison of >2 models: use sampling approach
    xp = spm_dirichlet_exceedance(alpha,Nsamp);
end

bor = 1/(1+exp((L-L0)));
% Compute protected exceedance probs - Eq 7 in Rigoux et al.
pxp=(1-bor)*xp+bor/K;

end

function xp = spm_dirichlet_exceedance(alpha,Nsamp)
% Compute exceedance probabilities for a Dirichlet distribution
% FORMAT xp = spm_dirichlet_exceedance(alpha,Nsamp)
% 
% Input:
% alpha     - Dirichlet parameters
% Nsamp     - number of samples used to compute xp [default = 1e6]
% 
% Output:
% xp        - exceedance probability
%__________________________________________________________________________
%
% This function computes exceedance probabilities, i.e. for any given model
% k1, the probability that it is more likely than any other model k2.  
% More formally, for k1=1..Nk and for all k2~=k1, it returns p(x_k1>x_k2) 
% given that p(x)=dirichlet(alpha).
% 
% Refs:
% Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ
% Bayesian Model Selection for Group Studies. NeuroImage (in press)
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% Will Penny & Klaas Enno Stephan
% $Id: spm_dirichlet_exceedance.m 3118 2009-05-12 17:37:32Z guillaume $

if nargin < 2
    Nsamp = 1e6;
end

Nk = length(alpha);

% Perform sampling in blocks
%--------------------------------------------------------------------------
blk = ceil(Nsamp*Nk*8 / 2^28);
blk = floor(Nsamp/blk * ones(1,blk));
blk(end) = Nsamp - sum(blk(1:end-1));

xp = zeros(1,Nk);
for i=1:length(blk)
    
    % Sample from univariate gamma densities then normalise
    % (see Dirichlet entry in Wikipedia or Ferguson (1973) Ann. Stat. 1,
    % 209-230)
    %----------------------------------------------------------------------
    r = zeros(blk(i),Nk);
    for k = 1:Nk
%         r(:,k) = spm_gamrnd(alpha(k),1,blk(i),1);
        r(:,k) = gamrnd(alpha(k),1,blk(i),1);
%             r = gamrnd(repmat(vec(suffStat.d),1,N),1,K,N);
%             y = r ./ repmat(sum(r,1),K,1);        
    end
    sr = sum(r,2);
    for k = 1:Nk
        r(:,k) = r(:,k)./sr;
    end
    
    % Exceedance probabilities:
    % For any given model k1, compute the probability that it is more
    % likely than any other model k2~=k1
    %----------------------------------------------------------------------
    [~, j] = max(r,[],2);
    xp = xp + histc(j, 1:Nk)';
    
end
xp = xp / Nsamp;

end

function F = spm_Bcdf(x,v,w)
% Inverse Cumulative Distribution Function (CDF) of Beta distribution
% FORMAT F = spm_Bcdf(x,v,w)
%
% x   - Beta variates (Beta has range [0,1])
% v   - Shape parameter (v>0)
% w   - Shape parameter (w>0)
% F   - CDF of Beta distribution with shape parameters [v,w] at points x
%__________________________________________________________________________
%
% spm_Bcdf implements the Cumulative Distribution Function for Beta
% distributions.
%
% Definition:
%--------------------------------------------------------------------------
% The Beta distribution has two shape parameters, v and w, and is
% defined for v>0 & w>0 and for x in [0,1] (See Evans et al., Ch5).
% The Cumulative Distribution Function (CDF) F(x) is the probability
% that a realisation of a Beta random variable X has value less than
% x. F(x)=Pr{X<x}: This function is usually known as the incomplete Beta
% function. See Abramowitz & Stegun, 26.5; Press et al., Sec6.4 for
% definitions of the incomplete beta function.
%
% Variate relationships:
%--------------------------------------------------------------------------
% Many: See Evans et al., Ch5
%
% Algorithm:
%--------------------------------------------------------------------------
% Using MATLAB's implementation of the incomplete beta finction (betainc).
%
% References:
%--------------------------------------------------------------------------
% Evans M, Hastings N, Peacock B (1993)
%       "Statistical Distributions"
%        2nd Ed. Wiley, New York
%
% Abramowitz M, Stegun IA, (1964)
%       "Handbook of Mathematical Functions"
%        US Government Printing Office
%
% Press WH, Teukolsky SA, Vetterling AT, Flannery BP (1992)
%       "Numerical Recipes in C"
%        Cambridge
%__________________________________________________________________________
% Copyright (C) 1999-2011 Wellcome Trust Centre for Neuroimaging

% Andrew Holmes
% $Id: spm_Bcdf.m 4182 2011-02-01 12:29:09Z guillaume $


%-Format arguments, note & check sizes
%--------------------------------------------------------------------------
if nargin<3, error('Insufficient arguments'), end

ad = [ndims(x);ndims(v);ndims(w)];
rd = max(ad);
as = [[size(x),ones(1,rd-ad(1))];...
      [size(v),ones(1,rd-ad(2))];...
      [size(w),ones(1,rd-ad(3))]];
rs = max(as);
xa = prod(as,2)>1;
if sum(xa)>1 && any(any(diff(as(xa,:)),1))
    error('non-scalar args must match in size');
end

%-Computation
%--------------------------------------------------------------------------
%-Initialise result to zeros
F = zeros(rs);

%-Only defined for x in [0,1] & strictly positive v & w.
% Return NaN if undefined.
md = ( x>=0  &  x<=1  &  v>0  &  w>0 );
if any(~md(:))
    F(~md) = NaN;
    warning('Returning NaN for out of range arguments');
end

%-Special cases: F=1 when x=1
F(md & x==1) = 1;

%-Non-zero where defined & x>0, avoid special cases
Q  = find( md  &  x>0  &  x<1 );
if isempty(Q), return, end
if xa(1), Qx=Q; else Qx=1; end
if xa(2), Qv=Q; else Qv=1; end
if xa(3), Qw=Q; else Qw=1; end

%-Compute
F(Q) = betainc(x(Qx),v(Qv),w(Qw));
end