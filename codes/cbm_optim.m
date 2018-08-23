function [tx,F,H,G,flag,k,P,NLL] = cbm_optim(h,optconfig,rng,numrep,init0,fid)
% This function minimizes h using the fminunc routine in matlab (with
% matlab versions after 2013a). 
% 
% copied from matlab documentation R2017b:
% 
% Quasi-Newton Algorithm
% The quasi-newton algorithm uses the BFGS Quasi-Newton method with a cubic line search procedure.
% This quasi-Newton method uses the BFGS ([1],[5],[8], and [9]) formula for updating the approximation of the Hessian matrix. 
% You can select the DFP ([4],[6], and [7]) formula, which approximates the inverse Hessian matrix, 
% by setting the HessUpdate option to 'dfp' (and the Algorithm option to 'quasi-newton'). 
% You can select a steepest descent method by setting HessUpdate to 'steepdesc' (and Algorithm to 'quasi-newton'), 
% although this setting is usually inefficient. See fminunc quasi-newton Algorithm.
% 
% Trust Region Algorithm
% The trust-region algorithm requires that you supply the gradient in fun and set SpecifyObjectiveGradient to true using optimoptions. 
% This algorithm is a subspace trust-region method and is based on the interior-reflective Newton method described in [2] and [3]. 
% Each iteration involves the approximate solution of a large linear system using the method of preconditioned conjugate gradients (PCG). 
% See fminunc trust-region Algorithm, Trust-Region Methods for Nonlinear Minimization and Preconditioned Conjugate Gradient Method.
% 
% References
% [1] Broyden, C. G. ?The Convergence of a Class of Double-Rank Minimization Algorithms.? 
% Journal Inst. Math. Applic., Vol. 6, 1970, pp. 76?90.
% [2] Coleman, T. F. and Y. Li. ?An Interior, Trust Region Approach for Nonlinear Minimization Subject to Bounds.? 
% SIAM Journal on Optimization, Vol. 6, 1996, pp. 418?445.
% [3] Coleman, T. F. and Y. Li. ?On the Convergence of Reflective Newton Methods for Large-Scale Nonlinear Minimization Subject to Bounds.? 
% Mathematical Programming, Vol. 67, Number 2, 1994, pp. 189?224.
% [4] Davidon, W. C. ?Variable Metric Method for Minimization.? A.E.C. Research and Development Report, ANL-5990, 1959.
% [5] Fletcher, R. ?A New Approach to Variable Metric Algorithms.? Computer Journal, Vol. 13, 1970, pp. 317?322.
% [6] Fletcher, R. ?Practical Methods of Optimization.? Vol. 1, Unconstrained Optimization, John Wiley and Sons, 1980.
% [7] Fletcher, R. and M. J. D. Powell. ?A Rapidly Convergent Descent Method for Minimization.? Computer Journal, Vol. 6, 1963, pp. 163?168.
% [8] Goldfarb, D. ?A Family of Variable Metric Updates Derived by Variational Means.? Mathematics of Computing, Vol. 24, 1970, pp. 23?26.
% [9] Shanno, D. F. ?Conditioning of Quasi-Newton Methods for Function Minimization.? Mathematics of Computing, Vol. 24, 1970, pp. 647?656.
% 
% implemented by Payam Piray, Aug 2018
%==========================================================================
if nargin<4, numrep=1; end
if nargin<5, init0=[]; end
if nargin<6, fid  =1;  end

%--------------------------------------------------------------------------
% % for R2017a
% objgradient = optconfig.gradient;
% objhessian  = optconfig.hessian;
% % if the algorithm is trust-region, then objgradient should be on
% if strcmpi(objgradient,'on'), objgradient = true; end
% if strcmpi(objgradient,'off'), objgradient = false; end
% if strcmpi(objhessian,'on'),  HessianFcn = 'objective'; end
% if strcmpi(objhessian,'off'), HessianFcn = []; end
% 
% options = optimoptions('fminunc','Algorithm',optconfig.algorithm,'Display','notify',...
%                         'SpecifyObjectiveGradient',objgradient,'HessianFcn',HessianFcn,...
%                         'ObjectiveLimit',optconfig.ObjectiveLimit);                    

%--------------------------------------------------------------------------                    
% for R2014b
options = optimoptions('fminunc','Algorithm',optconfig.algorithm,'Display','off',...
                        'GradObj',optconfig.gradient,'Hessian',optconfig.hessian,...
                        'ObjectiveLimit',optconfig.ObjectiveLimit);
                    
                    
% % P and NLL are vectors of prms and nll
% options = optimset('LargeScale',optconfig.largescale,...
%     'Display','off','TolFun',10^-10,'GradObj',...
%     optconfig.gradient,'Hessian',optconfig.hessian);

tolG       = optconfig.tolgrad;
numrep_up  = optconfig.numinit_up;
numrep_med = optconfig.numinit_med;
verbose    = optconfig.verbose;

F    = 10^16;
flag = 0;
tx   = nan;
H    = nan;
G    = nan;
r = rng(2,:)-rng(1,:); % for random initialization
numrep = numrep + size(init0,1);

k  = 0;
P  = [];
NLL= [];
while( (k<numrep) || (k>=numrep && k<numrep_med && flag==.5) || (k>=numrep && k<numrep_up && flag==0) )
    k=k+1;
    try
        init = init0(k,:);
    catch %#ok<CTCH>
        init = rand(size(r)).*r+rng(1,:);
    end
    
    try
        [tx_tmp, F_tmp, ~,~,G_tmp,H_tmp] = fminunc(h, init, options);
        [~,ishesspos] = chol(H_tmp);
        ishesspos = ~logical(ishesspos);
        
        sumG = mean(abs(G_tmp));
        
        if (flag~=1 || (F_tmp<F)) && ishesspos && (sumG<tolG)
                flag = 1;
                tx = tx_tmp;
                F = F_tmp;
                H = H_tmp;
                G = G_tmp;
        end
        if (flag~=1 && (F_tmp<F)) && ishesspos && (sumG>tolG) % minimal condition
            flag  = .5;
            tx = tx_tmp;
            F  = F_tmp;
            H  = H_tmp;
            G  = G_tmp;
        end        
        
        P   = [P; tx_tmp]; %#ok<AGROW>
        NLL = [NLL; F_tmp]; %#ok<AGROW>
    catch msg
        logging(verbose,fid,sprintf('--- This initialization was aborted (there might be a problem with the model)\n'));
        logging(verbose,fid,sprintf('--- The message of optimization routine is:\n'));
        logging(verbose,fid,sprintf('---    %s\n',msg.message));
    end
end

switch flag
    case 0
        logging(verbose,fid,sprintf('--- No positive hessian found in spite of %d initialization.\n',k));
    case .5
        logging(verbose,fid,sprintf('--- Positive hessian found, but not a good gradient in spite of %d initialization.\n',k));
    case 1
        if k>numrep
%             logging(verbose,fid,sprintf('--- Optimized with %d initializations(>%d specified by user).\n',k,numrep));
        end
end

end

function logging(verbose,fid,str)
% this function is similar to fprintf!
if verbose, fprintf(str); end
if fid>1, fprintf(fid,str); end
end
