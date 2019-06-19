function pconfig = cbm_hbi_config(pconfig)
% 
% pconfig.verbose: if 1, report will be written on the standard output
% (e.g. the screen)
% fname_prog is the file-address for saving outputs on every iteration
% save_prog is a loggical: if 1, outputs will be saved on every iteration
% save_prog is a loggical: if 1, outputs will be saved on every iteration
% initialize indicates how HBI is initialized
% maxiter is the number of maximum iterations in HBI
% tolx indicates the tolerance for dx on every iteration
% tolL indicates the tolerance for dL on every iteration
% parallel is a logical. If it is 1, HBI implements parallel computing 
% (not implemented in cbm)
% 
% implemented by Payam Piray, Aug 2018
%==========================================================================

if nargin<1, pconfig = []; end
verbose_default = 1; usejava('desktop');
if isempty(pconfig), pconfig = struct('verbose',verbose_default); end
p0 = pconfig;

save_prog = 0;
if isfield(p0,'fname_prog')
    if ~isempty(p0.fname_prog)
        save_prog = 1;
    end
end

p = inputParser;
p.addParameter('verbose',verbose_default);
p.addParameter('fname_prog',sprintf('cbm_%s_%0.4f.mat','hbi',now),@valid_fname);
p.addParameter('flog',[],@valid_flog); % is the passed value is -1, there will be no log-file at all
p.addParameter('save_prog',save_prog,@(arg)(arg==1)||(arg==0));
p.addParameter('initialize','all_r_1',@(arg)any(strcmp(arg,{'all_r_1','cluster_r'})));

p.addParameter('maxiter',50);
p.addParameter('tolx',0.01,@(arg)isscalar(arg)); %increase to have faster fitting
p.addParameter('tolL',-log(.5),@(arg)isscalar(arg));
p.addParameter('parallel',false);

% if parallel processing
if isfield(pconfig,'parallel')
    if pconfig.parallel
        p.addParameter('parallel',false);
        p.addParameter('loop_runtime',30,@(arg)(isscalar(arg)&& (arg<300)));
        p.addParameter('loop_maxruntime',90,@(arg)(isscalar(arg)&& (arg<600)));
        p.addParameter('loop_pausesec',30,@(arg)(isscalar(arg)&& (arg<60)));
        p.addParameter('loop_maxnumrun',3,@(arg)(floor(arg)==arg));    
        p.addParameter('loop_discard_bad',true,@(arg)(islogical(arg)));
    end
end

p.parse(pconfig);
pconfig    = p.Results;

if pconfig.save_prog==0, pconfig.fname_prog = []; end
end

function valid = valid_fname(arg)
valid =1;
if isempty(arg)
    return;
end
try
    [fdir,~,fext]=fileparts(arg);
    valid = exist(fdir,'dir') && strcmp(fext,'.mat');
catch msg
    warning(msg.message);
    valid = 0;
end

end

function valid = valid_flog(arg)
if arg==-1
    valid = 1;
    return;
end

valid = 0;
if ischar(arg)
    try
        [fdir]=fileparts(arg);
        valid = exist(fdir,'dir');
    catch msg
        warning(msg.message);
        valid = 0;        
    end
elseif arg==1
    valid = 1;
elseif isempty(arg)
    valid = 1;
end
end