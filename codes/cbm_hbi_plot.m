function cbm_hbi_plot(cbm, model_names, param_names, transform, k)
% cbm_hbi_plot plots HBI main outputs
%       cbm = cbm_hbi(cbm, k, model_names, param_names, tranform)
% 1st input: cbm struct or its file-address (output of cbm_hbi and cbm_hbi_null)
% 2nd input: a cell input containing the name of the models (each cell is
% a string) (optional). If not supplied, generic names will be used
% 3rd input: another cell input containing the name of the parameters 
% (each cell is a string) (optional). If not supplied, generic names (h_1,
% h_2 etc) will be used.
% 4th input: is a cell containig the transformation functions for
% parameters (each cell is a function handle). If not supplied, no
% transformation will be applied to the parameters
% 5th input: a scaler indicating the model of interest (usually the 
% winning model) (optional). If not supplied, parameters of the most
% frequent model will be plotted. If it is zero or nan, it does not plot
% parameters.
% 
% cbm_hbi_plot generates two figures:
% The first figure shows metrics of Bayesian model comparison, i.e.
% protected exceedance probabilities (PXP) and model frequencies.
% The second figure shows group-level parameters and group-level errorbars.
% if transform functions are supplied (the optinonal 5th input), this plot
% shows parameters in the trasnformed space. In this case, the error-bars 
% are obtained by applying the corresponding transformation function on the
% hierarchical errors estimated by the HBI and, therefore error-bars are
% not necessarily symmetric after transformation.
% 
% See also cbm_hbi and cbm_hbi_null
% Reference:
% Piray et al. (2018), Hierarchical Bayesian inference for concurrent model 
% fitting and comparison for group studies
% https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007043
% 
% implemented by Payam Piray, April 2019

%==========================================================================
if ischar(cbm)
    try
        fcbm = load(cbm);
    catch
        error('cannot load %s',cbm);
    end
    cbm = fcbm.cbm;    
end
freq = cbm.output.model_frequency;
K = length(freq);


if nargin<2
    model_names = cell(1,K);
    for i=1:K
        model_names{i} = sprintf('model%d',i);
    end
end
if length(model_names)~=K
    error('The dimension of the 3th input should be %d',K);
end

if nargin<5
    [~,k] = max(freq); 
    fprintf('plotting the group parameters of the most frequenct model');
end

x  = cbm.output.group_mean{k};
D = length(x);
if nargin<4
    param_names = cell(1,D);
    for i=1:D
        param_names{i} = sprintf('h_%d',i);
    end
end
if length(param_names)~=D
    error('The dimension of the 3rd input should be %d',D);
end

% add $ to parameters name (necessary for latex interpreter)
for i=1:D
    if ~strcmp(param_names{i}(1),'$'), param_names{i} = sprintf('$%s',param_names{i}); end
    if ~strcmp(param_names{i}(end),'$'), param_names{i} = sprintf('%s$',param_names{i}); end
end


do_transform = 1;
if nargin<4, do_transform = 0; end
if do_transform && length(transform)~=D
    error('The dimension of the 5th input should be %d',D);
end
for i=1:D
    if ischar(transform{i})
        transform{i} = str2func(transform{i});
        try 
            transform{i}(1);
        catch message
            error('problem in transform function: %s',message);
        end
    end
end

%-------
freq = cbm.output.model_frequency;
pxp  = cbm.output.protected_exceedance_prob;
xplabel ='PXP';
if all(isnan(pxp))
    fprintf('There is no protected exceedance probability as cbm_hbi_null has not been executed\n');
    fprintf('Plotting exceedance probability instead...\n');
    pxp = cbm.output.exceedance_prob;
    xplabel ='XP';
end

% for illustration purpose only
pxp(pxp<.005) = 0.005;

x    = cbm.output.group_mean{k};
xh   = cbm.output.group_mean{k} + cbm.output.group_hierarchical_errorbar{k};
xl   = cbm.output.group_mean{k} - cbm.output.group_hierarchical_errorbar{k};

tx = x;
txh = xh;
txl = xl;
if do_transform
    for i=1:D
        tx(:,i) =  transform{i}(x(:,i));
        txh(:,i) = transform{i}(xh(:,i));
        txl(:,i) = transform{i}(xl(:,i));
    end
end

%-------
% figure properties:
fs  = 10; % basic font
fst = 18; % font of title
fsy = 16; % font of y-label
fsp = 20; % font of parameters labels
fsx = 14; % font of xticklabel

fn  = 'calibri'; % font name

xst = .5; % position on x-axis for the title
yst = 1.12;  % position on y-axis for the title

alf = .6; % alpha value
colbar = [1 .2 .2]; % color

%--------------------------------------------------------------------------
fpos0 = [.1    0.5    .4*1.0000    .45*0.8133]; % figure position (normalized)
bw    = .25; % bar width
np = 2; % number of plots in this figure
dp = .1; % distance between plots
hp = .9; % height of the plot (normalized).


figure;
set(gcf,'units','normalized');
set(gcf,'position',fpos0);

%---------
hk = subplot(1,2,[1 2]);
pos = get(hk,'position');
pos(4) = hp*pos(4);
set(hk,'position',pos);
text(xst,yst,sprintf('Bayesian model comparison'),'fontsize',fst,'fontname',fn,...
    'Unit','normalized','fontweight','bold','Parent',hk,'HorizontalAlignment','center'); hold on;
set(gca,'visible','off');

xm = {pxp, freq};
xe = {pxp*0, freq*0};
ylabels = {xplabel,'Model frequency'};
%---------------
x0 = pos(1);    
lp = 1/np* (pos(3) -(np-1)*dp);
    
for i=1:np
    pos1 = pos;
    pos1(1) = x0+(i-1)*(lp+dp);
    pos1(3) = lp;

    h = axes('Position',pos1);    
    errorbarKxN(xm{i},xe{i},model_names,colbar,bw); hold on; 

    alpha(h,alf);
    set(gca,'fontsize',fs,'fontname',fn);
    
    hax = ancestor(h, 'axes');
    hxaxes = get(hax,'XAxis');
    set(hxaxes,'Fontsize',fsx);
    
    ylabel(ylabels{i},'fontsize',fsy);
    set(gca,'ylim',[0 1.05]);
end  


%--------------------------------------------------------------------------
% second figure: parameters

fpos0 = [0.1    0.1    1.0000    .45*0.8133];
lp = .075;
dp = .075;
hp = 0.9;
np = D;
wf = min(fpos0(3), lp*np+dp);
fpos0(3) = wf;
bw = .5*np/7;

figure;
set(gcf,'units','normalized');
set(gcf,'position',fpos0);

hk = subplot(1,2,[1 2]);
pos = get(hk,'position');
pos(4) = hp*pos(4);
set(hk,'position',pos);

text(xst,yst,sprintf('Parameters of %s', model_names{k}),'fontsize',fst,'fontname',fn,...
    'Unit','normalized','fontweight','bold','Parent',hk,'HorizontalAlignment','center'); hold on;
pos = get(gca,'Position');
set(gca,'visible','off');

x0 = pos(1);    
lp = 1/np* (pos(3) -(np-1)*dp);    
for i=1:D
    pos1 = pos;
    pos1(1) = x0+(i-1)*(lp+dp);
    pos1(3) = lp;

    h = axes('Position',pos1);    
    errorbarKxN(tx(i),[txl(i); txh(i)],param_names{i},colbar,bw); hold on;   
    alpha(gca,alf);
    set(h,'fontsize',fs,'fontname',fn);
    
    hax = ancestor(h, 'axes');
    hxaxes = get(hax,'XAxis');
    set(hxaxes,'TickLabelInterpreter','latex','Fontsize',fsp,'fontweight','bold');
    
    if i==1
        ylabel('Group parameters','fontsize',fsy);
    end
    
    ytick = get(h,'ytick');
    ytick(end) = [];
    set(h,'ytick', ytick);     
end  

end

function errorbarKxN(mx,ex,labels,colmap,barwidth)

if nargin<4, barwidth = []; end

[K,N] = size(mx);

if size(ex,1)==K
    el    = mx-ex;
    eh    = mx+ex;
elseif size(ex,1)==(2*K)
    el    = ex(1:K,:);
    eh    = ex(K+(1:K),:);
else
    error('!');
end

kk = 0:(1/K):1; kk(end)=[];
dx = 2;

if nargin<3
if K==1
    colmap = [.5 .5 .5];
else
    colmap = repmat( (0:(1/(K-1)):1)',1,3);
end
end

basevalue = 0;
if isempty(barwidth)
    wb = 1/(K+.5);
else
    wb = barwidth;
end
a     = nan(1,N);
% figure;
for i=1:N
    ax = -median(kk) + kk + +dx*(i-1);
    axs(i,:) = ax; %#ok<AGROW>
    a(i) = median(ax);
    for k=1:K
        bar(ax(k),mx(k,i),wb,'FaceColor',colmap(k,:),'EdgeColor','k','linewidth',1,'basevalue',basevalue);
        hold on;
    end
    for k=1:K        
        plot([ax(k);ax(k)],[el(k,i);eh(k,i)],'-','color','k','linewidth',2);
    end
end

set(gca,'xtick',a);
if ~isempty(labels)
    set(gca,'xticklabel',labels);
end
if K>1
dd = axs(1,2)-axs(1,1);
xlims = [axs(1,1)-dd axs(end,end)+dd];
set(gca,'xlim',xlims);
end

% set axes propertis
set(gca,'box','off');
set(gca,'ticklength', [0 0]);


end

function tx = sigmoid(x) %#ok<DEFNU>
tx = 1./(1+exp(-x));
end
function tx = none(x) %#ok<DEFNU>
tx = x;
end