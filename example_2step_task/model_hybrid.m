function [F] = model_hybrid(params,data)
% hybrid model of Daw et al, 2011

%-------------------------
fxu   = @(t)(1./(1+exp(-t)));
fxp   = @(t)exp(t);

alpha1  = fxu(params(1));
alpha2  = fxu(params(2));
lambda  = fxu(params(3));
tau     = fxp(params(4)); 
weight  = fxu(params(5));
phi     = params(6);
beta    = fxp(params(7));

%-------------------------
b     = [tau*weight tau*(1-weight) phi beta];

%%---
% unpack data
a1v = data.choice1;
a2v = data.choice2;
rv  = data.outcome;
s2v = data.state2;

missed = a1v==0 | a2v==0 | rv==0;
a1v(missed)=[];
a2v(missed)=[];
rv(missed)=[];
s2v(missed)=[];
rv (rv==2)=0;

T = length(rv);
n = zeros(3,2);
QTD = zeros(3,2);
QMB = zeros(1,2);
rep = zeros(1,2);

xQTD1 = nan(T,1);
xQTD2 = nan(T,1);
xQMB  = nan(T,1);
xrep  = nan(T,1);
for t=1:T
    s2 = s2v(t);
    a1 = a1v(t);
    a2 = a2v(t);
    
    nota1 = 3-a1;
    nota2 = 3-a2;
    xQTD1(t) = QTD(1,a1) - QTD(1,nota1);
    xQTD2(t) = QTD(s2,a2)- QTD(s2,nota2);
    xQMB (t) = QMB(1,a1) - QMB(1,nota1);
    xrep (t) = rep(1,a1) - rep(1,nota1);
    
    r  = rv(t);
    
    delta1 = 0 + QTD(s2,a2) - QTD(1,a1);
    QTD(1,a1) = QTD(1,a1) + alpha1*delta1;
    
    delta2 = r - QTD(s2,a2);
    QTD(s2,a2) = QTD(s2,a2) + alpha2*delta2;
    QTD(1,a1) = QTD(1,a1) + alpha1*lambda*delta2;
    
    % update transition probabilities
    n(s2,a1) = n(s2,a1)+1;    
    % transition to sB follwoing aA plus sC following aB
    n1 = n(2,1)+n(3,2);
    % or vice versa, to sC following aA plus sB following aB
    n2 = n(3,1)+n(2,2); 
    if n1>n2
        ptr(2,1) = 0.7;
        ptr(3,2) = 0.7;
        ptr(3,1) = 0.3;
        ptr(2,2) = 0.3;
    elseif n1<n2
        ptr(2,1) = 0.3;
        ptr(3,2) = 0.3;
        ptr(3,1) = 0.7;
        ptr(2,2) = 0.7;
    elseif n1==n2
        ptr(2,1) = 0.5;
        ptr(3,2) = 0.5;
        ptr(3,1) = 0.5;
        ptr(2,2) = 0.5;        
    end
    
    for aj=1:2
        QMB(aj) = ptr(2,aj)*max(QTD(2,:)) + ptr(3,aj)*max(QTD(3,:));
    end

    rep = rep*0;
    rep(a1) = 1;    
end

x      = [xQMB xQTD1 xrep];
X      = blkdiag(x,xQTD2);
%----

z = bsxfun(@times,X,b);
f = (1./(1+exp(-sum(z,2))));
F = sum(log(f+eps));
end
