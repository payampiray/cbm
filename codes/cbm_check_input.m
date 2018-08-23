function ok = cbm_check_input(mu,hfunc,data,fname)
N = length(data);
fcheck = nan(N,1);
for s=1:N
    fcheck(s) = hfunc(mu,data{s});
end

if isempty(fname)
    ok_fname = 1;
else
    ok_fname = 'checking inputs...'; %#ok<NASGU>
    try
       save(fname,'ok_fname');
       ok_fname = 1;
    catch
        ok_fname = 0;
    end
    delete(fname);
end
ok = all(isfinite(fcheck) & isreal(fcheck)) && iscell(data) && numel(data)==N && ok_fname;
end
