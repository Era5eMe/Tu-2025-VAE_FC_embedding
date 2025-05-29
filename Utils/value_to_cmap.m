function xcolors = value_to_cmap(x,cmin,cmax,cmap)
    N = length(cmap);
    maptounity = @(v)(v-cmin)/(cmax-cmin);
    x = round(maptounity(x)*N)+1;
    x(x>N) = N;
    x(x<1)=1;
    xcolors = cmap(x,:);
end