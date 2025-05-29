function UDidx = get_triu_idx(N)
    idx = ones(N);
    UDidx = find(triu(idx,1));
end