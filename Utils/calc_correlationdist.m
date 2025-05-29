function [D,R] = calc_correlationdist(zmat,subsample)
% Jiaxin Cindy Tu 2022.11.22
% Calculates the correlation for NxN matrix but excludes the diagonal
% Use subsample to speed up the calculation (for dconns get ~0.99
% correlation for 1/100th subsample
% Unlike the Evan Gordon or Ruby Kong version, the subsampling is not
% random, but more evenly distributed and reproducible by sampling every
% kth vertex

    if ~exist('subsample','var')||isempty(subsample)
        subsample = 1;% don't subsample
    end

    % check that the matrix is the right format
    assert(size(zmat,1)==size(zmat,2));
    assert(length(size(zmat))==2);
    
    zmat(eye(size(zmat))==1) = NaN;% this calculation took too long (~25s)
    R = corr(zmat(1:subsample:end,:), 'rows', 'pairwise');
    
    D = 1-R;
end