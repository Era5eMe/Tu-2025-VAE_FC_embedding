function zmat = cal_zmat(ptseries)
%CAL_ZMAT Calculate Fisher-z transformed fc matrix from parcellized
%time series data
%   Detailed explanation goes here
    fc_mat = corr(ptseries);
    zmat = 0.5 * log((1 + fc_mat) ./ (1 - fc_mat));
    zmat(1: size(fc_mat, 1) + 1 : end) = 0;
end

