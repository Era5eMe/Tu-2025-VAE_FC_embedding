clear; close all; clc;
addpath(genpath('./Utils/'));
%% Load .mat files
load('./data/hcp/HCP_allsubjs.mat');
load('./data/hcp/HCP_94subjs.mat');
load('./mask/IM_Gordon_13nets_333Parcels.mat');
load('./mask/Parcels_Gordon.mat', 'Parcels');
[n_subjects, n_parcels, n_features] = size(hcpAll_latents);

% Initialize dispersion vector
dispersion965_tmp = zeros(n_parcels, 1);
dispersion94_tmp = zeros(n_parcels, 1);

% Loop over parcels
for p = 1:n_parcels
    % Extract the (n_subjects * n_features) matrix for parcel p
    features965 = squeeze(hcpAll_latents(:, p, :));  % shape: (n_subjects, n_features)
    features94 = squeeze(hcp94_latents(:, p, :));  % shape: (n_subjects, n_features)
    % Compute the centroid (mean across subjects)
    centroid965 = mean(features965, 1);       % shape: (1, n_features)
    centroid94 = mean(features94, 1);
    % Compute squared Euclidean distances to centroid
    squared_dists965 = sum((features965 - centroid965).^2, 2);  % shape: (n_subjects, 1)
    squared_dists94 = sum((features94 - centroid94).^2, 2);
    % Sum across subjects to get dispersion
    dispersion965_tmp(p) = sum(squared_dists965);
    dispersion94_tmp(p) = sum(squared_dists94);
end


cmap = interp1(linspace(0,100,10),redbluecmap(10),linspace(0,100,100));

figure;
t = tiledlayout(2,1,'TileSpacing','tight');
% title(t, 'Dispersion')
nexttile;
plot_parcels_by_values(dispersion965_tmp, 'med', Parcels, [prctile(dispersion965_tmp, 5), prctile(dispersion965_tmp, 95)], cmap) 
nexttile;
plot_parcels_by_values(dispersion965_tmp,'lat',Parcels,[prctile(dispersion965_tmp, 5), prctile(dispersion965_tmp, 95)], cmap) 
%{
cb = colorbar;
colormap(cmap);
caxis([prctile(dispersion965, 5), prctile(dispersion965, 95)]);
cb.Position = [0.85, 0.2, 0.03, 0.6];
cb.Ticks = [100, 700];
cb.FontSize = 20;
%}

figure;
t = tiledlayout(2,1,'TileSpacing','tight');
% title(t, 'Dispersion')
nexttile;
plot_parcels_by_values(dispersion94_tmp, 'med', Parcels, [prctile(dispersion94_tmp, 5), prctile(dispersion94_tmp, 95)], cmap) 
nexttile;
plot_parcels_by_values(dispersion94_tmp,'lat',Parcels,[prctile(dispersion94_tmp, 5), prctile(dispersion94_tmp, 95)], cmap) 
%{
cb = colorbar;
colormap(cmap);
caxis([prctile(dispersion94, 5), prctile(dispersion94, 95)]);
cb.Position = [0.85, 0.2, 0.03, 0.6];
cb.Ticks = [10, 70];
cb.FontSize = 20;
%}