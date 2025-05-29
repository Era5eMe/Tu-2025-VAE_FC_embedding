% This helps to visualize homogeneity/variance at individual parcel
function plot_parcels_by_values(x,parcelview,Parcels,clim,cmap)
    x = reshape(x,[],1);
%% Prepare
    load('MNI_coord_meshes_32k.mat')
    Anat.CtxL = MNIl;Anat.CtxR = MNIr;
    clear MNIl MNIr
    %% Find parcels to remove
    idx = find(isnan(x));
    if ~isempty(idx)
        Parcels.CtxL(any(Parcels.CtxL==idx',2))=0;
        Parcels.CtxR(any(Parcels.CtxR==idx',2))=0;
    end
    %% View all parcels on MNI
    % (1) Add parcel nodes to Cortices
    Anat.CtxL.data=Parcels.CtxL;
    Anat.CtxR.data=Parcels.CtxR;
    % (2) Set parameters to view as desired
    params.Cmap.P = NaN(length(x),3); % can be NaN too just leave these indices as gaps
    params.Cmap.P(~isnan(x),:)=value_to_cmap(x(~isnan(x)),clim(1),clim(2),cmap);
    params.TC=1;
    params.ctx='inf';           % 'std','inf','vinf'
    params.view= parcelview;       % 'dorsal','post','lat','med'
    params.fig_handle = gca;
    PlotLRMeshes(Anat.CtxL,Anat.CtxR, params);
    set(gcf,'color','w');
end