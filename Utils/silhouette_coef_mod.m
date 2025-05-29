
function [ silhouettes,alternativeid ] = silhouette_coef_mod( parcels, D, neigh,type )
%SILHOUETTE_COEF Silhouette coefficient of a parcellation.
%   For each vertex in a cortical surface, SILHOUETTE_COEF compares the
%   within-parcel dissimilarity defined as the average distance to all
%   vertices in the same parcel, to the inter-parcel dissimilarity
%   obtained from those assigned to other parcels.
%
%   Let Pi be the parcel to which vertex i is assigned and ai and bi be the
%   average distance from i to the vertices in Pi and to the vertices in
%   other parcels adjacent to Pi. Silhouette coefficient (SI) for i is then
%   computed as:
%
%   SI(i) = (bi - ai)/max(ai,bi)
%
%   This guarantees SI values within [-1, +1], as long as a distance
%   measure is used.
%
%   INPUT
%   =====
%   parcels: A parcellation.
%   D: A distance matrix (such as Pearson's distance).
%   neigh: originally used in Arslan's code to find spatially neighboring parcels but
%   for the network case we use all parcels
%
%   OUTPUT
%   ======
%   silhouettes: Silhouette coefficients for all vertices
%
%   USAGE
%   =====
%   [ SILS ] = SILHOUETTE_COEF( PARCELS, D, NEIGH ) returns an N-by-1
%   vector, in which the ith element indicates the Silhouette value of the
%   ith vertex and N is the number of vertices. PARCELS can be a
%   parcellation of any resolution. D must be an N-by-N matrix, where each
%   vertex pair (x,y) equals to the distance between x and y. NEIGH is a
%   K-by-K adjaceny matrix, where K denotes the parcellation resolution.
%
%   REFERENCE
%   This code is part of the evaluation pipelines described in the brain
%   parcellation survey, "Human Brain Mapping: A Systematic Comparison of
%   Parcellation Methods for the Human Cerebral Cortex", NeuroImage, 2017
%   doi.org/10.1016/j.neuroimage.2017.04.014
%
%   For parcellations and more visit the Brain Parcellation Survey page at
%   https://biomedia.doc.ic.ac.uk/brain-parcellation-survey/
%
%   Author: Salim Arslan, April 2017 (twitter @salimarslan)

if ~exist('type','var')
    type='next'; % default to next best which is how % Rousseeuw et al. 1987 originally defines it
end

n = length(D);
num = max(parcels);
silhouettes = NaN(n,1);
alternativeid = zeros(n,1);
for i = 1 : num
    in_members = parcels == i;
    nk = sum(in_members);
    
    if nk < 2
        continue; % Singleton parcel detected. Can happen with N-Cuts.
    end
    
    dists = D(in_members,in_members);
    dists(logical(eye(length(dists)))) = 0;
    dists_in = sum(dists,2)/(length(dists)-1);
    
    ids = find(neigh(:,i));
    [dists_out,min_id] = find_dist_to_neighs( ids, parcels, D, in_members ,type);

    silhouettes(in_members) = (dists_out - dists_in) ./ ...
        max([dists_in dists_out],[],2);
    
    alternativeid(in_members) = min_id;
end

assert(sum(isnan(silhouettes))==0);

end


% Compute distance to/from vertices in adjacent parcels
function [ min_votes_out,min_id] = find_dist_to_neighs(ids, parcels, D, in_members,type)

% Jiaxin Cindy Tu 20221121 the Arslan method use mean across any neighbors
% which is kind of weird! should be the minimum of average across other
% parcels. -> Update 2024: I think they used the minimum of average across all spatially
% adjacent parcels not all parcels. -> which is not as bad, kind of like
% the DCBC in Zhi et al. 2022 Human Brain Mapping

switch type 
    case 'all' %compare to all the rest
        out_members = false(size(parcels));
        for j = 1 : length(ids)
            out_members = out_members | parcels == ids(j);
        end
        nk = sum(out_members);
        corrs = D(in_members,out_members);
        min_votes_out = sum(corrs,2)/nk;
        min_id = NaN;
    case 'next' % compare to next best
        votes_out = NaN(sum(in_members),length(ids));
        for j = 1 : length(ids)
            out_members = parcels == ids(j);
            nk = sum(out_members);
            corrs = D(in_members,out_members);
            votes_out(:,j) = sum(corrs,2)/nk;
        end
        [min_votes_out,min_id] = min(votes_out,[],2);
        
        ambiguous_ids = sum(min_votes_out==votes_out,2)~=1;
        min_id(ambiguous_ids) = NaN;% update 2023.05.23 set the ambiguous ids to NaN.
        
        if ~isnan(min_id)
            min_id = ids(min_id);
        end
end

end





