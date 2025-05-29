function [outval] = with_without_mw_conversion(outstr)
% get the indices for with or without medial wall
    assert(isstring(outstr)||ischar(outstr));
    verts_per_hem = 32492;
    load('medialwall_verts.mat','mw_verts')
    trunc2full = setdiff(1:(verts_per_hem * 2), mw_verts)';
    full2trunc = zeros(verts_per_hem * 2,1);
    full2trunc(trunc2full) = 1:length(trunc2full);
    Lindfull= setdiff(1:verts_per_hem,mw_verts)';
    Lindtrunc = [1:length(Lindfull)]';
    Rindfull = setdiff((verts_per_hem+1):(verts_per_hem * 2),mw_verts)';
    Rindtrunc = [1:length(Rindfull)]'+length(Lindfull);
    try
        outval = eval([outstr ';']);
    catch
        error(['unknown outstr: ',outstr])
    end
end