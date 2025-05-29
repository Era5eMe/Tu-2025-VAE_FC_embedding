function [hCB,hf] = makecolorbar(cmap,climits,direction,labelstr)
if ~exist('direction','var')
    direction = 'h';
end
if ~exist('labelstr','var')
    labelstr = '';
end
if strcmp(direction,'v')
    hf = figure('position',[100 100 200 300],'Units','normalized');
    hCB = colorbar('west');
    hCB.Position = [0.3 0.15 0.1 0.8];
elseif strcmp(direction,'h')
    hf = figure('position',[100 100 300 200],'Units','normalized');
    hCB = colorbar('south');
    hCB.Position = [0.1 0.15 0.8 0.05];
end

colormap(cmap)
caxis(climits);
set(gca,'Visible',false,'FontSize',15,'FontWeight','Bold')
hCB.Title.String = labelstr;
% hCB.Label.String = labelstr;
hCB
% hf.Position(3) = 0.1;
end