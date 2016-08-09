close all
load('ZippedMADXYDepthName.mat')

fig1 = figure;
% Create axes
axes1 = axes('Parent',fig1,'YMinorTick','on','XMinorTick','on',...
    'XTickLabel',{'2','4','6','8','10','8','6','4','2'},...
    'XTick',[2 4 6 8 10 12 14 16 18]);
hold(axes1,'on');
% Create xlabel
xlabel('Depth from start position (mm)');
% Create ylabel
ylabel('Average Mean Absolute Deviation (Pixels)');
scatter(MaddeningX(:,2)/10,MaddeningX(:,1), 'xb')

fig2 = figure;
% Create axes
axes1 = axes('Parent',fig2,'YMinorTick','on','XMinorTick','on',...
    'XTickLabel',{'2','4','6','8','10','8','6','4','2'},...
    'XTick',[2 4 6 8 10 12 14 16 18]);
hold(axes1,'on');
% Create xlabel
xlabel('Depth from start position (mm)');
% Create ylabel
ylabel('Average Mean Absolute Deviation (Pixels)');
scatter(MaddeningY(:,2)/10,MaddeningY(:,1), 'xb')
