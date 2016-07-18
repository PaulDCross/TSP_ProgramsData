close all
% load(['TSP_Pictures\ArduinoWeightTest167.5\408.5mm\01\WeightLogFile.txt'])
% for i = 1:length(WeightLogFile)
%     WeightLogFile(i, 9) = WeightLogFile(i, 3) - WeightLogFile(i, 7);
% end
% a1 = 1;
% b1 = 101;
% b = 101;
% for i = 1:101
%     average(i,1) = mean(WeightLogFile(a1:b1, 9));
%     average(i,2) = WeightLogFile(b1, 8);
%     a1 = a1 + b;
%     b1 = b1 + b;
% end
% 
% figure1 = figure;
% % subplot(111)
% % title('First Hysteresis Test, 100 samples were taken at each z coordinate')
% subplot1 = subplot(2,1,1,'Parent',figure1,'XDir','reverse',...
%     'XMinorTick','on',...
%     'YMinorTick','on',...
%     'XTickLabel',{'398.0','399.0','400.0','401.0','402.0','403.0','404.0','405.0','406.0','407.0','408.0','409.0'},...
%     'XTick',[398.0 399.0 400.0 401.0 402.0 403.0 404.0 405.0 406.0 407.0 408.0 409.0]);
% hold on
% grid on
% title('Graph showing the Weight applied against the Z Coordinate of the head')
% xlabel('Z Coordinate, (milimetres)')
% ylabel('Weight, (grams)')
% xlim(subplot1,[398 409]);
% ylim(subplot1,[-100 1600]);
% IncreasingWeight = scatter(WeightLogFile(:, 4), WeightLogFile(:,3), 'k.');
% DecreasingWeight = scatter(WeightLogFile(:, 8), WeightLogFile(:,7), 'r.');
% legend([IncreasingWeight DecreasingWeight], 'Loading', 'Unloading', 'Location', 'SouthEast')
% 
% subplot(2,1,2,'Parent',figure1,'XDir','reverse',...
%     'XMinorTick','on',...
%     'YMinorTick','on',...
%     'XTickLabel',{'398.0','399.0','400.0','401.0','402.0','403.0','404.0','405.0','406.0','407.0','408.0','409.0','410.0'},...
%     'XTick',[398.0 399.0 400.0 401.0 402.0 403.0 404.0 405.0 406.0 407.0 408.0 409.0 410.0])
% hold on
% grid on
% title('Difference between the weight readings for pushing in and lifting off the pillow at the given z coordinate')
% xlabel('Z Coordinate, (milimetres)')
% ylabel('Difference in weight, (grams)')
% axis([398, 409, -25, 45])
% rawdata = scatter(WeightLogFile(:, 8), WeightLogFile(:,9), 10, 'yo', 'filled');
% averaged = scatter(average(:,2), average(:,1), 10, 'ko', 'filled');
% legend([rawdata averaged], 'The difference between the loading and unloading curves', 'The Average difference between the loading and unloading curves', 'Location', 'SouthEast')




load(['TSP_Pictures\ArduinoWeightTest167.5\408.5mm\02\WeightLogFile.txt'])
for i = 1:length(WeightLogFile)
    WeightLogFile(i, 9) = WeightLogFile(i, 3) - WeightLogFile(i, 7);
end
a1 = 1;
b1 = 101;
b = 101;
for i = 1:101
    average(i,1) = mean(WeightLogFile(a1:b1, 9));
    average(i,2) = WeightLogFile(b1, 8);
    a1 = a1 + b;
    b1 = b1 + b;
end

figure1 = figure;
set(figure1, 'Position', [0, 50, 1000, 620]);
subplot3 = subplot(1,1,1,'Parent',figure1,'XDir','reverse',...
    'XMinorTick','on',...
    'YMinorTick','on',...
    'XTickLabel',{'398.0','399.0','400.0','401.0','402.0','403.0','404.0','405.0','406.0','407.0','408.0','409.0'},...
    'XTick',[398.0 399.0 400.0 401.0 402.0 403.0 404.0 405.0 406.0 407.0 408.0 409.0]);
hold on
grid on
% title('Graph showing the Weight applied against the Z Coordinate of the head.')
xlabel('Z Coordinate, (milimetres)')
ylabel('Weight, (grams)')
xlim(subplot3,[398 409]);
ylim(subplot3,[-100 1600]);
IncreasingWeight = scatter(WeightLogFile(:, 4), WeightLogFile(:,3), 'k.');
DecreasingWeight = scatter(WeightLogFile(:, 8), WeightLogFile(:,7), 'r.');
legend([IncreasingWeight DecreasingWeight], 'Loading', 'Unloading', 'Location', 'SouthEast')


figure2 = figure;
set(figure2, 'Position', [1000, 50, 1000, 620]);
subplot(1,1,1,'Parent',figure2,'XDir','reverse',...
    'XMinorTick','on',...
    'YMinorTick','on',...
    'XTickLabel',{'398.0','399.0','400.0','401.0','402.0','403.0','404.0','405.0','406.0','407.0','408.0','409.0','410.0'},...
    'XTick',[398.0 399.0 400.0 401.0 402.0 403.0 404.0 405.0 406.0 407.0 408.0 409.0 410.0])
hold on
grid on
% title('Difference between the weight readings for pushing in and lifting off the pillow at the given z coordinate')
xlabel('Z Coordinate, (milimetres)')
ylabel('Difference in weight, (grams)')
axis([398, 409, -25, 45])
rawdata = scatter(WeightLogFile(:, 8), WeightLogFile(:,9), 10, 'yo', 'filled');
averaged = scatter(average(:,2), average(:,1), 10, 'ko', 'filled');
legend([rawdata averaged], 'The difference between the loading and unloading curves', 'The Average difference between the loading and unloading curves', 'Location', 'SouthEast')
