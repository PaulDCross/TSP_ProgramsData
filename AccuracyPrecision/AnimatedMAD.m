close all

K=101;
a=[];
b=[];
c=[];
d=[];
x=[];
y=[];
for i = 1 : 1 : length([stdX{:,1}])
%     hold on
%     grid on
%     scatter3(abs([CompleteData(1:b,i).Depth]), ones(b,1) * i, [stdY{i,1:b}]);
%     axis([0 11 0 120 0 0.5]);
    a(i, :) = ones(1, K) * i;
    b(i, :) = abs([CompleteData(1:K,i).Depth]);
    x(i, :) = [CompleteData(1:K,i).OriginalXcoord];
    y(i, :) = [CompleteData(1:K,i).OriginalYcoord];
%     scatter3(abs([CompleteData(1:K,i).Depth]), ones(1, K) * i, [stdX{i, :}])
end
figure1 = figure;
subplot1 = subplot(1,1,1,'Parent',figure1,'ZMinorTick','on','YMinorTick','on','XMinorTick','on');

zlim(subplot1,[0 0.6]);
view(subplot1,[90 0]);
grid(subplot1,'on');
hold(subplot1,'on');
for i = 1 : 1 : length([stdX{1,:}])
    pause(.01)
    scatter(0,0)
    hold on
    title({'Graph showing Mean Absolute Deviation of each pin as the head is being pushed into the Pillow.'; 'Standard Diviation of the Y coordinate'; num2str(b(1, i))});
    xlabel('X coordinate (px)')
    ylabel('Y coordinate (px)')
    zlabel('Standard Diviation');
    view(subplot1,[45 40]);
    zlim(subplot1,[0 0.6]);
    tri = delaunay(x(:,i),y(:,i));
    trisurf(tri,x(:,i),y(:,i),madX(:,i),'Parent',subplot1)
%     scatter3(x(:,i),y(:,i),madX(:,i),'Parent',subplot1,'filled')
    hold off
end