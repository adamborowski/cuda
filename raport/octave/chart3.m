clc;
%----------------

x=[2 4 8 16];
y=[1 10 100 300 400];
v=[4511.264	2255.661	1128.521	581.630
451.538	225.853	113.276	58.409
137.834	69.333	45.664	41.388
108.184	54.565	39.200	40.897
106.540	53.719	40.592	41.507];





xx=x';
yy=y(2:end)';

v=v(2:end, :);



close all;

hold on;
hSurface = surf(xx, yy,v);
%set(hSurface,'EdgeColor',[0, 1, 0.2], 'FaceAlpha', 0.6);



title("czas w zależności od liczby bloków i rozmiaru danych");
xlabel('liczba wątków typu B')
ylabel('liczba bloków')
zlabel('czas [ms]')
grid on;
box off;

hold off;
view(140,15);

