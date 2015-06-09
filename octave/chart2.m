clc;
%----------------

x=[1 10 100	300 400];
y=[2000 20000 200000 2000000];
v=[1.113	0.614	0.613	1.310	1.662
0.986	0.546	1.532	4.023	5.270
6.133	1.112	0.845	1.657	2.010
5.375	1.638	2.649	5.143	6.389
58.379	6.127	4.814	5.273	5.678
51.219	15.284	15.179	17.146	18.394
581.630	58.409	41.388	40.897	41.507
510.511	153.188	129.860	133.345	135.090];



not_optim = v(1:2:end,:);
optim = v(2:2:end, :);
xx=x';
yy=y';





close all;

hold on;
hSurface = mesh(xx, yy,not_optim);
set(hSurface,'EdgeColor',[0.4, 1, 0.2], 'FaceAlpha', 0);
hSurface=mesh(xx, yy,optim);
set(hSurface,'EdgeColor',[1, 0.4, 0.2], 'FaceAlpha', 0);

legend("dev 0","dev 1");
title("czas w zależności od liczby bloków i rozmiaru danych");
xlabel('liczba bloków')
ylabel('rozmiar danych')
zlabel('czas [ms]')
grid on;
box off;

hold off;
view(40,15);

