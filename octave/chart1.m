clc;
%----------------

x=[1 10 100	300 400];
y=[2000 20000 200000 2000000];
v=[1.225536	0.6965312	0.7614912	1.310656	1.6621696
1.113248	0.6139584	0.613312	1.3096576	1.6621632
6.7933056	1.302368	0.987456	1.6616768	2.0266688
6.132832	1.1120512	0.8446144	1.6570304	2.0100672
64.7160124	7.178976	6.0410432	5.396192	5.7982848
58.3787138	6.127456	4.8143424	5.2731392	5.6779008
645.6842406	68.9663726	53.9147454	42.7544318	42.3759304
581.6300538	58.4085494	41.3876808	40.8971068	41.5074624];


x=x(2:1:end)

v=v(:,2:1:end);

not_optim = v(1:2:end,:);
optim = v(2:2:end, :);
xx=x';
yy=y';





close all;

hold on;
hSurface = mesh(xx, yy,not_optim);
set(hSurface,'EdgeColor',[1, 0.4, 0.2], 'FaceAlpha', 0);
hSurface=mesh(xx, yy,optim);
set(hSurface,'EdgeColor',[0.4, 1, 0.2], 'FaceAlpha', 0);

legend("brak optymalizacji","optymalizacja");
title("czas w zależności od liczby bloków i rozmiaru danych");
xlabel('liczba bloków')
ylabel('rozmiar danych')
zlabel('czas [ms]')
grid on;
box off;

hold off;
view(40,15);

