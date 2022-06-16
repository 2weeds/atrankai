%#ok<*NOPTS>

% praejusios programos isvalymas
clc
clear
close all

% duomenu uzkrovimas
load 'sunspot.txt'

% duomenu grafikas
figure(1)
plot(sunspot(:,1),sunspot(:,2),'b-*')
title('Sun Spot amounts throughout the years')
xlabel('years') 
ylabel('sunspot activity') 

% matricu sukurimas
L = length(sunspot);            % data size
P = [   sunspot(1:L-2,2)' ;     % input data
        sunspot(2:L-1,2)'];    
T = sunspot(3:L,2)';            % output data

% trimate diagrama
figure(2)
plot3(P(1,:),P(2,:),T,'bo')
title('Sun Spots')
xlabel('P1') 
ylabel('P2') 
zlabel('T')

% isskirti apmokymo duomenu matricas
Pu = P(:,1:200);
Tu = T(:,1:200);

disp('Pu array size:')
disp(size(Pu))
disp('Tu array size:')
disp(size(Tu))

% sukurti neuronu tinkla
net = newlind(Pu,Tu);

% pavaizduojami neurono koefficientai
disp('Weight coefficients and bias:' )
disp( net.IW{1} )
disp( net.b{1} )

% modelio verifikacija
Tsu = sim(net,Pu);

% modelio rezultato ir tikru duomenu palyginimas 200  stulpeliu
figure(3)
plot(Tsu,'g-*')
hold on
plot(Tu,'b-*')

xlabel('Years');
ylabel('Sun spot activity');
title('Sun spot activity during 1700 - 1900');
legend('Prediction','Real data')

% modelio rezultato ir tikru duomenu palyginimas visiems stulpeliams
Ts = sim(net,P);

figure(4)
plot(Ts,'g-*')
hold on
plot(T,'b-*')

xlabel('Years');
ylabel('Sun spot activity');
title('Sun spot activity during 1700 - 2014');
legend('Prediction','Real data')

% klaidu apskaiciavimas
E = T-Ts;

% klaidu grafikas
figure(5);
plot(E);
title('Prediction error vector');
xlabel('Year');
ylabel('Error');

% klaidu histogramos
figure(6);
hist(E);
title('Prediction error histogram');
xlabel('Error');
ylabel('Times');

% klaidos apskaiciavimas
mse = mse(E)
mad = median(abs(E))





















