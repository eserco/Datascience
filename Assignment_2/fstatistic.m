% Author: Emile Muller
% Matlab code to calculate the F-statistic of sets of given independant (IV) and dependant variables (DV)

function fstatistic()
% initial conditions, works with 3 categories
IV = [9;7;6.5;8;7.5;7;9.5;8;6.5;7.5;8;6;7;6.5;7.5;8;6;6;6.5;6.5];
H = 1;
M = 2;
L = 3;
DV = [H;H;H;H;H;H;H;H;H;M;M;M;M;M;M;L;L;L;L;L];
F = myOneWayANOVA(IV,DV)
end