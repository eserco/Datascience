% Author: Emile Muller
% Matlab code to calculate the F-statistic of sets of given independant (IV) and dependant variables (DV)
function F = myOneWayANOVA(IV,DV)

errormsg1 = 'IV and DV must have the same size';
if numel(IV) ~= numel(DV)
    error(errormsg1)
end

errormsg2 = 'DV has not enough values';
if numel(DV) <= 1
    error(errormsg2)
end

DV2 = accumarray(DV,1);
%DV2 = [9;6;5];

% A: all the 'H' elements of IV
A = IV(1 : DV2(1));
meanA = mean(A);

% B: all the 'M' elements of IV
B = IV(DV2(1)+1 : DV2(1)+DV2(2));
meanB = mean(B);

% C: all the 'L' elements of IV
C = IV(DV2(1)+DV2(2)+1 : DV2(1)+DV2(2)+DV2(3));
meanC = mean(C);

% calculating weighted mean of means
grandMean = (DV2(1) * meanA + DV2(2) * meanB + DV2(3) * meanC) / (DV2(1) + DV2(2) + DV2(3));

% calculating the mean of sums square between groups
SSB = ((DV2(1)*(meanA - grandMean).^2) + (DV2(2)*(meanB - grandMean).^2) + (DV2(3)*(meanC - grandMean).^2)) / (numel(DV2)-1);

sumVarA = sum((A - meanA).^2);
varianceA = sumVarA / (DV2(1) - 1);;

sumVarB = sum((B - meanB).^2);
varianceB = sumVarB / (DV2(2) - 1);;

sumVarC = sum((C - meanC).^2);
varianceC = sumVarC / (DV2(3) - 1);;

% calculating the mean of sums square within groups
SSW = ((DV2(1)-1)*varianceA + (DV2(2)-1)*varianceB + (DV2(3)-1)*varianceC) / (DV2(1)-1 + DV2(2)-1 + DV2(3)-1);

F = SSB / SSW;
end