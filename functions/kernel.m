%% Kernel functions
%% whichd: which index in X represent the dimension of X, not the smaples we
%% want to compute.
%% If dimension of X is 1, whichd = 0;
%% typeKert: quar, norm, Epanechnikov
function kernelVal1 = kernel(xVal,whichd,typeKert)

switch (typeKert)
    case 'quar' % this is quartic kernel
        xVals1 = 0.9375*(1-xVal.^2).^2.*(abs(xVal)<=1);
    case 'norm' % this use pdf of normal distribution as kernel
        xVals1 = 1/sqrt(2*pi)*exp(-xVal.^2/2);
    case 'Epanechnikov' % this is Epanechnikov kernel
        xVals1 = 0.75*(1-xVal.^2).*(abs(xVal)<=1);
    case 'normRKHS' % this is Epanechnikov kernel
        xVals1 = exp(-xVal.^2/2);
end
if (whichd~=0)
    kernelVal1 = prod(xVals1,whichd);
else
    kernelVal1 = xVals1;
end
% if (length(xVals1(:,2))==1)
%     kernelVal1 = xVals1';
% end
end