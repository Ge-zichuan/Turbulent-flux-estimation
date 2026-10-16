function loss = regressionBSplinePenalty(BSBeta,yObs,xObs,varargin)
% yObs: observed responseY, column vector
% xObs: observed predictor X, column vector
% xEst: predictor X needs to be estimated, column vector
% yObs = rand(10,1);
% xObs = rand(10,3);
% xEst = rand(5,3);
samplesize = size(xObs,1);
d = size(xObs,2);
BSOrder = 3;
numOfKnots = 5;

if nargin > 3
    %     if rem(nargin,2)
    %         posA = varargin{1};
    %         V = varargin(2:end);
    %     else
    %         V = varargin;
    %     end
    V = varargin;
    for n = 1:2:size(V,2)
        switch V{n}
            case 'lambda' % lambda
                lambda = V{n+1};

            case 'Order' % Order of B-Spline
                BSOrder = V{n+1};

            case 'NumOfKnots' % approaches of product all columns of X.
                %   all: using tensor
                %   columnwise: compute each column, the product them
                %   together
                numOfKnots = V{n+1};

            otherwise
                error('Error! Please check input arguements.')
        end
    end
end
%% fit and predict the model
% find the knots, equal distance is the best
dataKnotsIdeal = linspace(min(xObs),max(xObs),numOfKnots);
[minVal,minInd] = min(abs(xObs-dataKnotsIdeal),[],1);
closestCounts = sum(abs(xObs-dataKnotsIdeal)==repmat(minVal,samplesize,1),1);
if sum(closestCounts~=1)>0 % more than one cloest knots
    %
end
dataKnots = sort(xObs(minInd));

knotT = [repmat(min(dataKnots),BSOrder,1);dataKnots;repmat(max(dataKnots),BSOrder,1)];

if d == 1
    [sortedX,sortID,~] = unique(xObs);
    BSXvalue = BSpline(sortedX,knotT,BSOrder);
    fittedY = BSXvalue*BSBeta;
    
    fittedYprime2 = diff(diff(fittedY)./diff(sortedX))./diff(sortedX(1:end-1));
    if size(fittedYprime2,1) == 1
        loss = sum((yObs(sortID)-fittedY).^2)+lambda*fittedYprime2^2;
    else
        loss = sum((yObs(sortID)-fittedY).^2)+lambda*sum(trapz(sortedX(2:end-1),fittedYprime2).^2);
    end
    % fprintf(strcat(num2str([size(sortedX) size(fittedYprime2)]),"\n"));

elseif d > 1

else
    fprintf("Incorrect dimension of variable X!!!");
end



end