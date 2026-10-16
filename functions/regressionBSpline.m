function [estValue,BSBeta,knotT] = regressionBSpline(yObs,xObs,xEst,varargin)
% yObs: observed responseY, column vector
% xObs: observed predictor X, column vector
% xEst: predictor X needs to be estimated, column vector
% yObs = rand(10,1);
% xObs = rand(10,3);
% xEst = rand(5,3);
samplesize = size(xObs,1);
samplesizeEst = size(xEst,1);
d = size(xEst,2);
BSOrder = 3;
xProductType = 'all';
numOfKnots = 5;
predictSwitch = false;
DataKnotsLimit = 0;

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
            case 'Order' % Order of B-Spline
                BSOrder = V{n+1};

            case 'Prediction' % Model of fitting or prediction
                predictSwitch = V{n+1};
            case 'KnotInput' % Knots vector
                knotT = V{n+1};
            case 'BetaInput' % Input of parameter \beta
                BSBeta = V{n+1};

            case 'DataBreaks' % Knots vector
                DataBreaks = V{n+1};
            case 'NumOfKnots' % Number of Knots.
                numOfKnots = V{n+1};
            case 'DataKnotsLimit' % percentage of limits in variable dataKnotsIdeal
                % 0: using min --- max
                % x: using x% --- (100-x)%
                DataKnotsLimit = V{n+1};


            otherwise
                error('Error! Please check input arguements.')
        end
    end
end
if (predictSwitch==false)
%% fit and predict the model
    % find the knots, equal distance is the best
    if (DataKnotsLimit==0)
        dataKnotsIdeal = linspace(min(xObs),max(xObs),numOfKnots);
    else
        % temp = linspace(min(xObs),max(xObs),numOfKnots+2);
        % dataKnotsIdeal = temp(2:end-1);
        dataKnotsIdeal = linspace(quantile(xObs,DataKnotsLimit/100),quantile(xObs,1-DataKnotsLimit/100),numOfKnots);
    end
    if (exist('DataBreaks','var') == 1) && (sum(isnan(DataBreaks),'all') == 0)
        dataKnotsIdeal = linspace(DataBreaks(1),DataBreaks(2),numOfKnots);
    end
    [minVal,minInd] = min(abs(xObs-dataKnotsIdeal),[],1);
    closestCounts = sum(abs(xObs-dataKnotsIdeal)==repmat(minVal,samplesize,1),1);
    if sum(closestCounts~=1)>0 % more than one cloest knots
        %
    end
    dataKnots = sort(xObs(minInd));
    knotT = [repmat(min(dataKnots),BSOrder,1);dataKnots;repmat(max(dataKnots),BSOrder,1)];

    if d == 1
        BSXvalue = BSpline(xObs,knotT,BSOrder);
        BSBeta = fitlm(BSXvalue,yObs,'Intercept',false);
        BSBeta = table2array(BSBeta.Coefficients(:,1));
        BSXvalue = BSpline(xEst,knotT,BSOrder);
        estValue = BSXvalue*BSBeta;
    elseif d > 1

    else
        fprintf("Incorrect dimension of variable X!!!");
    end
else
%% Direct predction, without fitting.
    % if (~exist("knotT","var") || ~exist("BSBeta","var"))
    %     fprintf("Please input Knots or Beta!!!");
    % else
    %     BSXvalue = BSpline(xEst,knotT,BSOrder);
    %     estValue = BSXvalue*BSBeta;
    % end
    if (exist("BSBeta","var") && exist("knotT","var"))
        BSXvalue = BSpline(xEst,knotT,BSOrder);
        estValue = BSXvalue*BSBeta;
    elseif (exist('DataBreaks','var') == 1) && (sum(isnan(DataBreaks),'all') == 0)
        dataKnots = linspace(DataBreaks(1),DataBreaks(2),numOfKnots);
        knotT = [repmat(min(dataKnots),BSOrder,1);dataKnots';repmat(max(dataKnots),BSOrder,1)];
        BSXvalue = BSpline(xEst,knotT,BSOrder);
        estValue = BSXvalue*BSBeta;
    end
end



end