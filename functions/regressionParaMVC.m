function [predValue,paraEst,paraFit] = regressionParaMVC(uTarget,xTarget,uObs,xObs,yObs,varargin)
%% Varying coefficient model, Fan and Zhang, 2008, Statistical methods with varying coefficient models
% regressionParaMVC(uEst,xEst,uObs,xObs,yObs,varargin)
% yObs: observed responseY, column vector
% xObs: observed predictor X, column vector
% paraEst is a column vector of [a;vec(b)]

predValue = zeros(size(xTarget,1),1);
uEstShort = unique(uTarget,"rows");
modelType = 'bspline';
BSOrder = 3;
lambdaBS = 1;
numOfKnots = 5;
DataBreakLimit = 0;
DataKnotsLimit = 0;

kernelType = 'Epanechnikov';
bandadj = 5;
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
            case 'BandAdjustment' % bandwidth adjustment tuning parameter.
                bandadj = V{n+1};

            case 'Kernel' % kernel function type.
                %   norm: normal kernel function.
                %   Epanechnikov: Epanechnikov kernel function.
                kernelType = V{n+1};

            case 'Model' % kernel function type.
                %   linear: linear model without intercept
                %   linearInter: linear model with interactions, without
                %   intercept
                %   purequadratic: quadratic of predictors only, no
                %   intercept
                %   quadratic: quadratic model, including linear term, no
                %   intercept
                modelType = V{n+1};

            case 'BSplineOrder'
                BSOrder = V{n+1};
            case 'BSplineLambda'
                lambdaBS = V{n+1};
            case 'NumOfKnots' % Number of Knots.
                numOfKnots = V{n+1};
            case 'DataBreakLimit' % percentage of limits in variable DataBreakMat
                % 0: using min --- max
                % x: using x% --- (100-x)%
                DataBreakLimit = V{n+1};
            case 'DataKnotsLimit' % percentage of limits in variable dataKnotsIdeal
                % 0: using min --- max
                % x: using x% --- (100-x)%
                DataKnotsLimit = V{n+1};

            otherwise
                error('Error! Please check input arguements.')
        end
    end
end


% Estimate the observed parameters from some model
uUnique = unique(uObs, 'stable','rows');
switch modelType
    case {'linear','interactions','purequadratic','quadratic'}
        px = determineNumOfPara(size(xObs,2),modelType);
        paraEst = zeros(size(uEstShort,1),px);
        betaObs = zeros(size(uUnique,1),px);
        for iu = 1:size(uUnique,1)
            subIndex = uObs(:,1)==uUnique(iu,1);
            % fit a linear (or genearlize) model
            regX = xObs(subIndex,:);
            regY = yObs(subIndex,:);
            if size(regX,1)>1
                modelFit = fitlm(regX,regY,modelType,'Intercept',false);
                regCoef = modelFit.Coefficients.Estimate;
                betaObs(iu,:) = regCoef;
            else
                betaObs(iu,:) = NaN;
            end
        end
    case 'bspline'
        % fit an initial B-Spline model
        [~,paraResult] = GAMfun(xObs,yObs,xObs, 'lambda',lambdaBS,...
            'Method','bspline','BSOrder',BSOrder,'NumOfKnots',numOfKnots,'DataBreaks',repmat([0 1],size(xObs,2),1));
        px = size(xObs,2)*size(paraResult{1,1},1)+1;
        paraEst = zeros(size(uEstShort,1),px);
        betaObs = zeros(size(uUnique,1),px);
        % knotObs = zeros(size(uUnique,1),size(xObs,2)*size(paraResult{1,2},1));
        DataBreakMat = zeros(size(xObs,2),2);
        for ig = 1:size(DataBreakMat,1)
            Group = uObs(:,1);
            Value = xObs(:,ig);
            tbl = table(Group, Value);
            if (DataBreakLimit==0)
                DataBreakMat(ig,1) = mean(groupsummary(tbl, "Group", "min", "Value").min_Value);
                DataBreakMat(ig,2) = mean(groupsummary(tbl, "Group", "max", "Value").max_Value);
            else
                DataBreakMat(ig,1) = mean(groupsummary(tbl, "Group", @(x) quantile(x, DataBreakLimit/100), "Value").fun1_Value);
                DataBreakMat(ig,2) = mean(groupsummary(tbl, "Group", @(x) quantile(x, 1-DataBreakLimit/100), "Value").fun1_Value);
            end
        end
        for iu = 1:size(uUnique,1)
            subIndex = uObs(:,1)==uUnique(iu,1);
            % fit a B-Spline model
            regX = xObs(subIndex,:);
            regY = yObs(subIndex,:);
            if size(regX,1)>1
                [~,paraResult] = GAMfun(regX,regY,regX,'lambda',lambdaBS,'Method','bspline',...
                    'BSOrder',BSOrder,'NumOfKnots',numOfKnots,'DataBreaks',DataBreakMat);
                betaObs(iu,:) = cell2mat({cat(1, paraResult{:,1})});
            else
                betaObs(iu,:) = NaN;
            end
        end
end
noNaNrows = ~isnan(betaObs(:,1));
paraFit = betaObs;
for i = 1:px
    if size(uEstShort,1)==1
        paraEstTemp = regressionKernel(betaObs(noNaNrows,i),uUnique(noNaNrows,:),[uEstShort;uEstShort],'BandAdjustment',bandadj,'Kernel',kernelType);
        paraEst(:,i) = paraEstTemp(1,:);
    else
        [uEstNorm,uEstMatC,uEstMatS] = normalize(uUnique(noNaNrows,:),1);
        uEstShortNorm = normalize(uEstShort,"center",uEstMatC,"scale",uEstMatS);
        paraEst(:,i) = regressionKernel(betaObs(noNaNrows,i),uEstNorm,uEstShortNorm,'BandAdjustment',bandadj,'Kernel',kernelType);
    end
end


switch modelType
    case {'linear','interactions','purequadratic','quadratic'}
        xEstNew = x2fx(xTarget,modelType);
        xEstNew = xEstNew(:,2:end);
        for i = 1:size(uEstShort,1)
            subIndex = uTarget(:,1)==uEstShort(i,1);
            predValue(subIndex,1) = xEstNew(subIndex,:)*paraEst(i,:)';
        end
    case 'bspline'
        for i = 1:size(uEstShort,1)
            subIndex = uTarget(:,1)==uEstShort(i,1);
            betaMat = {reshape(paraEst(i,1:end-1),size(paraResult{1,1},1),size(xObs,2))',paraEst(i,end)};
            [predictedY,~,~] = GAMfun(xTarget(subIndex,:),regY,regX,'BSOrder',BSOrder,'NumOfKnots',numOfKnots, ...
                'beta',betaMat,'Method','bsplinePred', ...
                'DataKnotsLimit',DataKnotsLimit, ...
                'DataBreaks',DataBreakMat);
            predValue(subIndex,1) = predictedY;
        end
end



end
