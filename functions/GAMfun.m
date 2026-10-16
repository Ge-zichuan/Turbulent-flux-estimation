function [predictedY,paraResult,options] = GAMfun(xPred,yObs,xObs,varargin)
% Genearlized Additive Model
% g(E(Y)) = alpha_0+f_1(X_1)+f_2(X_2)+...+f_p(X_p)
% Input:
%       predX: variable X at where predict the responses.
%       obsY: observed response Y.
%       obsX: observed variable X.
%       varargin: arguments see below.
%            'lambda'
%               lambda value for the penalty function(default:1)
%            'maxIteration' %
%               number of maximum iterations (default:100)
%            'tolerance' %
%               tolerance when to stop (default:0.011)
%            'stepDetail'
%               off (default): not show the detail
%               on: show the detail of algorithm
%            'Kernel': kernel function type.
%               norm: normal kernel function.
%               Epanechnikov (default): Epanechnikov kernel function.
%            'Method': how to get bandwidth h
%               auto (default): kernel method, minimize h from the penalty optimand
%               manual: kernel method, manually assign h
%               bspline: B-Spline.
%            'bandwidth': how to get bandwidth h
%               auto: minimize h from the penalty optimand
%               manual: manually assign h
%            'BSOrder':
%               order of B-Spline (default: 3)
%            'interaction' % interaction term x_ix_j
%               on:
%               off (default):
% Output:
%       predictedY: predicted response Y using GAM.
%       parameters (optional): parameters computed within GAM.
%% Basic settings
optionsMin = optimoptions('fminunc','Display','off','MaxIterations',100);
lambdaValue = 1;
methodType = "auto";
interactionSwitch = "off";
bandwidthInput = [];
BSOrder = 3;
numOfKnots = 5;
tol = 0.01;
iterTol = 100;
stepDisplay = "off";
options = cell(3,1); % options during algorithm
DataKnotsLimit = 0;

if nargin > 2
    V = varargin;
    for n = 1:2:size(V,2)
        switch V{n}
            case 'lambda' % lambda
                lambdaValue = V{n+1};
            case 'maxIteration' % number of maximum iterations
                iterTol = V{n+1};
            case 'tolerance' % tolerance when to stop
                tol = V{n+1};
            case 'stepDetail'
                stepDisplay = V{n+1};

            case 'Kernel' % kernel function type.
                %   norm: normal kernel function.
                %   Epanechnikov: Epanechnikov kernel function.
                kernelType = V{n+1};
            case 'Method' % how to get bandwidth h
                %   auto: kernel method, minimize h from the penalty optimand
                %   manual: kernel method, manually assign h
                %   bspline: B-Spline.
                methodType = V{n+1};

            case 'bandwidth' % how to get bandwidth h
                %   auto: minimize h from the penalty optimand
                %   manual: manually assign h
                bandwidthInput = V{n+1};
            case 'BSOrder' % order of B-Spline
                BSOrder = V{n+1};
            case 'NumOfKnots' % Number of Knots.
                numOfKnots = V{n+1};
            case 'knot' % input of external knot for prediction
                inputKnotT = V{n+1};
            case 'beta' % input of external beta
                inputPara = V{n+1};
                inputBeta = inputPara{1,1};
                inputAlpha = inputPara{1,2};
            case 'DataBreaks' % Knots vector
                DataBreaksMat = V{n+1};
            case 'interaction' % interaction term x_ix_j
                %   on:
                %   off (default):
                interactionSwitch = V{n+1};
                interactionVarIndex = nchoosek(1:size(xObs,2),2);
            case 'DataKnotsLimit' % percentage of limits in variable dataKnotsIdeal
                % 0: using min --- max
                % x: using x% --- (100-x)%
                DataKnotsLimit = V{n+1};

            otherwise
                error('Error! Please check input arguements.')
        end
    end
end

if (isempty(bandwidthInput) && strcmp(methodType,"manual"))
    warning("Please input bandwidth! ('bandwidth')");
end
%% Initialization
samplesizePred = size(xPred,1);
samplesizeObs = size(xObs,1);
switch interactionSwitch
    case "off"
        numOfPredictor = size(xObs,2);

    case "on"
        numOfPredictor = size(xObs,2)+nchoosek(size(xObs,2),2);
end

if (exist('DataBreaksMat','var') == 1)
else
    DataBreaksMat = NaN(numOfPredictor,1);
end

currentFx = zeros(samplesizeObs,numOfPredictor);
predictFx = zeros(samplesizePred,numOfPredictor);
switch methodType
    case 'auto'
        bandwidths = zeros(1,numOfPredictor);
        paraResult = [];
    case 'bspline'
        [~,~,~] = regressionBSpline(yObs,xObs(:,1),xObs(:,1), ...
            'Order',BSOrder,'DataKnotsLimit',DataKnotsLimit);
        if strcmp(interactionSwitch,"on")
            paraResult = cell(numOfPredictor+nchoosek(size(xObs,2),2),2);
        else
            paraResult = cell(numOfPredictor,2);
        end
    case 'bsplinePred'
        if size(yObs,1) == 1
            [~,~,~] = regressionBSpline([yObs;yObs],[xObs(:,1);xObs(:,1)],[xObs(:,1);xObs(:,1)], ...
                'Order',BSOrder,'DataKnotsLimit',DataKnotsLimit);
        else
            [~,~,~] = regressionBSpline(yObs,xObs(:,1),xObs(:,1), ...
                'Order',BSOrder,'DataKnotsLimit',DataKnotsLimit);
        end
        if strcmp(interactionSwitch,"on")
            paraResult = cell(numOfPredictor+1+nchoosek(size(xObs,2),2),2);
        else
            paraResult = cell(numOfPredictor+1,2);
        end
    case 'bspline-penalty'
        [~,~,~] = regressionBSpline(yObs,xObs(:,1),xObs(:,1), ...
            'Order',BSOrder,'DataKnotsLimit',DataKnotsLimit);
        paraResult = cell(numOfPredictor+1,2);
    case 'manual'
        paraResult = cell(numOfPredictor+1,2);
end

alphaEst = mean(yObs);
currentYHat = alphaEst*ones(samplesizeObs,1)+sum(currentFx,2);
diffInY = 1;
iter = 1;
predictionOld = currentYHat*0;

if samplesizeObs > 40000
    samplesRnd = randsample(samplesizeObs,1000,false);
else
    samplesRnd = 1:samplesizeObs;
end

if strcmp(methodType,"bsplinePred")
    for ii = numOfPredictor+1:-1:1
        if ii==numOfPredictor+1 % updating intercept alpha_0
            % alphaEst = mean(currentYHat);
            alphaEst = inputAlpha;
        else % updating f_i
            responseIter = yObs-alphaEst-sum(currentFx(:,setdiff(1:numOfPredictor,ii)),2,"omitnan");
            % fit the iteration residual to ith variable
            if strcmp(interactionSwitch,"off")
                predictEach2 = regressionBSpline(responseIter,xObs(:,ii),xPred(:,ii),'Prediction',true, ...
                    'Order',BSOrder,'NumOfKnots',numOfKnots, ...
                    'BetaInput',inputBeta(ii,:)', ...
                    'DataKnotsLimit',DataKnotsLimit, ...
                    'DataBreaks',DataBreaksMat(ii,:));
                paraResult = [];
            elseif strcmp(interactionSwitch,"on")
                if ii<=size(xObs,2)
                    predictEach2 = regressionBSpline(responseIter,xObs(:,ii),xPred(:,ii),'Prediction',true, ...
                        'Order',BSOrder,'NumOfKnots',numOfKnots,'BetaInput',inputBeta(ii,:), ...
                        'DataKnotsLimit',DataKnotsLimit,'DataBreaks',DataBreaksMat(ii,:));
                    paraResult = [];
                else
                    varInd = interactionVarIndex(ii-size(xObs,2),:);
                    predictEach2 = regressionBSpline(responseIter,prod(xObs(:,varInd),2),prod(xPred(:,varInd),2),'Prediction',true, ...
                        'Order',BSOrder,'NumOfKnots',numOfKnots, ...
                        'BetaInput',inputBeta(ii,:), ...
                        'DataKnotsLimit',DataKnotsLimit, ...
                        'DataBreaks',DataBreaksMat(ii,:));
                    paraResult = [];
                end
            end
            predictEach2 = predictEach2-mean(predictEach2,"omitnan");
            predictFx(:,ii) = predictEach2;
        end
    end
else
    while (diffInY > tol && iter<iterTol)
        if strcmp(stepDisplay,"on")
            fprintf(strcat("iteration:",num2str(iter)));
        end
        iter = iter + 1;
        for ii = numOfPredictor+1:-1:1
            if ii==numOfPredictor+1 % updating intercept alpha_0
                alphaEst = mean(currentYHat);
                paraResult{ii,1} = alphaEst;
            else % updating f_i
                responseIter = yObs-alphaEst-sum(currentFx(:,setdiff(1:numOfPredictor,ii)),2,"omitnan");
                % fit the iteration residual to ith variable
                if strcmp(interactionSwitch,"off")
                    switch methodType
                        case 'auto'
                            if ii == 22
                                a = 1;
                            end
                            bandwidthOpt = fminunc(@(x)smoothKernel(x,responseIter(samplesRnd),xObs(samplesRnd,ii), ...
                                'lambda',lambdaValue),1,optionsMin);
                            bandwidths(ii) = exp(bandwidthOpt);
                            predictEach1 = regressionKernel(responseIter,xObs(:,ii),xObs(:,ii),'BandAdjustment',exp(bandwidthOpt));
                            paraResult = bandwidths;
                        case 'manual'
                            predictEach1 = regressionKernel(responseIter,xObs(:,ii),xObs(:,ii),'BandAdjustment',bandwidthInput);
                            paraResult{ii,1} = bandwidthInput;
                        case 'bspline'
                            [predictEach1,BSBeta,knotT] = regressionBSpline(responseIter,xObs(:,ii),xObs(:,ii), ...
                                'NumOfKnots',numOfKnots,'Order',BSOrder, ...
                                'DataKnotsLimit',DataKnotsLimit, ...
                                'DataBreaks',DataBreaksMat(ii,:));
                            paraResult{ii,1} = BSBeta;paraResult{ii,2} = knotT;
                        case 'bspline-penalty'
                            [~,BSBeta,knotT] = regressionBSpline(responseIter(samplesRnd),xObs(samplesRnd,ii),xObs(samplesRnd,ii), ...
                                'NumOfKnots',numOfKnots,'Order',BSOrder, ...
                                'DataKnotsLimit',DataKnotsLimit, ...
                                'DataBreaks',DataBreaksMat(ii,:));
                            betaSmooth = fminunc(@(x)regressionBSplinePenalty(x,responseIter(samplesRnd),xObs(samplesRnd,ii), ...
                                'lambda',lambdaValue,'NumOfKnots',numOfKnots,'Order',BSOrder,'DataBreaks',DataBreaksMat(ii,:)),BSBeta,optionsMin);
                            predictEach1 = regressionBSpline(responseIter,xObs(:,ii),xObs(:,ii),'Order',BSOrder, ...
                                'Prediction',true,'BetaInput',betaSmooth, ...
                                'DataKnotsLimit',DataKnotsLimit, ...
                                'DataBreaks',DataBreaksMat(ii,:));
                            paraResult{ii,1} = betaSmooth';paraResult{ii,2} = knotT;
                    end
                elseif strcmp(interactionSwitch,"on")
                    if ii<=size(xObs,2)
                        switch methodType
                            case 'auto'
                                bandwidthOpt = fminunc(@(x)smoothKernel(x,responseIter(samplesRnd),xObs(samplesRnd,ii), ...
                                    'lambda',lambdaValue),1,optionsMin);
                                bandwidths(ii) = exp(bandwidthOpt);
                                predictEach1 = regressionKernel(responseIter,xObs(:,ii),xObs(:,ii),'BandAdjustment',exp(bandwidthOpt));
                                paraResult = bandwidths;
                            case 'manual'
                                predictEach1 = regressionKernel(responseIter,xObs(:,ii),xObs(:,ii),'BandAdjustment',bandwidthInput);
                                paraResult{ii,1} = bandwidthInput;
                            case 'bspline'
                                [predictEach1,BSBeta,knotT] = regressionBSpline(responseIter(samplesRnd),xObs(samplesRnd,ii),xObs(samplesRnd,ii), ...
                                    'NumOfKnots',numOfKnots,'Order',BSOrder, ...
                                    'DataKnotsLimit',DataKnotsLimit, ...
                                    'DataBreaks',DataBreaksMat(ii,:));
                                paraResult{ii,1} = BSBeta;paraResult{ii,2} = knotT;
                            case 'bspline-penalty'
                                [~,BSBeta,knotT] = regressionBSpline(responseIter(samplesRnd),xObs(samplesRnd,ii),xObs(samplesRnd,ii), ...
                                    'NumOfKnots',numOfKnots,'Order',BSOrder, ...
                                    'DataKnotsLimit',DataKnotsLimit, ...
                                    'DataBreaks',DataBreaksMat(ii,:));
                                betaSmooth = fminunc(@(x)regressionBSplinePenalty(x,responseIter(samplesRnd),xObs(samplesRnd,ii), ...
                                    'lambda',lambdaValue,'NumOfKnots',numOfKnots,'Order',BSOrder,'DataBreaks',DataBreaksMat(ii,:)),BSBeta,optionsMin);
                                predictEach1 = regressionBSpline(responseIter,xObs(:,ii),xObs(:,ii),'Order',BSOrder, ...
                                    'Prediction',true,'BetaInput',betaSmooth, ...
                                    'DataKnotsLimit',DataKnotsLimit, ...
                                    'DataBreaks',DataBreaksMat(ii,:));
                                paraResult{ii,1} = betaSmooth';paraResult{ii,2} = knotT;
                        end
                    else
                        varInd = interactionVarIndex(ii-size(xObs,2),:);
                        switch methodType
                            case 'auto'
                                bandwidthOpt = fminunc(@(x)smoothKernel(x,responseIter(samplesRnd),prod(xObs(samplesRnd,varInd),2), ...
                                    'lambda',lambdaValue),1,optionsMin);
                                bandwidths(ii) = exp(bandwidthOpt);
                                predictEach1 = regressionKernel(responseIter,prod(xObs(:,varInd),2),prod(xObs(:,varInd),2),'BandAdjustment',exp(bandwidthOpt));
                                paraResult = bandwidths;
                            case 'manual'
                                predictEach1 = regressionKernel(responseIter,prod(xObs(:,varInd),2),prod(xObs(:,varInd),2),'BandAdjustment',bandwidthInput);
                                paraResult{ii,1} = bandwidthInput;
                            case 'bspline'
                                [predictEach1,BSBeta,knotT] = regressionBSpline(responseIter(samplesRnd), ...
                                    prod(xObs(samplesRnd,varInd),2),prod(xObs(samplesRnd,varInd),2), ...
                                    'NumOfKnots',numOfKnots,'Order',BSOrder, ...
                                    'DataKnotsLimit',DataKnotsLimit, ...
                                    'DataBreaks',DataBreaksMat(varInd,:));
                                paraResult{ii,1} = BSBeta;paraResult{ii,2} = knotT;
                            case 'bspline-penalty'
                                [~,BSBeta,knotT] = regressionBSpline(responseIter(samplesRnd),prod(xObs(samplesRnd,varInd),2),prod(xObs(samplesRnd,varInd),2), ...
                                    'NumOfKnots',numOfKnots,'Order',BSOrder, ...
                                    'DataKnotsLimit',DataKnotsLimit, ...
                                    'DataBreaks',DataBreaksMat(varInd,:));
                                betaSmooth = fminunc(@(x)regressionBSplinePenalty(x,responseIter(samplesRnd),prod(xObs(samplesRnd,varInd),2), ...
                                    'lambda',lambdaValue,'NumOfKnots',numOfKnots,'Order',BSOrder,'DataBreaks',DataBreaksMat(varInd,:)),BSBeta,optionsMin);
                                predictEach1 = regressionBSpline(responseIter,prod(xObs(:,varInd),2),prod(xObs(:,varInd),2), ...
                                    'Order',BSOrder,'Prediction',true,'BetaInput',betaSmooth, ...
                                    'DataKnotsLimit',DataKnotsLimit, ...
                                    'DataBreaks',DataBreaksMat(varInd,:));
                                paraResult{ii,1} = betaSmooth';paraResult{ii,2} = knotT;
                        end
                    end
                end
                predictEach1 = predictEach1-mean(predictEach1,"omitnan");
                currentFx(:,ii) = predictEach1;
            end
        end
        currentYHat = alphaEst+sum(currentFx,2,"omitnan");
        diffInY = mean(abs(predictionOld-currentYHat)./abs(predictionOld));
        predictionOld = currentYHat;
        if strcmp(stepDisplay,"on")
            fprintf(repmat('\b',1,numel(num2str(iter))+10));
        end
    end
    for ii = 1:numOfPredictor+1
        if ii==numOfPredictor+1 % updating intercept alpha_0
        else % updating f_i
            responseIter = yObs-alphaEst-sum(currentFx(:,setdiff(1:numOfPredictor,ii)),2,"omitnan");
            % fit the iteration residual to ith variable
            if strcmp(interactionSwitch,"off")
                switch methodType
                    case 'auto'
                        bandwidthOpt = log(bandwidths(ii));
                        predictEach2 = regressionKernel(responseIter,xObs(:,ii),xPred(:,ii),'BandAdjustment',exp(bandwidthOpt));
                    case 'manual'
                        predictEach2 = regressionKernel(responseIter,xObs(:,ii),xPred(:,ii),'BandAdjustment',bandwidthInput);
                    case 'bspline'
                        predictEach2 = regressionBSpline(responseIter,xObs(:,ii),xPred(:,ii),'Prediction',true, ...
                            'Order',BSOrder,'BetaInput',paraResult{ii,1}, ...
                            'DataKnotsLimit',DataKnotsLimit, ...
                            'KnotInput',paraResult{ii,2});
                    case 'bspline-penalty'
                        predictEach2 = regressionBSpline(responseIter,xObs(:,ii),xPred(:,ii),'Order',BSOrder, ...
                            'Prediction',true,'BetaInput',paraResult{ii,1}, ...
                            'DataKnotsLimit',DataKnotsLimit, ...
                            'KnotInput',paraResult{ii,2});
                end
            elseif strcmp(interactionSwitch,"on")
                if ii<=size(xObs,2)
                    switch methodType
                        case 'auto'
                            bandwidthOpt = log(bandwidths(ii));
                            predictEach2 = regressionKernel(responseIter,xObs(:,ii),xPred(:,ii),'BandAdjustment',exp(bandwidthOpt));
                        case 'manual'
                            predictEach2 = regressionKernel(responseIter,xObs(:,ii),xPred(:,ii),'BandAdjustment',bandwidthInput);
                        case 'bspline'
                            predictEach2 = regressionBSpline(responseIter,xObs(:,ii),xPred(:,ii),'Prediction',true, ...
                                'Order',BSOrder,'BetaInput',paraResult{ii,1}, ...
                                'DataKnotsLimit',DataKnotsLimit, ...
                                'DataBreaks',DataBreaksMat(varInd,:));
                        case 'bspline-penalty'
                            predictEach2 = regressionBSpline(responseIter,xObs(:,ii),xPred(:,ii),'Order',BSOrder, ...
                                'Prediction',true,'BetaInput',paraResult{ii,1}, ...
                                'DataKnotsLimit',DataKnotsLimit, ...
                                'DataBreaks',DataBreaksMat(varInd,:));
                    end
                else
                    varInd = interactionVarIndex(ii-size(xObs,2),:);
                    switch methodType
                        case 'auto'
                            bandwidthOpt = log(bandwidths(ii));
                            predictEach2 = regressionKernel(responseIter,prod(xObs(:,varInd),2),prod(xPred(:,varInd),2),'BandAdjustment',exp(bandwidthOpt));
                        case 'manual'
                            predictEach2 = regressionKernel(responseIter,prod(xObs(:,varInd),2),prod(xPred(:,varInd),2),'BandAdjustment',bandwidthInput);                                paraResult = bandwidthInput;
                        case 'bspline'
                            predictEach2 = regressionBSpline(responseIter,prod(xObs(:,varInd),2),prod(xPred(:,varInd),2),'Prediction',true, ...
                                'Order',BSOrder,'BetaInput',paraResult{ii,1}, ...
                                'DataKnotsLimit',DataKnotsLimit, ...
                                'DataBreaks',DataBreaksMat(varInd,:));
                        case 'bspline-penalty'
                            predictEach2 = regressionBSpline(responseIter,prod(xObs(:,varInd),2),prod(xPred(:,varInd),2), ...
                                'Order',BSOrder,'Prediction',true,'BetaInput',paraResult{ii,1}, ...
                                'DataKnotsLimit',DataKnotsLimit, ...
                                'DataBreaks',DataBreaksMat(varInd,:));
                    end
                end
            end
            predictEach2 = predictEach2-mean(predictEach2,"omitnan");
            predictFx(:,ii) = predictEach2;
        end
    end
    if strcmp(stepDisplay,"on")
        fprintf(repmat('\b',1,numel(num2str(iter))+10));
    end
    options{1,1} = iter;
    options{2,1} = diffInY;
end

%% sample data to test each different method
% xObs = rand(100,1);
% responseY = 1-(xObs-0.5).^2+normrnd(0,0.08,100,1);
% [predictionOld,BSBeta,~] = regressionBSpline(responseY,xObs,xObs);
% regressionBSplinePenalty(BSBeta,responseY,xObs,'lambda',2)
% scatter(xObs,responseY,'k.');
% hold on;
% scatter(xObs,predictionOld,'b*');
% hold off;

% scatter(responseY,predictionOld);
% hold on
% scatter(1:samplesize,predictionOld,'b*');
% scatter(1:samplesize,responseY,'k.');
% hold off;

%% Output results
predictedY = alphaEst+sum(predictFx,2,"omitnan");