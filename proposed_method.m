%% Code for "A conditional parameterization framework for turbulent flux estimation across heterogeneous land surfaces"

%% Add path
scriptFolder = fileparts(which("proposed_method.m"));
addpath(genpath(scriptFolder));

%% Read data
% stationID: 1.Horqin, 2.Beijing, 3.Huainan
stationID = 1;
featureTrain = table2array(readtable(strcat('station',num2str(stationID),'_feature.csv')));
factorTrain = table2array(readtable(strcat('station',num2str(stationID),'_factor.csv')));
responseTrain = table2array(readtable(strcat('station',num2str(stationID),'_response.csv')));

% For target feature, readers can select any set you want (cross-validation 
% or predicting a new set). Here we use the training set for illustration 
% purpose only.
featureTarget = featureTrain;
factorTarget = factorTrain;

%% proposed method
[predValue,paraEst,paraFit] = regressionParaMVC(featureTarget,factorTarget, ...
    featureTrain, factorTrain, responseTrain);

% predValue: predicted response \hat{Y}
% paraEst: predicted parameters for the target data
% paraFit: estimated parameters from the train data. 
