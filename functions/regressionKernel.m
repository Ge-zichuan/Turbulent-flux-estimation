function estValue = regressionKernel(yObs,xObs,xEst,varargin)
% yObs: observed responseY, column vector
% xObs: observed predictor X, column vector
% xEst: predictor X needs to be estimated, column vector
%     'BandAdjustment': bandwidth adjustment tuning parameter.
%     'Kernel': kernel function type.
%         % norm: normal kernel function.
%         % Epanechnikov: Epanechnikov kernel function.
%     'XProdType': approaches of product all columns of X.
%         % all: using tensor
%         % columnwise: compute each column, the product them together

bandadj = 1;
kernelType = 'norm';
xProductType = 'all';

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

            case 'XProdType' % approaches of product all columns of X.
                %   all: using tensor
                %   columnwise: compute each column, the product them
                %   together
                xProductType = V{n+1};

            otherwise
                error('Error! Please check input arguements.')
        end
    end
end
% yObs = rand(10,1);
% xObs = rand(10,3);
% xEst = rand(5,3);

samplesize = size(xObs,1);
samplesizeEst = size(xEst,1);
d = size(xEst,2);

if (numel(xObs)*size(xEst,1)>1e8)
    xProductType = 'columnwise';
end

bandx = mean(mad(xObs))*samplesize^(-1/(2*d+1));
if d == 1
    xall = repmat(xEst,1,samplesize)-repmat(xObs',samplesizeEst,1);
    whichD = 0;
    khx = squeeze(kernel(xall/(bandx*bandadj),whichD,kernelType)/(bandx*bandadj)^d);
else
    switch xProductType
        case 'all'
            xall = permute(repmat(xEst,[1 1 samplesize]),[2 1 3])-permute(repmat(xObs,[1 1 samplesizeEst]),[2 3 1]);
            whichD = 1;
            khx = squeeze(kernel(xall/(bandx*bandadj),whichD,kernelType)/(bandx*bandadj)^d);
        case 'columnwise'
            whichD = 0;
            khx = 1;
            for i = 1:d
                xall = repmat(xEst(:,i),1,samplesize)-repmat(xObs(:,i)',samplesizeEst,1);
                khx = khx.*(kernel(xall/(bandx*bandadj),whichD,kernelType)/(bandx*bandadj));
            end
    end
end
% clean 0 khx;
check0 = sum(khx,2);
if sum(check0)==0
    estValue = NaN(size(xEst,1),1);
else
    khx(check0==0,:) = min(khx(check0~=0,:),[],'all');
    estValue = sum(repmat(yObs',samplesizeEst,1).*khx,2)./sum(khx,2);
end

end
