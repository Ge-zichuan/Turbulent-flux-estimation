function loss = smoothKernel(bandwidth,responseY,predictorX,varargin)
%% using penalty to control the smoothness of kernel regression.
if nargin > 3
    V = varargin;
    for n = 1:2:size(V,2)
        switch V{n}
            case 'lambda' % lambda
                lambda = V{n+1};

            case 'Kernel' % kernel function type.
                %   norm: normal kernel function.
                %   Epanechnikov: Epanechnikov kernel function.
                kernelType = V{n+1};

            otherwise
                error('Error! Please check input arguements.')
        end
    end
end

[sortedX,sortID,~] = unique(predictorX);
fittedY = regressionKernel(responseY,predictorX,sortedX,'BandAdjustment',exp(bandwidth));
if size(predictorX,2)==1
    fittedYprime2 = diff(diff(fittedY)./diff(sortedX))./diff(sortedX(1:end-1));
    if size(fittedYprime2,1) == 1
        loss = sum((responseY(sortID)-fittedY).^2)+lambda*fittedYprime2^2;
    else
        loss = sum((responseY(sortID)-fittedY).^2)+lambda*sum(trapz(sortedX(2:end-1),fittedYprime2).^2);
    end
elseif size(predictorX,2)==2
    fittedYprime21 = diff(diff(fittedY)./diff(sortedX(:,1)))./diff(sortedX(1:end-1,1));
    % fittedYprime22 = diff(diff(fittedY)./diff(sortedX))./diff(sortedX(1:end-1));
    % fittedYprime212 = 2*diff(diff(fittedY)./diff(sortedX))./diff(sortedX(1:end-1));
    if size(fittedYprime2,1) == 1
        loss = sum((responseY(sortID)-fittedY).^2)+lambda*fittedYprime2^2;
    else
        loss = sum((responseY(sortID(:,1))-fittedY).^2)+lambda*sum(trapz(sortedX(2:end-1,1),fittedYprime21).^2);
    end
else
    fprintf("Multiple variables. Need adjustment!\n");
    fittedYprime2 = diff(diff(fittedY)./diff(sortedX))./diff(sortedX(1:end-1));
    if size(fittedYprime2,1) == 1
        loss = sum((responseY(sortID)-fittedY).^2)+lambda*fittedYprime2^2;
    else
        loss = sum((responseY(sortID(:,1))-fittedY).^2)+lambda*sum(trapz(sortedX(2:end-1,1),fittedYprime2).^2);
    end
    
end

end
