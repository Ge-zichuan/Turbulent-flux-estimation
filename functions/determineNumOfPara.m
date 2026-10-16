function numOfPara = determineNumOfPara(inputNumOfPara,modelType,varargin)
if nargin > 2
    V = varargin;
    bscoef = V{1};
end
switch modelType
    case 'linear'
        numOfPara = inputNumOfPara;
    case 'interactions'
        numOfPara = inputNumOfPara+nchoosek(inputNumOfPara,2);
    case 'purequadratic'
        numOfPara = inputNumOfPara*2;
    case 'quadratic'
        numOfPara = inputNumOfPara*2+nchoosek(inputNumOfPara,2);
    case 'bspline'
        numOfPara = inputNumOfPara*bscoef;
end