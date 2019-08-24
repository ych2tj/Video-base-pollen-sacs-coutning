% correction function ***************************************************

function [CorK,CorS,CorCov] = correctme(Kalman)
     
     I = eye(2);
     CorK = Kalman.PreCov/(Kalman.PreCov + Kalman.R);
     CorS = CorK * Kalman.M + (I - CorK)*Kalman.preS;
     CorCov = (I - CorK)*Kalman.PreCov;