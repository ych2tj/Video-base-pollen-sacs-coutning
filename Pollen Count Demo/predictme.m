% prediction function ***************************************************

function [predictS,predictCov] = predictme(Kalman)

%      Kalman.preS = Kalman.A * Kalman.CorS;
%      Kalman.PreCov = Kalman.A * Kalman.CorCov * Kalman.A' + Kalman.Q0;
     predictS = Kalman.A * Kalman.CorS;
     predictCov = Kalman.A * Kalman.CorCov * Kalman.A' + Kalman.Q0;