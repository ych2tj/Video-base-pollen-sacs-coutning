% this function calcualte the distance between one Kalman prediction and
% all of the detections
function dists = distanceme(kalobj,detections)

dist = [];
Numdet = size(detections,1);
   for i = 1:Numdet
       dist(i) = sqrt( (kalobj(1)- detections(i,1))^2 + (kalobj(2)- detections(i,2))^2 );
   end

dists = dist;