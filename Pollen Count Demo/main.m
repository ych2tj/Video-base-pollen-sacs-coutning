%% Pollen counting model
%**************************************************************************
% Author: Robert (Cheng) Yang
% Publish time: 2019-07-24
% Summary: this demo shows bee and pollen sacs detection, draw the bee
% flying trajectory and pollen count number.

% ########################################################################
% Mention: the model running very slow. Please wait 5s, the image will be
% come out. The main problem is that the Hough transform coding is not
% vectorized. The "for loop" slow down the programme.
% ########################################################################

% The Matlab essential tools: 
%   Image processing tool box
%   computer vision tool box
%   deep learning tool box
%   neural network tool box
% The model has been tested in Matlab 2018b.

% Methodology
% Motion detection and colour thresholding to detect bees
% Kalman filter and Hungarian method to track bees.
% Hough transform to find bee position in merge bee blobs.
% Faster RCNN model to detect pollen sacs


% Windows
% Yellow bounding boxes: single bee detection.
% Red bounding boxes: Bee position prediction for the current frame
% Cyan boudning boxes: Kalman correction of bee location
% Purple small bounding boxes: pollen sacs detection with probalility
% Purple circle with numbers on the left bottom of bounding box:
%  Number followiing format pol/sig:
%    pol: pollen detection number in frames
%    sig: number of bee detected as single bee in frames
%    Problablity of bee carrying pollen is pol/sig*100 (%)
%    If the % greater than 46%, the bee would be identified carrying pollen
%    sacs. The threshold of 46% was got from ROC analysis. More detrail see
%    the thesis.
% *************************************************************************
% create system objects 
% create system objects used for reading video, detecting moving objects,
% and displaying the results
       % create a video file reader
       objreader = vision.VideoFileReader('MP41080p20160423GOPR0983_10am_Prot.mp4',...
                    'ImageColorSpace', 'RGB');boundary = 0.74; % 0.74
%        objreader = vision.VideoFileReader('MP41080p20161105GOPR1035_10.46am.mp4',...% Vertical
%                     'ImageColorSpace', 'RGB');boundary = 0.85;  %
%        objreader = vision.VideoFileReader('MP41080p20160412GOPR0920_11.26am.mp4',...% Vertical
%                     'ImageColorSpace', 'RGB');boundary = 0.82;  % 
%        objreader = vision.VideoFileReader('MP41080p20160327_GOPR0877_14pm.mp4',...% Vertical
%                     'ImageColorSpace', 'RGB');boundary = 0.84; % 0.84
%        objreader = vision.VideoFileReader('MP41080p20160313GOPR0850_4pm.mp4',...% Vertical
%                     'ImageColorSpace', 'RGB'); boundary = 0.87;% 0.87

        % create two video players, one to display the video,
        % and one to display the foreground mask
        objvideoPlayer = vision.VideoPlayer('Position', [20, 200, 700, 400]);
        objmaskPlayer = vision.VideoPlayer('Position', [740, 200, 700, 400]);
        
        % The foreground detector is used to segment moving objects from
        % the background.       
        objdetector = vision.ForegroundDetector('NumGaussians', 5, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7,...
            'InitialVariance', (30/255)^2,... % initial standard deviation of 30/255
            'LearningRate',0.001);
        
        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.         
        objblobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', false, 'CentroidOutputPort', true, ...
            'OrientationOutputPort',true,'MinimumBlobArea', 1000);% no angle:1000,45degree:800
                
        % object for segment merged bees 
        singleblob = vision.BlobAnalysis( ...
                    'CentroidOutputPort', true, ...
                    'AreaOutputPort', true, ...
                    'BoundingBoxOutputPort', false, ...
                    'MajorAxisLengthOutputPort',false, ...
                    'MinorAxisLengthOutputPort',false, ...
                    'OrientationOutputPort',false, ...
                    'ExtentOutputPort', false, ...
                    'OutputDataType', 'single', ...
                    'MinimumBlobArea', 100);
        
        videoFWriter = vision.VideoFileWriter('C:\active\BeesTrack.mp4',...
                      'FileFormat', 'MPEG4',...
                      'FrameRate',objreader.info.VideoFrameRate);%'FileFormat','WMV'
                  
        % load the pollen detection Faster RCNN network model
        load FrRCNN_Pollen1996imagestraining5 
        
% Initialize Tracks *******************************************************************************
% The structure contains the following fields:
% create an empty array of bee tracks
        tracks = struct(...
            'id', {}, ...   % the integer ID of the tracked bee
            'bbox', {}, ... % prediction bbox or detectin bbox
            'corbbox',{},...% correction bbox
            'pollenbboxes',{},... % pollen detection bbox
            'pollenscores',{},... % pollen detection score
            'kalmanFilter', {}, ... % Kalman filter object
            'prevblob',{},... % Previous fram blob of a bee for HT
            'blobedge',{},... % The edge axis of the previous blob
            'positions',{},...% Bee locations through frame sequence
            'colors',{},... % colour for bee flying trajectory
            'mergedcount',{},...% number of a bee detected as merged
            'pollenflag',{},...% number of frames detect pollen
            'age', {}, ... % tthe number of frames since the bee was first detected
            'singleNum',{}, ...% number of frames bee detected as single
            'totalVisibleCount', {}, ... % the total number of frames in which the bee was detected (visible)
            'consecutiveInvisibleCount', {});% the number of consecutive frames for 
                                             % which the bee was not detected (invisible).
% create object for display prediction bounding box        
        prediction = struct('id', {},'bbox', {},'center',{});
% create objcet for Kalman filter measurement        
        Detection = struct('id', {},'detcen', {});
% create objcet for mergdetection display and operation
        mergedetection = []; mergeDInd = 1;

nextId = 1; % ID of the next track
merging = [];% assgined matrix for merging detection assign to prediction
trackbeenum = 5;% choose a bee to get the Kalman parameters
blobedgeaxis = cell(1,15000);% edge axis for a blob before the merged
centroidsinglemerge = cell(1,15000);% single bee axis with origin of center of blob
Houghdetcopy = cell(1,15000);getHoughdet = [0,0,0];% collect detection of hough transform in merged situation
mergeddetcopy = cell(1,15000);
predictioncopy = cell(1,15000);
detectioncopy = cell(1,15000);
pollencount = 0;% count the video detect pollen
pollenInside = 0;% count pollen pass the boundary which means go into hive
beeinside = 0;% count bee pass the boundary

Framecount = 1;
folderRoute = 'c:\active\';
image_name1 = 'frame.png';    
image_name2 = 'mask.png'; 

% detect moving objects, and track them across video frames
while ~isDone(objreader)
    frame = step(objreader);% Read the next video frame
% Detect Objects ***********************************************************************************
% The |detectObjects| function returns the centroids and the bounding boxes
% of the detected objects. It also returns the binary mask, which has the 
% same size as the input frame. Pixels with a value of 1 correspond to the
% foreground, and pixels with a value of 0 correspond to the background.
if Framecount >1
   % detect foreground
   mask_fg0 = step(objdetector,frame);
        
   % apply morphological operations to remove noise and fill in holes
   mask_fg1 = imopen(mask_fg0, strel('rectangle', [3,3]));
   mask_fg2 = imclose(mask_fg1, strel('rectangle', [15, 15])); 
   mask_fg = imfill(mask_fg2, 'holes');
        
   % detect the colour
%    beehsv = step(objhhsv,frame);
   beehsv = rgb2hsv(frame);
   Hue = beehsv(:,:,1);
   Sat = beehsv(:,:,2);
   Val = beehsv(:,:,3);
   %orange color Hue = 0.072,0.0303~0.11;;light yellow Hue = 0.11~0.19
   ThrHuemin = 0.02; ThrHuemax = 0.19;ThrSat = 0.12;
   maskSat = (Sat>ThrSat);%&(Sat<ThrSatmax);
   maskHue = (ThrHuemin <= Hue) & (Hue <= ThrHuemax);
   maskorange = maskSat & maskHue; % Orange colour
   maskorange = imclose(maskorange, strel('rectangle', [3,3]));
   
     % The black colour. Only the value can limit the black colour. the
   % saturation is S = (max(R,G,B)-min(R,G,B))/max(R,G,B)
   ThrValue3 = 0.5;
   maskblack = Val < ThrValue3; % all of the black colour
   maskblack = imclose(maskblack, strel('rectangle', [3,3]));
      
   % white colour detection, but think about some bees' wings also reflect
   % white brightness colour.
   ThrValue4 = 0.5; ThrSat3 = 0.4; ThrHue = 0.23;
   maskwhite1 =  Val >= ThrValue4;
   maskwhite2 =  Sat <= ThrSat3;
   maskwhite3 = Hue <= ThrHue;
   makswhite = maskwhite1 & maskwhite2 & maskwhite3;
   maskwhite = imfill(makswhite, 'holes');
   
   % combine the colours
   maksOrgWht = maskorange | maskwhite;
   maskOrgBlkWht = maskorange | maskblack | maskwhite;
   
   % combine the foreground and the  Orange colour detector
   maskAnd = maksOrgWht & mask_fg; %Clear the shadow (which almost black colour)
   maskAnd = imopen(maskAnd, strel('rectangle', [3,3]));
   maskAndcopy = maskAnd;
   maskAnd = imdilate(maskAnd, strel('disk',35,8)); %Do dilation for bee position
   
   % Combine the dilation and colour, motion
   % using motion can get more bigger blob, but get more wings info which
   % can make the blob big. It leads to the main body ellipse too fat too get main body
   % only use colour can get more detial of shape, but may lost dark pollen part
   maskbees = maskAnd & maskOrgBlkWht;
   maskbees = imclose(maskbees, strel('rectangle', [3,3]));% smooth the binary image
   maskbees = imopen(maskbees, strel('rectangle', [3,3])); % smooth the binary image
   mask = imfill(maskbees, 'holes'); % make area entity
   maskoriginal = mask; % save the original mask image for later pollen detection
   
   % cut the blob image with a boundry
   StopDetectRow = size(frame,1)*boundary;
   mask(round(StopDetectRow) : size(frame,1),:) = 0;
   
   % perform blob analysis to find connected components
   [centroids, bboxes, orients] = step(objblobAnalyser,mask);
   copybboxes = bboxes;
   
% Predict New Locations of Existing Tracks ***************************************************************
% Use the Kalman filter to predict the centroid of each track in the
% current frame, and update its bounding box accordingly.  
    for i = 1:length(tracks)
            bbox = tracks(i).bbox;
            
            % predict the current location of the track
%             if tracks(i).mergedcount == 0
               [tracks(i).kalmanFilter.preS,tracks(i).kalmanFilter.PreCov] = predictme(tracks(i).kalmanFilter);
            
            %**********************************************************
            % shift the bounding box so that its center is at 
            % the predicted location
            predictedConer = int32(tracks(i).kalmanFilter.preS') - bbox(3:4) / 2;
            tracks(i).bbox = [predictedConer, bbox(3:4)];
            % store the prediction
            prediction(i).id = tracks(i).id;
            prediction(i).bbox = tracks(i).bbox;
            prediction(i).center = tracks(i).kalmanFilter.preS;% recording the predicted position
            
    end
    % delete the tracks which out of the detection boundary
%     OutOfDetectDoundaryInd = [];
%     for i = 1:length(tracks)
%        OutOfDetectDoundaryInd(i) = (tracks(i).bbox(1,2)+tracks(i).bbox(1,4))>=StopDetectRow;
%     end
%     %if ~isempty(OutOfDetectDoundaryInd)
%         tracks = tracks(~OutOfDetectDoundaryInd);
%         prediction = prediction(~OutOfDetectDoundaryInd);
        
    %end
% Assign Detections to Tracks **************************************************************************
% Step 1: Compute the cost of assigning every detection to each track using
% the |distance| method of the |vision.KalmanFilter| System object. The 
% cost takes into account the Euclidean distance between the predicted
% centroid of the track and the centroid of the detection. It also includes
% the confidence of the prediction, which is maintained by the Kalman
% filter. The results are stored in an MxN matrix, where M is the number of
% tracks, and N is the number of detections.   
%
% Step 2: Solve the assignment problem represented by the cost matrix using
% the |assignDetectionsToTracks| function. The function takes the cost 
% matrix and the cost of not assigning any detections to a track. 
   nTracks = length(tracks);
   nDetections = size(centroids, 1);
        
   % compute the cost of assigning each detection to each track
   cost = zeros(nTracks, nDetections);
%    KalmanStates = [];
%    Acccentroids = []; Accind = 1;
if nDetections ~= 0
   for i = 1:nTracks
       
        cost(i, :) = distanceme(tracks(i).kalmanFilter.preS, centroids);

   end
end

   % solve the assignment problem
   costOfNonAssignment = 100;
   [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
           
% % fix the unassigned tracks which is in the merged bees detection *********************************
   numAssTracks = size(assignments, 1);
   merging = zeros(20,3); mergecount = 1; 
   assignidx = []; unassignedidx=[];
   maxcol = size(frame,2); maxrow = StopDetectRow;
   unassdeletecount = 1; assdeletecount = 1;%record the unassignedTracks index
   for i = 1:length(unassignedTracks)
          ind = unassignedTracks(i);
          unassbbox = tracks(ind).bbox;
          uncentroid = [tracks(ind).kalmanFilter.preS(1,1),tracks(ind).kalmanFilter.preS(2,1)];
          unassx1 = unassbbox(1,1); unassx2 = unassbbox(1,1)+ unassbbox(1,3);
          unassy1 = unassbbox(1,2); unassy2 = unassbbox(1,2)+ unassbbox(1,4);
          predictmistake = 0;% this may not nessecery
          disy = 0; disx = 0;          
          assignedmentsInd = 0;
          % for unassigned prediction cross more than one detection bbox 
          predictmistakex = 0;
          predictmistakey = 0;
          anotherAss = zeros(1,10); onemore = 1; 
          anotherdisy = zeros(1,8);onemorey = 1;anotherdisx = zeros(1,8);onemorex = 1;
         % check the unassigned prediction pass the boundary or not
         halfunassx = (unassx1 + unassx2)/2; halfunassy = (unassy1 + unassy2)/2;
       if (halfunassx >= 1) && (halfunassx < maxcol) && (halfunassy >= 1) && (halfunassy < maxrow)
          for n = 1:numAssTracks
                  detectIdx = assignments(n, 2);
                  Assbbox = bboxes(detectIdx,:);
                  Assx1 = Assbbox(1,1); Assx2 = Assbbox(1,1)+Assbbox(1,3);
                  Assy1 = Assbbox(1,2); Assy2 = Assbbox(1,2)+Assbbox(1,4);
                  onemoreassinged = 0;
                  anotherdisy(onemorey) = 0;
                  anotherdisx(onemorex) = 0;
                  % if unassigned bbox is in the top of assigned box, make
                  % the unassigned bbox go down. The top is same as
                  % assigned. To make sure this unssgined box cross the
                  % assigned box, check the x1 or x2 of unssigned box
                  % inside the assigned box.
                  if (Assy1>unassy1)&&(Assy1<unassy2)&&(~((unassx2<Assx1)||(unassx1>Assx2)))
%                   if (Assy1>unassy1)&&(Assy1<unassy2)&&(((unassx1>Assx1)&&(unassx1<Assx2))||((unassx2>Assx1)&&(unassx2<Assx2)))                    
                     if predictmistake == 1 
                         anotherdisy(onemorey) = Assy1 - unassy1;
                         onemoreassinged = n;
                         onemorey = onemorey + 1;
                     else 
                         disy = Assy1 - unassy1;
                         predictmistakey = 1;
                         assignedmentsInd = n;
                     end
                  end
                  % if unassigned bbox is in the left of assigned box, make
                  % the unassigned bbox go right. The left is same as
                  % assigned. To make sure this unssgined box cross the
                  % assigned box, check the y1 or y2 of unssigned box
                  % inside the assigned box.
                  if (Assx1>unassx1)&&(Assx1<unassx2)&&(~((unassy2<Assy1)||(unassy1>Assy2)))
%                   if (Assx1>unassx1)&&(Assx1<unassx2)&&(((unassy1>Assy1)&&(unassy1<Assy2))||((unassy2>Assy1)&&(unassy2<Assy2)))        
                     if predictmistake == 1 
                         anotherdisx(onemorex) = Assx1 - unassx1;
                         onemoreassinged = n;
                         onemorex = onemorex + 1;
                     else
                         disx = Assx1 - unassx1;
                         predictmistakex = 1;
                         assignedmentsInd = n;
                     end
                  end
                  % if unassigned bbox is in the top of assigned box, make
                  % the unassigned bbox go down. The top is same as
                  % assigned. To make sure this unssgined box cross the
                  % assigned box, check the x1 or x2 of unssigned box
                  % inside the assigned box.
                  if (Assx2>unassx1)&&(Assx2<unassx2)&&(~((unassy2<Assy1)||(unassy1>Assy2)))
%                   if (Assx2>unassx1)&&(Assx2<unassx2)&&(((unassy1>Assy1)&&(unassy1<Assy2))||((unassy2>Assy1)&&(unassy2<Assy2)))                     
                      if predictmistake == 1 
                         anotherdisx(onemorex) = Assx2 - unassx2;
                         onemoreassinged = n;
                         onemorex = onemorex + 1;
                      else
                          disx = Assx2 - unassx2;
                          predictmistakex = 1;
                          assignedmentsInd = n;
                      end
                  end
                  % if unassigned bbox is in the bottom of assigned box,make
                  % the unassigned bbox go up. The bottom is same as
                  % assigned. To make sure this unssgined box cross the
                  % assigned box, check the x1 or x2 of unssigned box
                  % inside the assigned box.
                  if (Assy2>unassy1)&&(Assy2<unassy2)&&(~((unassx2<Assx1)||(unassx1>Assx2)))
                      if predictmistake == 1 
                         anotherdisy(onemorey) = Assy2 - unassy2;
                         onemoreassinged = n;
                         onemorey = onemorey + 1;
                      else
                          disy = Assy2 - unassy2;
                          predictmistakey = 1;
                          assignedmentsInd = n;
                      end
                  end
                  % if unassigned bbox is inside the assigned box, left the
                  % unassigned bbox, but move the other predicted bbox
                  if (unassy1 >= Assy1)&&(unassy2 <= Assy2)&&(unassx1 >=  Assx1)&&(unassx2 <= Assx2)
                      disy = 0;disx = 0;
                      predictmistake = 1;
                      assignedmentsInd = n;
                  end
                  % if unassigned has the merged situation, record it.
                  if  (predictmistakey == 1)||(predictmistakex == 1)
                      predictmistake = 1;
                  end
                  % record all of the merged unssigned predictions
                  if onemoreassinged ~= 0
                      anotherAss(onemore) = onemoreassinged;
                      onemore = onemore + 1;
                  end
          end 
       end
          % if the unassigned predcition cross more than one detection,
           % decide the shortest shift distance
           if sum(anotherAss)~=0 % check the one more assigned empty or not
               distance1 = sqrt(double(disx^2+disy^2));
               for n = 1;size(anotherAss,2)
                   distance2 = sqrt(double(anotherdisx(n)^2 + anotherdisy(n)^2));
                   if distance2 < distance1
                       distance1 = distance2;
                       disx = anotherdisx(n);
                       disy = anotherdisy(n);
                       assignedmentsInd = anotherAss(n);
                   end
               end
           end
           % the unassigned prediction is changed to merged, and record the fixed prediction
           if predictmistake == 1
               mcentroid =[uncentroid(1,1)+disx,uncentroid(1,2)+disy]; % modify the centroid
               unassbbox(1,1) = unassbbox(1,1)+disx;
               unassbbox(1,2) = unassbbox(1,2)+disy;
               % Add the fixed measure bounding box to the detection
               cenind = size(centroids,1); bboxind = size(bboxes,1);
               centroids(cenind+1,:) = mcentroid;% assumed detection / fix detection
               bboxes(bboxind+1,:) = unassbbox;% assumed detection / fix detection
               % collect the merged index
               orig_detectInd = assignments(assignedmentsInd, 2);% merged detection index
               merging(mergecount,:) = [ind,cenind+1,orig_detectInd];
               mergecount = mergecount + 1;  
               % reccord the unassigments tracking indexes              
               unassignedidx(unassdeletecount) = i;
               unassdeletecount = unassdeletecount + 1;
               % reccord the assigmented index to deal with or delete            
               assignidx(assdeletecount) = assignedmentsInd;
               assdeletecount = assdeletecount + 1;
           end
           
   end

   % delete the assigments and the unassigments
   if ~isempty(assignidx)
       assidx = unique(assignidx);
       % modify the assigments measurements
       for j = 1:length(assidx)
               assindx = assidx(j);
               detectIndx = assignments(assindx, 2);
               trackIndx = assignments(assindx, 1);
               disy2 = 0; disx2 = 0;
               % calculated the fixed prediction of assigned
               Dcentroid = centroids(detectIndx,:);% detcet centroid
               Dbbox = bboxes(detectIndx,:); % detect bounding box
               Dx1 = Dbbox(1,1); Dx2 = Dbbox(1,1)+Dbbox(1,3);
               Dy1 = Dbbox(1,2); Dy2 = Dbbox(1,2)+Dbbox(1,4);
               
               Pcentroid = [tracks(trackIndx).kalmanFilter.preS(1,1),tracks(trackIndx).kalmanFilter.preS(2,1)];% predicted centroid
               Pbbox = tracks(trackIndx).bbox; % predicted bounding box
               Px1 = Pbbox(1,1); Px2 = Pbbox(1,1)+Pbbox(1,3);
               Py1 = Pbbox(1,2); Py2 = Pbbox(1,2)+Pbbox(1,4);
               
               if (Dy1>Py1)&&(Dy1<Py2)
                      disy2 = Dy1 - Py1;                     
               end                    
               if (Dx1>Px1)&&(Dx1<Px2)
                      disx2 = Dx1 - Px1;
               end                  
               if (Dx2>Px1)&&(Dx2<Px2)
                      disx2 = Dx2 - Px2;
               end                 
               if (Dy2>Py1)&&(Dy2<Py2)
                      disy2 = Dy2 - Py2;
               end
               mcentroid2 =[Pcentroid(1,1)+disx2,Pcentroid(1,2)+disy2]; % modify the centroid
               Pbbox(1,1) = Pbbox(1,1)+disx2;
               Pbbox(1,2) = Pbbox(1,2)+disy2;
               % reccord the merge detection
               mergedetection(mergeDInd,:) = bboxes(detectIndx,:);
               mergeDInd = mergeDInd + 1;
               % change the merged detection to the assigned assumed detection
               centroids(detectIndx,:) = mcentroid2;
               bboxes(detectIndx,:) = Pbbox;
               % collect the merged index
               merging(mergecount,:) = [trackIndx,detectIndx,detectIndx];
               mergecount = mergecount + 1;
               
       end        
       assignments(assidx,:) = [];      
   end
   if ~isempty(unassignedidx)
       unassignedTracks(unassignedidx,:) = [];
   end
   
% % end of modification ****************************************

% Update Assigned Tracks ***********************************************************************
    numAssignedTracks = size(assignments, 1);
    for i = 1:numAssignedTracks
         trackIdx = assignments(i, 1);
         detectionIdx = assignments(i, 2);
         centroid = centroids(detectionIdx, :);
         bbox = bboxes(detectionIdx, :);
         
         % chenge R back
          tracks(trackIdx).kalmanFilter.R = [13,0;0,10]; 
          tracks(trackIdx).kalmanFilter.Q0 = [124,0;0,78]; 
%          if tracks(trackIdx).mergedcount ~= 0
%              tracks(trackIdx).kalmanFilter.PreCov = tracks(trackIdx).kalmanFilter.PreCov + tracks(trackIdx).kalmanFilter.Q0;
%              tracks(trackIdx).mergedcount = 0;
%          end
         % correct the estimate of the object's location
         % using the new detection
         tracks(trackIdx).kalmanFilter.M = centroid';
         [tracks(trackIdx).kalmanFilter.K,corS,corCov] = correctme(tracks(trackIdx).kalmanFilter);
         % corS is the current correct position, position from last kalman correct is follow after
         tracks(trackIdx).kalmanFilter.CorS = cat(1,corS,tracks(trackIdx).kalmanFilter.CorS(1:2));
         tracks(trackIdx).kalmanFilter.CorCov(3:4,3:4) = tracks(trackIdx).kalmanFilter.CorCov(1:2,1:2);
         tracks(trackIdx).kalmanFilter.CorCov(1:2,1:2) = corCov;
         
         % find the blob to crop and store for next detection
         tracks(trackIdx).prevblob = imcrop(mask,bbox);
         
         % replace predicted bounding box with detected
         % bounding box
         tracks(trackIdx).bbox = bbox;
         correctConer = int32(tracks(trackIdx).kalmanFilter.CorS(1:2)') - bbox(3:4) / 2;
         tracks(trackIdx).corbbox = [correctConer, bbox(3:4)];
         % update non-merged falg
         tracks(trackIdx).blobedge = 0;
         tracks(trackIdx).mergedcount = 0;
         % update track's age
         tracks(trackIdx).age = tracks(trackIdx).age + 1;
            % check the bee is not in boundary 
         bx1 = tracks(trackIdx).bbox(1);bx2 = tracks(trackIdx).bbox(1)+ tracks(trackIdx).bbox(3)-1;
         by1 = tracks(trackIdx).bbox(2);by2 = tracks(trackIdx).bbox(2)+ tracks(trackIdx).bbox(4)-1;
         % if the bee is not on the boundary, update the single frame No..
         if (bx1>2)&&(bx2<size(frame,2)-2)&&(by1>2)&&(by2<StopDetectRow-2)
            tracks(trackIdx).singleNum = tracks(trackIdx).singleNum + 1;
         end
         % update the positions
         addindex = tracks(trackIdx).age*2-1;
         tracks(trackIdx).positions(addindex:addindex+1) = centroid;   
         % update visibility
         tracks(trackIdx).totalVisibleCount = ...
                  tracks(trackIdx).totalVisibleCount + 1;
         tracks(trackIdx).consecutiveInvisibleCount = 0;
         
         Detection(trackIdx).id = tracks(trackIdx).id;
         Detection(trackIdx).detcen = centroid;
    end
% Update merged tracks ************************************************************************
   single_shape_aixs = cell(1,15); separate_ind = 1; rotation = []; markbeecopy=[];PeakValue = [];
   trackind = merging(1,1); % check the merging assigned matrix not empty
   if trackind ~= 0
     numMergedTracks = size(merging, 1);
     for i = 1:numMergedTracks
         trackind = merging(i,1);
         if trackind ~= 0
             detectind = merging(i,2);
             cropind = merging(i,3);
             bboxcrop = copybboxes(cropind, :);
             centroidm = centroids(detectind, :);
             bboxm = bboxes(detectind, :);
             
             % using the single blob to detect the *****************************
             if tracks(trackind).blobedge == 0 % if this is the first merge
                 %centroidsingle = round(centroidsingle);
                 % give two line external the corp blob image, for getting more edge
                 % pixels
                 blobsingle = tracks(trackind).prevblob;
                 %blobsingle2 = repmat(0,[size(blobsingle,1)+4 size(blobsingle,2)+4 1]);
                 blobsingle2 = zeros(size(blobsingle,1)+4, size(blobsingle,2)+4);
                 for r = 3:size(blobsingle,1)+2
                     for c = 3:size(blobsingle,2)+2
                         blobsingle2(r,c) = blobsingle(r-2,c-2);
                     end
                 end
                 edgesingle = edge(blobsingle2,'Canny'); % get the edge image
                 % get the center of the blob
                 [areasingle,centroidsingle2] = step(singleblob,logical(blobsingle2));
                 release(singleblob);
                 % if it get two blobs with two centres, choose the big blob's centre.
                 if size(centroidsingle2,1) >= 2
                     [M,I] = max(areasingle);
                     centroidsingle = centroidsingle2(I,:);
                 else
                     centroidsingle = centroidsingle2;
                 end
                 % transform the axis to center axis
                 rowbee1 = size(edgesingle,1); columnbee1 = size(edgesingle,2);% y is row, x is column
                 lengthaxis = sum(sum(edgesingle)); % obtain the edge curve pexils sum
                 centeraxis = zeros(lengthaxis,2);num = 1;
                 for r = 1:rowbee1
                     for c = 1:columnbee1
                         if edgesingle(r,c) == 1
                             centeraxis(num,1) = c - centroidsingle(1);% x axis
                             centeraxis(num,2) = r - centroidsingle(2);% y axis
                             num = num+1;
                         end
                     end
                 end
             else % if this is not the second merged
                 centeraxis = tracks(trackind).blobedge;
             end
             % draw the single bee shape on the merged shape image, the single bee
             % center follow the merged shape's edge. y is row, x is column
             centeraxis = round(centeraxis);
             if tracks(trackind).id == trackbeenum
                 blobedgeaxis{Framecount+1}= centeraxis;
                 centroidsinglemerge{Framecount+1} = centroidsingle;
             end
             mergedblob = imcrop(mask,bboxcrop);
             edgemerged = edge(mergedblob,'Canny');
             rowmerge = size(edgemerged,1); columnmerge = size(edgemerged,2);
             maxrotation = 21;
             markbee = zeros(rowmerge,columnmerge,maxrotation); degree = -10; centeraxis_d = [];
             centeraxis_d2 = cell(maxrotation,1);
             for d = 1:maxrotation
                 radian = degree/180*pi;
                 centeraxis_d(:,1) = centeraxis(:,1).*cos(radian)+centeraxis(:,2).*sin(radian); % x axis
                 centeraxis_d(:,2) = centeraxis(:,2).*cos(radian)-centeraxis(:,1).*sin(radian); % y axis
                 centeraxis_d2{d} = centeraxis_d;
                 markbee2d = zeros(rowmerge,columnmerge);
                 for r = 1:rowmerge
                     for c = 1:columnmerge
                         if edgemerged(r,c) == 1 % find the edge point
                             % find the single bee shape axis
                             for n = 1:size(centeraxis_d,1)
                                 xorg = round(centeraxis_d(n,1)+c); % x axis
                                 yorg = round(centeraxis_d(n,2)+r); % y axis
                                 if (xorg > 0) && (yorg > 0) && (xorg <= columnmerge) && (yorg <= rowmerge)
                                     % if the match pixel outside the merged blob, leave it as 0.
                                     insideflag = mergedblob(yorg,xorg);
                                     if insideflag == 1
                                         markbee2d(yorg,xorg) = markbee2d(yorg,xorg) + 1;
                                     end
                                 end
                             end
                         end
                     end
                 end
                 markbee(:,:,d) = markbee2d;
                 degree = degree + 1;
             end
             wrongdet = 0; % distance between detection and prediction in the merged situation
             if sum(sum(sum(markbee)))~=0
                 % find the highest mark of pixels, which could be the center of the single
                 % bee in the merged blob
                 rowmark = size(markbee,1); columnmark = size(markbee,2);
                 x_peak21 = zeros(maxrotation,1); y_peak21 = zeros(maxrotation,1);
                 Val_peaks = zeros(maxrotation,1); % column is j and x, row is i and y, orientation is a.
                 for d = 1:maxrotation % d is rotation
                     for r = 1:rowmark
                         for c = 1:columnmark
                             if markbee(r,c,d) > Val_peaks(d)
                                 Val_peaks(d) = markbee(r,c,d);
                                 x_peak21(d) = c; y_peak21(d) = r;
                                 % d_peak = d;
                             end
                         end
                     end
                 end
                 [Val_peaksmax,d_peak] = max(Val_peaks);% calculate the max peak value
                 % use the sum of area to get the fitting rotation a.
                 sumVal = 0;d_peak = 0;%d_p21 = 0;
                 for d = 1:maxrotation % d is rotation
                     % peakarea = zeros(21, 21);
                     x_p = x_peak21(d);y_p = y_peak21(d);
                     if (x_p -5) < 1
                         x_pleft = 1;
                     else
                         x_pleft = x_p -5;
                     end
                     if (x_p +5) > columnmark
                         x_pright = columnmark;
                     else
                         x_pright = x_p +5;
                     end
                     if (y_p - 5) < 1
                         y_pleft = 1;
                     else
                         y_pleft = y_p - 5;
                     end
                     if (y_p +5) > rowmark
                         y_pright = rowmark;
                     else
                         y_pright = y_p +5;
                     end
                     peakarea = markbee(y_pleft:y_pright,x_pleft:x_pright,d);
                     %d_p21(d) = sum(sum(peakarea));
                     if (sum(sum(peakarea)) > sumVal) && (Val_peaks(d) >= (Val_peaksmax-5))
                         sumVal = sum(sum(peakarea));
                         d_peak = d;
                     end
                 end
                 % the final bee postion without weight calculation
                 
                 % calculate the weight position *********************
                 % the final position reference
                 x_peakref = x_peak21(d_peak); y_peakref = y_peak21(d_peak); Warea = 20;
                 if (x_peakref -Warea) < 1
                     x_peakleft = 1;
                 else
                     x_peakleft = x_peakref -Warea;
                 end
                 if (x_peakref +Warea) > columnmark
                     x_peakright = columnmark;
                 else
                     x_peakright = x_peakref +Warea;
                 end
                 if (y_peakref -Warea) < 1
                     y_peakleft = 1;
                 else
                     y_peakleft = y_peakref -Warea;
                 end
                 if (y_peakref +Warea) > rowmark
                     y_peakright = rowmark;
                 else
                     y_peakright = y_peakref +Warea;
                 end
                 % weight of x and y
                 x_sum = 0;
                 for x = x_peakleft:x_peakright % x is column, y is row
                     for y = y_peakleft:y_peakright
                         x_sum = x_sum + x*markbee(y,x,d_peak);
                     end
                 end
                 x_weight = x_sum / sum(sum(markbee(y_peakleft:y_peakright,x_peakleft:x_peakright,d_peak)));
                 y_sum = 0;
                 for x = x_peakleft:x_peakright % x is column, y is row
                     for y = y_peakleft:y_peakright
                         y_sum = y_sum + y*markbee(y,x,d_peak);
                     end
                 end
                 y_weight = y_sum / sum(sum(markbee(y_peakleft:y_peakright,x_peakleft:x_peakright,d_peak)));
                 % weight calculation finish ************************
                 x_peak = round(x_weight); y_peak = round(y_weight);
                 % the final position
                 %x_peak = x_peak21(d_peak); y_peak = y_peak21(d_peak);
                 orig_axis = [x_peak+ bboxcrop(1),y_peak+ bboxcrop(2)];
                 % rotate the center reference axis to the fitting orientation
                 degree = d_peak - 11;radian = degree/180*pi; centeraxis_d = [];
                 centeraxis_d(:,1) = centeraxis(:,1).*cos(radian)+centeraxis(:,2).*sin(radian);% x
                 centeraxis_d(:,2) = centeraxis(:,2).*cos(radian)-centeraxis(:,1).*sin(radian);% y
                 % get the single bee shape axis in the whole frame
                 rowframe = size(frame,1); columnframe = size(frame,2); length_d = size(centeraxis_d,1);
                 insertvector = zeros(length_d,2);
                 for n = 1:length_d
                     xorg = int32(centeraxis_d(n,1)+orig_axis(1)); % x axis, column
                     yorg = int32(centeraxis_d(n,2)+orig_axis(2)); % y axis, row
                     if (xorg >= 1) && (yorg >= 1) && (xorg <= columnframe) && (yorg <= rowframe)
                         insertvector(n,1:2) = [xorg yorg];
                     end
                 end
                 % for debugging
                 %          markbeecopy{separate_ind}=markbee;% record all of merged bee in this frame
                 %          PeakValue{separate_ind} = Val_peaks; % record the peak value
                 %          rotation(separate_ind,1) = d_peak; rotation(separate_ind,2) = tracks(trackind).id;% record the fitting orient and bee's id
                 %get the fitting rotation of new edge axis of centre origin
                 tracks(trackind).blobedge = centeraxis_d;
                 single_shape_aixs{separate_ind} = insertvector; % for final display
                 getHoughdet(separate_ind,1) = tracks(trackind).id;% get the merged detection
                 getHoughdet(separate_ind,2:3) = orig_axis;
                 separate_ind = separate_ind + 1;
                 % finish separation ***********************************************
                 
                 % Kalman filter tracking bees ************************************
                 
                 % correct the estimate of the object's location using the new detection
                 Houghdet = double([orig_axis(1);orig_axis(2)]);
                 wrongdet = distanceme(tracks(trackind).kalmanFilter.preS,Houghdet');
%                  if wrongdet > 80
%                  % use the previous Kalman correction and the prediction to get the
%                  % measurement
%                 tracks(trackind).kalmanFilter.M = tracks(trackind).kalmanFilter.preS;% (tracks(trackind).kalmanFilter.CorS(1:2)).*2-tracks(trackind).kalmanFilter.preS;
%                 else
                 tracks(trackind).kalmanFilter.M = Houghdet;% take the Hough detection as measurement
%                 end
                 
             else
                 tracks(trackind).kalmanFilter.M = tracks(trackind).kalmanFilter.preS;
             end
             % change the measurement noise R
             tracks(trackind).kalmanFilter.R = [416,0;0,713];%500
             % if it is the first merge, change the P (covariance of prediction)
             if tracks(trackind).mergedcount == 0
                 % change Q0 from single tracking to merged tracking
                 tracks(trackind).kalmanFilter.PreCov = tracks(trackind).kalmanFilter.PreCov - tracks(trackind).kalmanFilter.Q0 + [435,0;0,825];
             end
             tracks(trackind).kalmanFilter.Q0 = [435,0;0,825];
             [tracks(trackind).kalmanFilter.K,corS,corCov] = correctme(tracks(trackind).kalmanFilter);
             % corS is the current correct position, position from last kalman correct is follow after
             tracks(trackind).kalmanFilter.CorS = cat(1,corS,tracks(trackind).kalmanFilter.CorS(1:2));
             tracks(trackind).kalmanFilter.CorCov(3:4,3:4) = tracks(trackind).kalmanFilter.CorCov(1:2,1:2);
             tracks(trackind).kalmanFilter.CorCov(1:2,1:2) = corCov;
             
             % replace correct bounding box
             correctConer = int32(tracks(trackind).kalmanFilter.CorS(1:2)') - tracks(trackind).bbox(3:4) / 2;
             tracks(trackind).corbbox = [correctConer, tracks(trackind).bbox(3:4)];
             
             % update merged tracks
             tracks(trackind).mergedcount = tracks(trackind).mergedcount + 1;
             tracks(trackind).age = tracks(trackind).age + 1;% update track's age
             % update the positions
             addindex = tracks(trackind).age*2-1;
             tracks(trackind).positions(addindex:addindex+1) = centroidm;
             % update visibility
             tracks(trackind).totalVisibleCount = ...
                 tracks(trackind).totalVisibleCount + 1;
             tracks(trackind).consecutiveInvisibleCount = 0;
             
         end
     end
   end
   
% Update Unassigned Tracks ********************************************************************
    for i = 1:length(unassignedTracks)
          ind = unassignedTracks(i);
          
          tracks(ind).age = tracks(ind).age + 1;
          % update the positions
          addindex = tracks(ind).age*2-1;
          tracks(ind).positions(addindex:addindex+1) = tracks(ind).positions(addindex-2:addindex-1);
          
          tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
    end
    
% Delete Lost Tracks **************************************************************************
        if ~isempty(tracks)
                  
        invisibleForTooLong = 1;
        ageThreshold = 2;
        
        % compute the fraction of the track's age for which it was visible
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % find the indices of 'lost' tracks
        % deletes tracks that have been invisible
        % for too many consecutive frames. It also deletes recently created tracks
        % that have been invisible for too many frames overall. 
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
        % check the bee has pollen, and count the pollen. if a bee has
        % pollen, count 1 bee has pollen. if a bee not only has pollen, but
        % also at the entracnce boundary, count 1 pollen go into the hive.
        chktracks = tracks(lostInds);
        pollenflag2 = 0;
        if ~isempty(chktracks)
           for i = 1:size(chktracks, 1)
               Probility = double(chktracks(i).pollenflag) / chktracks(i).singleNum;
               if (chktracks(i).singleNum > 2)&&(Probility > 0.46)% probabiblity threshold for decide the bee has pollen                   
                   pollencount = pollencount + 1; 
                   pollenflag2 = 1;
               end
                 % bottom of the bounding box just up to the counting boundary
               if (chktracks(i).corbbox(2)+chktracks(i).corbbox(4))>(StopDetectRow-10) % frame 2027, greater or smaller?                                   
                  if pollenflag2 == 1     
                     pollenInside = pollenInside +1; % bee with pollen go into the hive
                  else
                     beeinside = beeinside +1;% bee without pollen go into the hive 
                  end     
               end
              
               
           end
        end
        % delete lost tracks
        tracks = tracks(~lostInds);
        prediction = prediction(~lostInds);       
        end 
        
% Create New Tracks ******************************************************************************
        %Assume that any unassigned detection is a start of a new track.
        centroidsN = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        
        for i = 1:size(centroidsN, 1)
            
            centroid = centroidsN(i,:);
            doublecentroid = cat(2,centroid,centroid);
            bbox = bboxes(i, :);
            
            % create a Kalman filter object
            kalmanfilter = struct(...
                   'A', [2,0,-1,0;0,2,0,-1], ...% state transition model
                   'H', [1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1], ...% measurement model
                   'Q0',[124,0;0,78],...% constant velocity noise %150
                   'preS', [0,0]', ...% prediction states
                   'PreCov', 0, ...% prediction states coveriance
                   'M', centroid', ...% measurement of detection
                   'R',[13,0;0,10],...% measurement noise coveriance
                   'K', 0, ...
                   'CorS', doublecentroid',...% correction states
                   'CorCov',[10 0 0 0;0 10 0 0;0 0 10 0;0 0 0 10]);% correction states coveriance
               
            % find the blob to crop and store for next detection
            prevblob = imcrop(mask,bbox);
            % tracking lines color
            trackcolor = uint8(rand(1,3).*150);
            % create a new track
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'corbbox',bbox,...
                'pollenbboxes',[],...
                'pollenscores',[],...
                'kalmanFilter', kalmanfilter, ...                
                'prevblob',prevblob,...  
                'blobedge',0,...% the edge axis
                'positions',doublecentroid,...
                'colors',trackcolor,...    ·
                'mergedcount',0,...
                'pollenflag',0,...
                'age', 1, ...
                'singleNum',0, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            
            % check the single Number is 0 or 1, if the new bee is not
            % appeared in the boundary, the initial value is 1 rather than 0.
            bx1 = bbox(1);bx2 = bbox(1)+ bbox(3)-1;
            by1 = bbox(2);by2 = bbox(2)+ bbox(4)-1;
            % if the bee is not on the boundary, the single frame No. is 1.
            if (bx1>2)&&(bx2<size(frame,2)-2)&&(by1>2)&&(by2<StopDetectRow-2)
               newTrack.singleNum = 1;
            end
            
            % add it to the array of tracks
            tracks(end + 1) = newTrack;
            
            % increment the next id
            nextId = nextId + 1;
            
        end
        
% Detect bees which have pollen sacs************************************************************
if ~isempty(tracks)
    trackslength = length(tracks);
    pollencolour = [];
    for t = 1:trackslength
        % check the bee is not in boundary 
        bx1 = tracks(t).bbox(1);bx2 = tracks(t).bbox(1)+ tracks(t).bbox(3)-1;
        by1 = tracks(t).bbox(2);by2 = tracks(t).bbox(2)+ tracks(t).bbox(4)-1;
        tracks(t).pollenbboxes = [];
        tracks(t).pollenscores = [];
        % if the bee is single more than 2 frames ,dont merge,is not in the boudary, then
        % try detect pollen
        if (tracks(t).mergedcount == 0)&&(bx1>2)&&(bx2<size(frame,2)-2)&&(by1>2)&&(by2<StopDetectRow-2)
           pollenflag1 = 0;% clear the polleflag1 for the image pollen detection
           % crop out the bee and the blob image
           maxline = 40;
           cropbx1 = bx1 - maxline/2; cropbx2 = bx2 + maxline/2;
           cropby1 = by1 - maxline/2; cropby2 = by2 + maxline/2;
           if cropbx1 < 1
                cropbx1 = 1;
           end
           if cropby1 < 1
                cropby1 = 1;
           end
           if cropbx2 > size(frame,2)
                cropbx2 = size(frame,2);
           end
           if cropby2 > StopDetectRow
                cropby2 = StopDetectRow;
           end
           cropbee = imcrop(frame,[cropbx1,cropby1,(cropbx2-cropbx1+1),(cropby2-cropby1+1)]);
           imwrite(cropbee,'img_buffer.png');
           cropbee2 = imread('img_buffer.png');
           [plnbbox,plnscore,plnlabel] = detect(frrcnn,cropbee2,'Threshold',0.36);
           [plnbbox,plnscore,plnlabel] = selectStrongestBboxMulticlass(plnbbox,plnscore,plnlabel,'OverlapThreshold',0.1);           
           if ~isempty(plnbbox)
               pollenflag1 = 1;
               plnbbox2 = plnbbox;
               % Calcuate bbox from bee's image to video frame.
               plnbbox2(:,1) = int32(plnbbox(:,1))+cropbx1-1;
               plnbbox2(:,2) = int32(plnbbox(:,2))+cropby1-1;
               tracks(t).pollenbboxes = plnbbox2;
               tracks(t).pollenscores = plnscore;
           end
                                   
           % if the image has pollen, plus the pollen flag for the corresponding bee
           if pollenflag1 == 1
                      tracks(t).pollenflag = tracks(t).pollenflag +1;
           end
           
        end
        
    end
end
                
% Display Tracking Results *********************************************************************
% convert the frame and the mask to uint8 RGB 
        frame = im2uint8(frame);
        frame2 = frame;
        mask2 = mask;
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        
        minVisibleCount = 0;
        if ~isempty(tracks)
              
            % noisy detections tend to result in short-lived tracks
            % only display tracks that have been visible for more than 
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);
            
            % display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % get bounding boxes
                bboxes = cat(1, reliableTracks.bbox);
                pollenBboxes = cat(1, reliableTracks.pollenbboxes);
                pollenScores = cat(1, reliableTracks.pollenscores);
                % get ids
                ids = int32([reliableTracks(:).id]);
                
                % create labels for objects indicating the ones for 
                % which we display the predicted rather than the actual 
                % location
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);
                
                % create the prediction labels and predicted
                prebboxes = cat(1, prediction.bbox);
                preids = int32([prediction(:).id]);
                prelabels = cellstr(int2str(preids'));
                labPredicted = cell(size(prelabels));
                labPredicted(:) = {' predicted'};
                prelabels = strcat(prelabels, labPredicted);
                
                % create the correction labels and corrected
                corbboxes = cat(1, reliableTracks.corbbox);
                corlabels = cellstr(int2str(ids'));
                labCorrected = cell(size(corlabels));
                labCorrected(:) = {' corrected'};
                corlabels = strcat(corlabels, labCorrected);
                
                %create the merged detection labels and merged detectoin
                merglabels = cell(size(mergedetection,1),1);
                merglabels = merglabels';
                merglabels(:) = {'merged detect'};                
                                             
                % created the pollen flage and labels
                circle = []; ccount = 1;  pollenNum = [];% This record pollenNum/SingleNum              
                for p = 1:length(reliableTracks)
                    if reliableTracks(p).pollenflag >= 1
                        circle(ccount,:) = [reliableTracks(p).corbbox(1),reliableTracks(p).corbbox(2)+reliableTracks(p).corbbox(4),5];
                        pollenNum(ccount,:) = [int32(reliableTracks(p).pollenflag),int32(reliableTracks(p).singleNum)];
                        ccount = ccount + 1;
                    end
                end
                
                % draw on the frame
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);
                if ~isempty(prebboxes)
                  frame = insertObjectAnnotation(frame, 'rectangle', ...
                      prebboxes, prelabels, 'Color', 'red');
                end
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    corbboxes, corlabels, 'Color', 'cyan');
                if ~isempty( mergedetection)
                  frame = insertObjectAnnotation(frame, 'rectangle', ...
                      mergedetection, merglabels, 'Color', 'blue');
                end
                if ~isempty( pollenBboxes)
                  frame = insertObjectAnnotation(frame, 'rectangle', ...
                      pollenBboxes, pollenScores , 'Color', 'magenta');
                end
               
                % Insert the text of the pollen and pollen go into hive
                % Number of pollen apeared on the video
                frame = insertText(frame,[1,size(frame,1)*0.91],'Pollen Count:','FontSize',20,'BoxOpacity',0.2);
                frame = insertText(frame,[160,size(frame,1)*0.91],pollencount,'FontSize',20,'BoxOpacity',0.2);
                % Number of pollen sacs be brough back to bee hive
                frame = insertText(frame,[1,size(frame,1)*0.94],'pollenIntoHive:','FontSize',20,'BoxOpacity',0.2);
                frame = insertText(frame,[160,size(frame,1)*0.94],pollenInside,'FontSize',20,'BoxOpacity',0.2);
%                 frame = insertText(frame,[1,size(frame,1)*0.97],'BeesIntoHive:','FontSize',20,'BoxOpacity',0.2);
%                 frame = insertText(frame,[150,size(frame,1)*0.97],beeinside,'FontSize',20,'BoxOpacity',0.2);
                
                % display and lablel the pollen
                pollenlabel1 = [];pollenlabel2 = [];pollenlabels = [];
                if ~isempty(circle)
                    pollenlabel1 = cell(size(pollenNum,1),1);% the pollen numnber
                    pollenlabel1 = cellstr(int2str(pollenNum(:,1)));% the pollen numnber
                    pollenlabel1_2 = cell(size(pollenNum,1),1);% the single numnber
                    pollenlabel1_2 = cellstr(int2str(pollenNum(:,2)));% the single numnber
                    pollenlabel2 = cell(size(circle,1),1);
                    pollenlabel2(:) = {'(pol/sig)'};% pollen number / sinlge number
                    pollenlabel_slash = cell(size(circle,1),1);
                    pollenlabel_slash(:) = {'/'};
                    
                    pollenlabels = strcat(pollenlabel1,pollenlabel_slash,pollenlabel1_2,pollenlabel2);
                    frame = insertObjectAnnotation(frame, 'circle', ...
                    circle, pollenlabels, 'Color', 'magenta');
                    mask = insertObjectAnnotation(mask, 'circle', ...
                    circle, pollenlabels, 'Color','magenta');
                end

                % draw on the mask
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
                if ~isempty(prebboxes)
                  mask = insertObjectAnnotation(mask, 'rectangle', ...
                      prebboxes, prelabels, 'Color', 'red');
                end
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    corbboxes, corlabels, 'Color', 'cyan');
                if ~isempty( mergedetection)
                  mask = insertObjectAnnotation(mask, 'rectangle', ...
                      mergedetection, merglabels, 'Color', 'blue');
                end
                
                % clean the merged detection
                mergeddetcopy{Framecount+1} = mergedetection;
                mergedetection = [];               
                mergeDInd = 1;
                % draw the detection boundary
                %frame(int32(StopDetectRow)-1:int32(StopDetectRow),:,:) = 255;
                mask(int32(StopDetectRow)-1:int32(StopDetectRow),:) = 255;
                frame(int32(StopDetectRow)-2:int32(StopDetectRow)+2,:,1) = 255;
                frame(int32(StopDetectRow)-2:int32(StopDetectRow)+2,:,2) = 0;
                frame(int32(StopDetectRow)-2:int32(StopDetectRow)+2,:,3) = 255;
                
                
                % display the tracking curves
                tracksNum = size(reliableTracks,2);
                greatsize = 0;% record the greatest size
                posrecords = [];
                colrecords = [];
                % record all the positions and colours, find the greatest
                % size
                for i = 1:tracksNum
                  posrecord = reliableTracks(i).positions;
                  colrecord = reliableTracks(i).colors;
                  frame = insertShape(frame,'Line',posrecord,'Color',colrecord,'LineWidth',1);% for matlab 2015a, 'LineWidth' is 5 before
                end
            end
            
            % display the merged detection shape
            if ~isempty(single_shape_aixs{1})
                lengthsingle = separate_ind-1;
                for o = 1:lengthsingle
                    lengthaxis = length(single_shape_aixs{o});
                    for in = 1:lengthaxis
                        if  (single_shape_aixs{o}(in,2) ~= 0) && (single_shape_aixs{o}(in,1) ~= 0)
                            % x is column so is 2, y is row so it is 1
                            frame(single_shape_aixs{o}(in,2),single_shape_aixs{o}(in,1),1)= 0;
                            frame(single_shape_aixs{o}(in,2),single_shape_aixs{o}(in,1),2)= 255;
                            frame(single_shape_aixs{o}(in,2),single_shape_aixs{o}(in,1),3)= 0;
                            mask(single_shape_aixs{o}(in,2),single_shape_aixs{o}(in,1),1)= 0;
                            mask(single_shape_aixs{o}(in,2),single_shape_aixs{o}(in,1),2)= 255;
                            mask(single_shape_aixs{o}(in,2),single_shape_aixs{o}(in,1),3)= 0;
                        end
                    end
                end
            end
            
        end              
        
        % display the mask and the frame
        step(objmaskPlayer,mask);        
        step(objvideoPlayer,frame);
        step(videoFWriter,frame); 
    
    % get the position information
    Houghdetcopy{Framecount} = getHoughdet;
    getHoughdet = [];
    predictioncopy{Framecount} = prediction;
    detectioncopy{Framecount} = Detection; 
    
    Detection = [];
    % recording frames to a folder
    if Framecount<=100
        Framenumber = int2str(Framecount);           
        recordingfiles = strcat(folderRoute,Framenumber,image_name1);%folderRoute3
        imwrite(frame,recordingfiles); % store the colour image             
        recordingfiles = strcat(folderRoute,Framenumber,image_name2);
        imwrite(mask,recordingfiles); % store the binary image
   end
end
    if Framecount == 3000
        break;
    end
Framecount = Framecount +1;
end
release(videoFWriter);
display(pollencount);
display(pollenInside);
display(beeinside);