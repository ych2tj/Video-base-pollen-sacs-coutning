%% Train Faster R-CNN Pollen sacs Detector
%%
% Load Traning Data. 
load PollenSacsLabeler_training12Flip12_table

summary(PollenSacsTrainingData);

%%
% Build net works

net = vgg16; % lowest input size [32 32 3]
fasterRCNNLayers = net.Layers;
Input = imageInputLayer([32 32 3]);
fasterRCNNLayers(1) = Input;
fc6 = fullyConnectedLayer(4096);
fasterRCNNLayers(end-8) = fc6;
fc = fullyConnectedLayer(2);
fasterRCNNLayers(end-2) = fc;
cl = classificationLayer;
fasterRCNNLayers(end) = cl;

%%
% Set network training options:
%
% * Set the CheckpointPath to save detector checkpoints to a temporary
%   directory. Change this to another location if required.
tempdir = 'C:\MatlabTrainTemp\';
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 50, ...
    'VerboseFrequency', 200, ...
    'CheckpointPath', tempdir);

%%
% strat from checkpoint
% checkpoint = load('C:\MatlabDeepLearning\faster_rcnn_stage_4_checkpoint__4920__2018_06_25__17_00_26.mat');
% [frrcnn,info] = trainFasterRCNNObjectDetector(PollenSacsTrainingData, checkpoint.detector , options,...
%                 'PositiveOverlapRange',[0.4 1]);
% Train the Faster R-CNN detector. Training can take a few minutes to complete.
[frrcnn,info] = trainFasterRCNNObjectDetector(PollenSacsTrainingData, fasterRCNNLayers , options,...
                'PositiveOverlapRange',[0.5 1]);

%%
% show the trainging loss
figure;
for n = 1:4
    subplot(2,2,n);
    plot(info(n).TrainingLoss);
    xlabel('Iteration'); ylabel('Trainging Loss');
    if n == 1
        title('Train RPN');
    elseif n == 2
        title('Train Fast RCNN');
    elseif n == 3
        title('Retrain RPN');
    elseif n == 4
        title('Retrain Fast RCNN');
    end
end
%%
% show the trainging loss
figure;
for n = 1:4
    subplot(2,2,n);
    plot(info(n).TrainingAccuracy);
    xlabel('Iteration'); ylabel('Accuracy');
    if n == 1
        title('Train RPN');
    elseif n == 2
        title('Train Fast RCNN');
    elseif n == 3
        title('Retrain RPN');
    elseif n == 4
        title('Retrain Fast RCNN');
    end
end