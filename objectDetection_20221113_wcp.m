load('trained_models/gTruth_objdet_rutank_pruned.mat');
oldLeadPath = "D:\RU_UKR\russia\russia\Tanks";
newpath = fullfile("/downloads/Tanks/LabeledTanks");
badPaths = changeFilePaths(gTruth, {[oldLeadPath, newpath]})
%%
%Separate images from bounding boxes and set network input image size
[imds,blds] = objectDetectorTrainingData(gTruth);
inputSize = [227 227 3];
%%
% dataSource = gTruth.DataSource;
% labelDefs  = gTruth.LabelDefinitions;
% labelData  = gTruth.LabelData;
% labelDefs.Name = strrep(labelDefs.Name, labelDefs.Name, 'tank');
% varnames = labelData.Properties.VariableNames;
% labelData.Properties.VariableNames = strrep(labelData.Properties.VariableNames, varnames, 'tank');
% % gTruth = groundTruth(dataSource, labelDefs, labelData);
%%
%Shuffle and split data into 70/10/20 split
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 70% of the images for training.
numTrain = round(0.70 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 10% of the images for validation
numVal = round(0.10 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);
%%
% %replace all labels with "tank"
% replacedlabels = {};
% for i = 1:size(blds.LabelData,1)
%     cat = blds.LabelData{i,2};
%     for j = 1:size(cat)
%       cat(j) = "tank";
%     end
%     replacedLabels{i,1} = cat;
% end
% tbl = cell2table(blds.LabelData)
% tbl.Var2 = replacedLabels;
% tblds = boxLabelDatastore(tbl);
% %transblds = transform(blds,(@LabelData)replaceLabels(LabelData,'tank'));
% newgTruth = groundTruth()
%%
%Set box Label Train/Val/Test split
bldsTrain = blds.subset(trainingIdx);
bldsVal = blds.subset(valIdx);
bldsTest = blds.subset(testIdx);
%%
%Recombine box labels and images according to splits
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsVal,bldsVal);
testData = combine(imdsTest,bldsTest);
%%
%validateInputData(trainingData);
%validateInputData(validationData);
%validateInputData(testData);
%%
%Show what a bounding box looks like, usually skip
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,"Rectangle",bbox);
annotatedImage = imresize(annotatedImage,2);
%%
figure
imshow(annotatedImage)
reset(trainingData);
%%
%Get Class Names
%className = "tank";
classNames = gTruth.LabelDefinitions.Name;
numClasses = numel(classNames);
%%
%Display frequency, only one class so not necessary
tbl = countEachLabel(blds)
bar(tbl.Label,tbl.Count)
ylabel("Frequency")
%%
%JUST MAKES THINGS WORSE

%Balancing Box Label Classes
% cvBefore = std(tbl.Count)/mean(tbl.Count)
% numObservations = mean(tbl.Count) * numClasses;
% ThresholdValue = 0.5;
% 
% blockedImages = blockedImage(imds.Files,BlockSize=[50,50]);
% blockSize = [50,50];
%boxLabels = cell2table(blds.LabelData);
%%
% locationSet = balanceBoxLabels(boxLabels,blockedImages,blockSize,...
%         numObservations,'OverlapThreshold',ThresholdValue);
%%
% bldsBalanced = boxLabelDatastore(boxLabels,locationSet);
% balancedDatasetCount = countEachLabel(bldsBalanced);
%%
% hold on;
% balancedLabels = balancedDatasetCount.Label;
% balancedCount  = balancedDatasetCount.Count;
% h2 = histogram('Categories',balancedLabels,'BinCounts',balancedCount);
% % title(h2.Parent,"Balanced class labels (OverlapThreshold: " + ThresholdValue + ")" );
% legend(h2.Parent,{'Before','After'});
%%
% cvAfter = std(balancedCount)/mean(balancedCount)
%%
%Estimate Anchor boxes
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 6;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);

area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");
anchors = anchors(idx,:);
%%
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    };
% %%
% %Setup yoloV3 or yoloV2 detectors and input networks
% network = squeezenet; %resnet50();
% detector = yolov3ObjectDetector(network, classNames, anchorBoxes, 'DetectionNetworkSource', {'fire9-concat', 'fire5-concat'}, InputSize = inputSize);
% %featureLayer = "activation_40_relu";
% %lgraph = yolov2Layers(inputSize, numClasses, anchors, network, featureLayer);   
% %%
% %Yolo V4 detector setup
% %detector = yolov4ObjectDetector("tiny-yolov4-coco",classNames,anchorBoxes,InputSize=inputSize);
% %%
% %Augment Training Data
% augmentedTrainingData = transform(trainingData,@augmentData);
% %%
% %Visualize augmented data
% augmentedData = cell(4,1);
% for k = 1:4
%     data = read(augmentedTrainingData);
%     augmentedData{k} = insertShape(data{1},"rectangle",data{2});
%     reset(augmentedTrainingData);
% end
% figure
% montage(augmentedData,BorderSize=10)
% %%
% %yolov3 training options
% numEpochs = 20;
% miniBatchSize = 8;
% learningRate = 0.001;
% warmupPeriod = 1000;
% l2Regularization = 0.0005;
% penaltyThreshold = 0.5;
% velocity = [];
% preprocessedTrainingData = transform(augmentedTrainingData, @(data)preprocess(detector, data));
% mbqTrain = minibatchqueue(preprocessedTrainingData, 2,...
%         "MiniBatchSize", miniBatchSize,...
%         "MiniBatchFcn", @(images, boxes, labels) createBatchData(images, boxes, labels, classNames), ...
%         "MiniBatchFormat", ["SSCB", ""],...
%         "DispatchInBackground", true,...
%         "OutputCast", ["", "double"]);
% %%
% %YoloV3 training
% fig = figure;
% [lossPlotter, learningRatePlotter] = configureTrainingProgressPlotter(fig);
% 
% iteration = 0;
% % Custom training loop.
% for epoch = 1:numEpochs
%       
%     reset(mbqTrain);
%     shuffle(mbqTrain);
%     
%     while(hasdata(mbqTrain))
%         iteration = iteration + 1;
%        
%         [XTrain, YTrain] = next(mbqTrain);
%         
%         % Evaluate the model gradients and loss using dlfeval and the
%         % modelGradients function.
%         [gradients, state, lossInfo] = dlfeval(@modelGradients, detector, XTrain, YTrain, penaltyThreshold);
% 
%         % Apply L2 regularization.
%         gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, detector.Learnables);
% 
%         % Determine the current learning rate value.
%         currentLR = piecewiseLearningRateWithWarmup(iteration, epoch, learningRate, warmupPeriod, numEpochs);
% 
%         % Update the detector learnable parameters using the SGDM optimizer.
%         [detector.Learnables, velocity] = sgdmupdate(detector.Learnables, gradients, velocity, currentLR);
% 
%         % Update the state parameters of dlnetwork.
%         detector.State = state;
%           
%         % Display progress.
%         displayLossInfo(epoch, iteration, currentLR, lossInfo);  
%             
%         % Update training plot with new points.
%         updatePlots(lossPlotter, learningRatePlotter, iteration, currentLR, lossInfo.totalLoss);
%     end        
% end
% %%
% %YoloV4 Training Options
% % options = trainingOptions("adam",...
% %     GradientDecayFactor=0.9,...
% %     SquaredGradientDecayFactor=0.999,...
% %     InitialLearnRate=0.001,...
% %     LearnRateSchedule="piecewise",...
% %     MiniBatchSize=4,...
% %     L2Regularization=0.0005,...
% %     MaxEpochs=110,...
% %     BatchNormalizationStatistics="moving",...
% %     DispatchInBackground=false,...
% %     ResetInputNormalization=false,...
% %     Shuffle="every-epoch",...
% %     VerboseFrequency=20,...
% %     CheckpointPath=fullfile("./Checkpoints/"),...
% %     ValidationData=validationData,...
% %     OutputNetwork="best-validation-loss",...
% %     ExecutionEnvironment="auto");
% %%
% %Yolov4 or V2 training
% % [detector,info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);
% %[detector, info] = trainYOLOv2ObjectDetector(augmentedTrainingData,lgraph,options);
% %%
% %debug
% %detector = load("yolov3_squeeze_80epochs_pruned.mat").detector;
% %%
% %Read Test Image from Image Datastore
% imgIdx = 156;
% I = readimage(imds,imgIdx);
% %I = single(rescale(I)); 
% %I = imresize(I, [800 800]);%detector.InputSize(1:2)); 
% [bboxes,scores,labels] = detect(detector,I,Threshold=0.5);
% I = insertObjectAnnotation(I,"rectangle",bboxes,labels);
% figure
% imshow(I)
% %%
% %Preprocess Test Data and get a summary of evaluation Results
% preprocessedTestData = transform(testData, @(data)resizeImageAndLabel(data, inputSize));
% results = detect(detector,preprocessedTestData, MiniBatchSize=4, Threshold=0.0);
% [ap, precision, recall] = evaluateDetectionPrecision(results, preprocessedTestData);
% %%
% %Plot Precision vs Recall graph
% classID = 1;
% figure
% plot(recall,precision)  
% xlabel("Recall")
% ylabel("Precision")
% grid on
% title(sprintf("Average Precision = %.2f",ap(classID)))
% %%
% %Quantize network -- UNFINISHED
% quantObj = dlquantizer(detector,"ExecutionEnvironment","FPGA");
%%
[calibrationData,validationData] = splitEachLabel(imageData,0.5,'randomized');
%%
% hPC = dlhdl.ProcessorConfig('Bitstream', 'zcu102_single');
% simObj = dlhdl.Simulator('Network',netTransfer,'ProcessorConfig',hPC);
predict
%% Load Trained model instead
load("yolov3_20epoch.mat")

%% Deploy Model
hdlsetuptoolpath('ToolName', 'Xilinx Vivado', 'ToolPath', '/mnt/xilinx/Vivado/2020.2/bin/vivado') % may not be required
hTarget = dlhdl.Target('Xilinx', 'Interface', 'Ethernet')
hW = dlhdl.Workflow('Network', detector.Network, 'Bitstream', 'zc706_single', 'Target', hTarget)
hW.compile

%% made redeployable
hW.deploy

%% Test



%example from detector
% img_in = imread(fullfile('images/t72/','t72_l1.jpg'));
% figure
% imshow(img_in)
% img = detector.preprocess(img_in);
% img = im2single(img);
% [bboxes, scores, labels] = detector.detect(img, 'DetectionPreprocessing', 'none');
% output = insertObjectAnnotation(img, 'Rectangle', bboxes, labels);
% figure
% imshow(output)

%% Predict 
img = imread(fullfile('images/t72/','t72_l7.jpg'));
img_in = imresize(single(rescale(img)), detector.InputSize(1:2));
dlX = dlarray(img_in, 'SSC')
features = cell(size(hW.Network.OutputNames));
[features{:}] = hW.predict(dlX)
[bboxes, scores, labels] = processYOLOv3Output(detector.AnchorBoxes, detector.InputSize, detector.ClassNames, features, img_in);
figure
if size(scores) ~= 0
    resultImage = insertObjectAnnotation(img_in, 'rectangle', bboxes, scores);
    imshow(resultImage);
else
    imshow(img_in)
end


%% Functions

function data = replaceLabels(data, lbl)
    for i = 1:size(data.LabelData,1)
        cat = data.LabelData{i,2};
        for j = 1:size(cat)
            cat(j) = lbl;
        end
        data.LabelData{i,1} = cat;
    end
end
function currentLR = piecewiseLearningRateWithWarmup(iteration, epoch, learningRate, warmupPeriod, numEpochs)
    % The piecewiseLearningRateWithWarmup function computes the current
    % learning rate based on the iteration number.
    persistent warmUpEpoch;
    
    if iteration <= warmupPeriod
        % Increase the learning rate for number of iterations in warmup period.
        currentLR = learningRate * ((iteration/warmupPeriod)^4);
        warmUpEpoch = epoch;
    elseif iteration >= warmupPeriod && epoch < warmUpEpoch+floor(0.6*(numEpochs-warmUpEpoch))
        % After warm up period, keep the learning rate constant if the remaining number of epochs is less than 60 percent. 
        currentLR = learningRate;
        
    elseif epoch >= warmUpEpoch + floor(0.6*(numEpochs-warmUpEpoch)) && epoch < warmUpEpoch+floor(0.9*(numEpochs-warmUpEpoch))
        % If the remaining number of epochs is more than 60 percent but less
        % than 90 percent multiply the learning rate by 0.1.
        currentLR = learningRate*0.1;
        
    else
        % If remaining epochs are more than 90 percent multiply the learning
        % rate by 0.01.
        currentLR = learningRate*0.01;
    end
end
function [gradients, state, info] = modelGradients(detector, XTrain, YTrain, penaltyThreshold)
inputImageSize = size(XTrain,1:2);

% Gather the ground truths in the CPU for post processing
YTrain = gather(extractdata(YTrain));

% Extract the predictions from the detector.
[gatheredPredictions, YPredCell, state] = forward(detector, XTrain);

% Generate target for predictions from the ground truth data.
[boxTarget, objectnessTarget, classTarget, objectMaskTarget, boxErrorScale] = generateTargets(gatheredPredictions,...
    YTrain, inputImageSize, detector.AnchorBoxes, penaltyThreshold);

% Compute the loss.
boxLoss = bboxOffsetLoss(YPredCell(:,[2 3 7 8]),boxTarget,objectMaskTarget,boxErrorScale);
objLoss = objectnessLoss(YPredCell(:,1),objectnessTarget,objectMaskTarget);
clsLoss = classConfidenceLoss(YPredCell(:,6),classTarget,objectMaskTarget);
totalLoss = boxLoss + objLoss + clsLoss;

info.boxLoss = boxLoss;
info.objLoss = objLoss;
info.clsLoss = clsLoss;
info.totalLoss = totalLoss;

% Compute gradients of learnables with regard to loss.
gradients = dlgradient(totalLoss, detector.Learnables);
end

function boxLoss = bboxOffsetLoss(boxPredCell, boxDeltaTarget, boxMaskTarget, boxErrorScaleTarget)
% Mean squared error for bounding box position.
lossX = sum(cellfun(@(a,b,c,d) mse(a.*c.*d,b.*c.*d),boxPredCell(:,1),boxDeltaTarget(:,1),boxMaskTarget(:,1),boxErrorScaleTarget));
lossY = sum(cellfun(@(a,b,c,d) mse(a.*c.*d,b.*c.*d),boxPredCell(:,2),boxDeltaTarget(:,2),boxMaskTarget(:,1),boxErrorScaleTarget));
lossW = sum(cellfun(@(a,b,c,d) mse(a.*c.*d,b.*c.*d),boxPredCell(:,3),boxDeltaTarget(:,3),boxMaskTarget(:,1),boxErrorScaleTarget));
lossH = sum(cellfun(@(a,b,c,d) mse(a.*c.*d,b.*c.*d),boxPredCell(:,4),boxDeltaTarget(:,4),boxMaskTarget(:,1),boxErrorScaleTarget));
boxLoss = lossX+lossY+lossW+lossH;
end

function objLoss = objectnessLoss(objectnessPredCell, objectnessDeltaTarget, boxMaskTarget)
% Binary cross-entropy loss for objectness score.
objLoss = sum(cellfun(@(a,b,c) crossentropy(a.*c,b.*c,'TargetCategories','independent'),objectnessPredCell,objectnessDeltaTarget,boxMaskTarget(:,2)));
end

function clsLoss = classConfidenceLoss(classPredCell, classTarget, boxMaskTarget)
% Binary cross-entropy loss for class confidence score.
clsLoss = sum(cellfun(@(a,b,c) crossentropy(a.*c,b.*c,'TargetCategories','independent'),classPredCell,classTarget,boxMaskTarget(:,3)));
end

function [XTrain, YTrain] = createBatchData(data, groundTruthBoxes, groundTruthClasses, classNames)
% Returns images combined along the batch dimension in XTrain and
% normalized bounding boxes concatenated with classIDs in YTrain

% Concatenate images along the batch dimension.
XTrain = cat(4, data{:,1});

% Get class IDs from the class names.
classNames = repmat({categorical(classNames')}, size(groundTruthClasses));
[~, classIndices] = cellfun(@(a,b)ismember(a,b), groundTruthClasses, classNames, 'UniformOutput', false);

% Append the label indexes and training image size to scaled bounding boxes
% and create a single cell array of responses.
combinedResponses = cellfun(@(bbox, classid)[bbox, classid], groundTruthBoxes, classIndices, 'UniformOutput', false);
len = max( cellfun(@(x)size(x,1), combinedResponses ) );
paddedBBoxes = cellfun( @(v) padarray(v,[len-size(v,1),0],0,'post'), combinedResponses, 'UniformOutput',false);
YTrain = cat(4, paddedBBoxes{:,1});
end
function data = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.

data = cell(size(A));
for ii = 1:size(A,1)
    I = A{ii,1};
    bboxes = A{ii,2};
    labels = A{ii,3};
    sz = size(I);

    if numel(sz) == 3 && sz(3) == 3
        I = jitterColorHSV(I,...
            contrast=0.0,...
            Hue=0.1,...
            Saturation=0.2,...
            Brightness=0.2);
    end
    
    % Randomly flip image.
    tform = randomAffine2d(XReflection=true,Scale=[1 1.1]);
    rout = affineOutputView(sz,tform,BoundsStyle="centerOutput");
    I = imwarp(I,tform,OutputView=rout);
    
    % Apply same transform to boxes.
    [bboxes,indices] = bboxwarp(bboxes,tform,rout,OverlapThreshold=0.25);
    labels = labels(indices);
    
    % Return original data only when all boxes are removed by warping.
    if isempty(indices)
        data(ii,:) = A(ii,:);
    else
        data(ii,:) = {I,bboxes,labels};
    end
end
end

function data = preprocessData(data,targetSize)
% Resize the images and scale the pixels to between 0 and 1. Also scale the
% corresponding bounding boxes.

for ii = 1:size(data,1)
    I = data{ii,1};
    imgSize = size(I);
    
    bboxes = data{ii,2};

    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);
    
    data(ii,1:2) = {I,bboxes};
end
end

function data = resizeImageAndLabel(data,targetSize)
% Resize the images and scale the corresponding bounding boxes.

    scale = (targetSize(1:2))./size(data{1},[1 2]);
    data{1} = imresize(data{1},targetSize(1:2));
    data{2} = bboxresize(data{2},scale);

    data{2} = floor(data{2});
    imageSize = targetSize(1:2);
    boxes = data{2};
    % Set boxes with negative values to have value 1.
    boxes(boxes<=0) = 1;
    
    % Validate if bounding box in within image boundary.
    boxes(:,3) = min(boxes(:,3),imageSize(2) - boxes(:,1)-1);
    boxes(:,4) = min(boxes(:,4),imageSize(1) - boxes(:,2)-1);
    
    data{2} = boxes; 

end
function [lossPlotter, learningRatePlotter] = configureTrainingProgressPlotter(f)
% Create the subplots to display the loss and learning rate.
figure(f);
clf
subplot(2,1,1);
ylabel('Learning Rate');
xlabel('Iteration');
learningRatePlotter = animatedline;
subplot(2,1,2);
ylabel('Total Loss');
xlabel('Iteration');
lossPlotter = animatedline;
end

function displayLossInfo(epoch, iteration, currentLR, lossInfo)
% Display loss information for each iteration.
disp("Epoch : " + epoch + " | Iteration : " + iteration + " | Learning Rate : " + currentLR + ...
   " | Total Loss : " + double(gather(extractdata(lossInfo.totalLoss))) + ...
   " | Box Loss : " + double(gather(extractdata(lossInfo.boxLoss))) + ...
   " | Object Loss : " + double(gather(extractdata(lossInfo.objLoss))) + ...
   " | Class Loss : " + double(gather(extractdata(lossInfo.clsLoss))));
end

function updatePlots(lossPlotter, learningRatePlotter, iteration, currentLR, totalLoss)
% Update loss and learning rate plots.
addpoints(lossPlotter, iteration, double(extractdata(gather(totalLoss))));
addpoints(learningRatePlotter, iteration, currentLR);
drawnow
end

function detector = downloadPretrainedYOLOv3Detector()
% Download a pretrained yolov3 detector.
if ~exist('yolov3SqueezeNetVehicleExample_21aSPKG.mat', 'file')
    if ~exist('yolov3SqueezeNetVehicleExample_21aSPKG.zip', 'file')
        disp('Downloading pretrained detector...');
        pretrainedURL = 'https://ssd.mathworks.com/supportfiles/vision/data/yolov3SqueezeNetVehicleExample_21aSPKG.zip';
        websave('yolov3SqueezeNetVehicleExample_21aSPKG.zip', pretrainedURL);
    end
    unzip('yolov3SqueezeNetVehicleExample_21aSPKG.zip');
end
pretrained = load("yolov3SqueezeNetVehicleExample_21aSPKG.mat");
detector = pretrained.detector;
end

function [bboxes, scores, labels] = processYOLOv3Output(anchorBoxes, inputSize, classNames, features, img)
% This function converts the feature maps from multiple detection heads to bounding boxes, scores and labels
% processYOLOv3Output is C code generatable

% Breaks down the raw output from predict function into Confidence score, X, Y, Width,
% Height and Class probabilities for each output from detection head
predictions = iYolov3Transform(features, anchorBoxes);

% Initialize parameters for post-processing
inputSize2d = inputSize(1:2);
info.PreprocessedImageSize = inputSize2d(1:2);
info.ScaleX = size(img,1)/inputSize2d(1);
info.ScaleY = size(img,2)/inputSize2d(1);
params.MinSize = [1 1];
params.MaxSize = size(img(:,:,1));
params.Threshold = 0.5;
params.FractionDownsampling = 1;
params.DetectionInputWasBatchOfImages = false;
params.NetworkInputSize = inputSize;
params.DetectionPreprocessing = "none";
params.SelectStrongest = 1;
bboxes = [];                                                                                                                               
scores = [];                                                                                                                                
labels = [];                                                                                                                             

% Post-process the predictions to get bounding boxes, scores and labels
[bboxes, scores, labels] = iPostprocessMultipleDetection(anchorBoxes, inputSize, classNames, predictions, info, params);
end

function [bboxes, scores, labels] = iPostprocessMultipleDetection (anchorBoxes, inputSize, classNames, YPredData, info, params)
% Post-process the predictions to get bounding boxes, scores and labels

% YpredData is a (x,8) cell array, where x = number of detection heads
% Information in each column is:
% column 1 -> confidence scores
% column 2 to column 5 -> X offset, Y offset, Width, Height of anchor boxes
% column 6 -> class probabilities
% column 7-8 -> copy of width and height of anchor boxes

% Initialize parameters for post-processing
classes = classNames;
predictions = YPredData;
extractPredictions = cell(size(predictions));
% Extract dlarray data
for i = 1:size(extractPredictions,1)
    for j = 1:size(extractPredictions,2)
        extractPredictions{i,j} = extractdata(predictions{i,j});
    end
end

% Storing the values of columns 2 to 5 of extractPredictions
% Columns 2 to 5 represent information about X-coordinate, Y-coordinate, Width and Height of predicted anchor boxes
extractedCoordinates = cell(size(predictions,1),4);
for i = 1:size(predictions,1)
    for j = 2:5 
        extractedCoordinates{i,j-1} = extractPredictions{i,j};
    end
end

% Convert predictions from grid cell coordinates to box coordinates.
boxCoordinates = anchorBoxGenerator(anchorBoxes, inputSize, classNames, extractedCoordinates, params.NetworkInputSize);
% Replace grid cell coordinates in extractPredictions with box coordinates
for i = 1:size(YPredData,1)
    for j = 2:5 
        extractPredictions{i,j} = single(boxCoordinates{i,j-1});
    end
end

% 1. Convert bboxes from spatial to pixel dimension
% 2. Combine the prediction from different heads.
% 3. Filter detections based on threshold.

% Reshaping the matrices corresponding to confidence scores and  bounding boxes
detections = cell(size(YPredData,1),6);
for i = 1:size(detections,1)
    for j = 1:5
        detections{i,j} = reshapePredictions(extractPredictions{i,j});
    end
end
% Reshaping the matrices corresponding to class probablities
numClasses = repmat({numel(classes)},[size(detections,1),1]);
for i = 1:size(detections,1)
    detections{i,6} = reshapeClasses(extractPredictions{i,6},numClasses{i,1}); 
end

% cell2mat converts the cell of matrices into one matrix, this combines the
% predictions of all detection heads
detections = cell2mat(detections);

% Getting the most probable class and corresponding index
[classProbs, classIdx] = max(detections(:,6:end),[],2);
detections(:,1) = detections(:,1).*classProbs;
detections(:,6) = classIdx;

% Keep detections whose confidence score is greater than threshold.
detections = detections(detections(:,1) >= params.Threshold,:);

[bboxes, scores, labels] = iPostProcessDetections(detections, classes, info, params);
end

function [bboxes, scores, labels] = iPostProcessDetections(detections, classes, info, params)
% Resizes the anchor boxes, filters anchor boxes based on size and apply
% NMS to eliminate overlapping anchor boxes
if ~isempty(detections)

    % Obtain bounding boxes and class data for pre-processed image
    scorePred = detections(:,1);
    bboxesTmp = detections(:,2:5);
    classPred = detections(:,6);
    inputImageSize = ones(1,2);
    inputImageSize(2) = info.ScaleX.*info.PreprocessedImageSize(2);
    inputImageSize(1) = info.ScaleY.*info.PreprocessedImageSize(1);
    % Resize boxes to actual image size.
    scale = [inputImageSize(2) inputImageSize(1) inputImageSize(2) inputImageSize(1)];
    bboxPred = bboxesTmp.*scale;
    % Convert x and y position of detections from centre to top-left.
    bboxPred = iConvertCenterToTopLeft(bboxPred);

    % Filter boxes based on MinSize, MaxSize.
    [bboxPred, scorePred, classPred] = filterBBoxes(params.MinSize, params.MaxSize, bboxPred, scorePred, classPred);

    % Apply NMS to eliminate boxes having significant overlap
    if params.SelectStrongest
        [bboxes, scores, classNames] = selectStrongestBboxMulticlass(bboxPred, scorePred, classPred ,...
            'RatioType', 'Union', 'OverlapThreshold', 0.4);
    else
        bboxes = bboxPred;
        scores = scorePred;
        classNames = classPred;
    end

    % Limit width detections
    detectionsWd = min((bboxes(:,1) + bboxes(:,3)),inputImageSize(1,2));
    bboxes(:,3) = detectionsWd(:,1) - bboxes(:,1);

    % Limit height detections
    detectionsHt = min((bboxes(:,2) + bboxes(:,4)),inputImageSize(1,1));
    bboxes(:,4) = detectionsHt(:,1) - bboxes(:,2);
    bboxes(bboxes<1) = 1;

    % Convert classId to classNames.
    labels = categorical(classes,cellstr(classes));
    labels = labels(classNames);

else
    % If detections are empty then bounding boxes, scores and labels should
    % be empty
    bboxes = zeros(0,4,'single');
    scores = zeros(0,1,'single');
    labels = categorical(classes);
end
end

function x = reshapePredictions(pred)
% Reshapes the matrices corresponding to scores, X, Y, Width and Height to
% make them compatible for combining the outputs of different detection
% heads
[h,w,c,n] = size(pred);
x = reshape(pred,h*w*c,1,n);
end

function x = reshapeClasses(pred,numClasses)
% Reshapes the matrices corresponding to the class probabilities, to make it
% compatible for combining the outputs of different detection heads
[h,w,c,n] = size(pred);
numAnchors = c/numClasses;
x = reshape(pred,h*w,numClasses,numAnchors,n);
x = permute(x,[1,3,2,4]);
[h,w,c,n] = size(x);
x = reshape(x,h*w,c,n);
end

function bboxes = iConvertCenterToTopLeft(bboxes)
% Convert x and y position of detections from centre to top-left.
bboxes(:,1) = bboxes(:,1) - bboxes(:,3)/2 + 0.5;
bboxes(:,2) = bboxes(:,2) - bboxes(:,4)/2 + 0.5;
bboxes = floor(bboxes);
bboxes(bboxes<1) = 1;
end

function tiledAnchors = anchorBoxGenerator(anchorBoxes, inputSize, classNames,YPredCell,inputImageSize)
% Convert grid cell coordinates to box coordinates.
% Generate tiled anchor offset.
tiledAnchors = cell(size(YPredCell));
for i = 1:size(YPredCell,1)
    anchors = anchorBoxes{i,:};
    [h,w,~,n] = size(YPredCell{i,1});
    [tiledAnchors{i,2},tiledAnchors{i,1}] = ndgrid(0:h-1,0:w-1,1:size(anchors,1),1:n);
    [~,~,tiledAnchors{i,3}] = ndgrid(0:h-1,0:w-1,anchors(:,2),1:n);
    [~,~,tiledAnchors{i,4}] = ndgrid(0:h-1,0:w-1,anchors(:,1),1:n);
end

for i = 1:size(YPredCell,1)
    [h,w,~,~] = size(YPredCell{i,1});
    tiledAnchors{i,1} = double((tiledAnchors{i,1} + YPredCell{i,1})./w);
    tiledAnchors{i,2} = double((tiledAnchors{i,2} + YPredCell{i,2})./h);
    tiledAnchors{i,3} = double((tiledAnchors{i,3}.*YPredCell{i,3})./inputImageSize(2));
    tiledAnchors{i,4} = double((tiledAnchors{i,4}.*YPredCell{i,4})./inputImageSize(1));
end
end

function predictions = iYolov3Transform(YPredictions, anchorBoxes)
% This function breaks down the raw output from predict function into Confidence score, X, Y, Width,
% Height and Class probabilities for each output from detection head

predictions = cell(size(YPredictions,1),size(YPredictions,2) + 2);

for idx = 1:size(YPredictions,1)
    % Get the required info on feature size.
    numChannelsPred = size(YPredictions{idx},3);  %number of channels in a feature map
    numAnchors = size(anchorBoxes{idx},1);    %number of anchor boxes per grid
    numPredElemsPerAnchors = numChannelsPred/numAnchors;
    channelsPredIdx = 1:numChannelsPred;
    predictionIdx = ones([1,numAnchors.*5]);

    % X positions.
    startIdx = 1;
    endIdx = numChannelsPred;
    stride = numPredElemsPerAnchors;
    predictions{idx,2} = YPredictions{idx}(:,:,startIdx:stride:endIdx,:);
    predictionIdx = [predictionIdx startIdx:stride:endIdx];

    % Y positions.
    startIdx = 2;
    endIdx = numChannelsPred;
    stride = numPredElemsPerAnchors;
    predictions{idx,3} = YPredictions{idx}(:,:,startIdx:stride:endIdx,:);
    predictionIdx = [predictionIdx startIdx:stride:endIdx];

    % Width.
    startIdx = 3;
    endIdx = numChannelsPred;
    stride = numPredElemsPerAnchors;
    predictions{idx,4} = YPredictions{idx}(:,:,startIdx:stride:endIdx,:);
    predictionIdx = [predictionIdx startIdx:stride:endIdx];

    % Height.
    startIdx = 4;
    endIdx = numChannelsPred;
    stride = numPredElemsPerAnchors;
    predictions{idx,5} = YPredictions{idx}(:,:,startIdx:stride:endIdx,:);
    predictionIdx = [predictionIdx startIdx:stride:endIdx];

    % Confidence scores.
    startIdx = 5;
    endIdx = numChannelsPred;
    stride = numPredElemsPerAnchors;
    predictions{idx,1} = YPredictions{idx}(:,:,startIdx:stride:endIdx,:);
    predictionIdx = [predictionIdx startIdx:stride:endIdx];

    % Class probabilities.
    classIdx = setdiff(channelsPredIdx,predictionIdx);
    predictions{idx,6} = YPredictions{idx}(:,:,classIdx,:);
end

for i = 1:size(predictions,1)
    predictions{i,7} = predictions{i,4};
    predictions{i,8} = predictions{i,5};
end

% Apply activation to the predicted cell array
% Apply sigmoid activation to columns 1-3 (Confidence score, X, Y)
for i = 1:size(predictions,1)
    for j = 1:3
        predictions{i,j} = sigmoid(predictions{i,j});
    end
end
% Apply exponentiation to columns 4-5 (Width, Height)
for i = 1:size(predictions,1)
    for j = 4:5
        predictions{i,j} = exp(predictions{i,j});
    end
end
% Apply sigmoid activation to column 6 (Class probabilities)
for i = 1:size(predictions,1)
    for j = 6
        predictions{i,j} = sigmoid(predictions{i,j});
    end
end
end

function [bboxPred, scorePred, classPred] = filterBBoxes(minSize, maxSize, bboxPred, scorePred, classPred)
% Filter boxes based on MinSize, MaxSize
[bboxPred, scorePred, classPred] = filterSmallBBoxes(minSize, bboxPred, scorePred, classPred);
[bboxPred, scorePred, classPred] = filterLargeBBoxes(maxSize, bboxPred, scorePred, classPred);
end

function varargout = filterSmallBBoxes(minSize, varargin)
% Filter boxes based on MinSize
bboxes = varargin{1};
tooSmall = any((bboxes(:,[4 3]) < minSize),2);
for ii = 1:numel(varargin)
    varargout{ii} = varargin{ii}(~tooSmall,:);
end
end

function varargout = filterLargeBBoxes(maxSize, varargin)
% Filter boxes based on MaxSize
bboxes = varargin{1};
tooBig = any((bboxes(:,[4 3]) > maxSize),2);
for ii = 1:numel(varargin)
    varargout{ii} = varargin{ii}(~tooBig,:);
end
end

function m = cell2mat(c)
% Converts the cell of matrices into one matrix by concatenating
% the output corresponding to each feature map

elements = numel(c);
% If number of elements is 0 return an empty array
if elements == 0
    m = [];
    return
end
% If number of elements is 1, return same element as matrix
if elements == 1
    if isnumeric(c{1}) || ischar(c{1}) || islogical(c{1}) || isstruct(c{1})
        m = c{1};
        return
    end
end
% Error out for unsupported cell content
ciscell = iscell(c{1});
cisobj = isobject(c{1});
if cisobj || ciscell
    disp('CELL2MAT does not support cell arrays containing cell arrays or objects.');
end
% If input input is struct, extract field names of structure into a cell
if isstruct(c{1})
    cfields = cell(elements,1);
    for n = 1:elements
        cfields{n} = fieldnames(c{n});
    end
    if ~isequal(cfields{:})
        disp('The field names of each cell array element must be consistent and in consistent order.');
    end
end
% If number of dimensions is 2 
if ndims(c) == 2
    rows = size(c,1);
    cols = size(c,2);
    if (rows < cols)
        % If rows is less than columns first concatenate each column into 1
        % row then concatenate all the rows
        m = cell(rows,1);
        for n = 1:rows
            m{n} = cat(2,c{n,:});
        end
        m = cat(1,m{:});
    else
        % If columns is less than rows, first concatenate each corresponding
        % row into columns, then combine all columns into 1
        m = cell(1,cols);
        for n = 1:cols
            m{n} = cat(1,c{:,n});
        end
        m = cat(2,m{:});
    end
    return
end
end
