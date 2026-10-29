%% ME 3310 | Multi-subject DNN versus linear SVM
% step06_compare_dnn.m
%
% Engineering question:
% Does a NONLINEAR model improve motion recognition across people?
%
% Same 3-second sliding windows and original 8 engineered IMU features.
% Three models with identical evaluation subjects:
%   A. Linear SVM using 2 acceleration features (original baseline)
%   B. Linear SVM using ALL 8 features (controls for feature count)
%   C. Small DNN using ALL 8 features (tests model nonlinearity)
%
% DNN is a feedforward multi-layer perceptron (MLP), not a 1D CNN.
% fitcnet belongs to Statistics and Machine Learning Toolbox; no
% Deep Learning Toolbox is needed.
%
% Validation: leave ONE training subject out at a time; average balanced
% accuracy over subjects, not over correlated sliding windows.
%
% The original four test subjects were already inspected in earlier
% experiments. Their results can be printed as a SECONDARY DIAGNOSTIC,
% but must NOT be described as a new untouched test.
%
% This script NEVER alters original measurements or existing scripts.

clear; clc;
scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder), scriptFolder = pwd; end
dataPath = fullfile(scriptFolder,'ME3310_USC_HAD_WindowFeatures.mat');
assert(isfile(dataPath), ...
    'Feature file missing: run step03_extract_window_features.m first.');
S = load(dataPath);
assert(isfield(S,'X') && isfield(S,'Y') && ...
    isfield(S,'subjectID'), 'Unexpected feature file format.');

X = double(S.X);
Y = double(S.Y(:));
subjectID = double(S.subjectID(:));
assert(size(X,2)>=8,'Expected the original eight IMU features.');
assert(numel(Y)==size(X,1) && numel(subjectID)==numel(Y), ...
    'Feature, label and subject arrays have inconsistent lengths.');
assert(all(isfinite(X(:))) && all(ismember(unique(Y),[0 1])), ...
    'Predictors must be finite; labels must be 0 and 1.');
assert(exist('fitcnet','file')==2, ...
    ['fitcnet is required. Check Statistics and Machine Learning ' ...
     'Toolbox and your MATLAB version.']);
assert(exist('fitcsvm','file')==2, ...
    'fitcsvm requires Statistics and Machine Learning Toolbox.');

%% FIXED model settings (do not tune using the previously seen subjects)
rng(3310);
hiddenLayers = [32 16];
lambdaL2 = 1e-4;
maxIterations = 250;
showPreviouslySeenSubjects = true;
modelNames = {'SVM: 2 features','SVM: 8 features','DNN: 8 features'};
modelCount = numel(modelNames);

fprintf('Features used by DNN: %d\n',8);
fprintf('DNN: hidden layers %s, ReLU, L2 lambda %.1g\n', ...
    mat2str(hiddenLayers),lambdaL2);
fprintf('3-second windows are correlated; subject is the CV unit.\n');

%% Reproduce the EXACT subject split of the existing SVM demonstration
allSubjects = unique(subjectID);
allSubjects = allSubjects(randperm(numel(allSubjects)));
nTrainSubject = max(2,floor(0.75*numel(allSubjects)));
trainSubjects = sort(allSubjects(1:nTrainSubject));
previouslySeenSubjects = sort(allSubjects(nTrainSubject+1:end));
trainSubjectsMask = ismember(subjectID,trainSubjects);
previousMask = ismember(subjectID,previouslySeenSubjects);
assert(numel(trainSubjects)>=3 && any(previousMask), ...
    'Not enough subjects for group-based comparison.');

fprintf('\nTRAINING subjects: %s\n',mat2str(trainSubjects(:)'));
fprintf('PREVIOUSLY VIEWED subjects: %s\n', ...
    mat2str(previouslySeenSubjects(:)'));

%% Subject-wise cross-validation on TRAINING subjects only
nGroups=numel(trainSubjects);
foldBalanced = nan(nGroups,modelCount);
foldRaw = nan(nGroups,modelCount);

for f=1:nGroups
    valSubject=trainSubjects(f);
    foldTrain = trainSubjectsMask & (subjectID~=valSubject);
    foldVal = trainSubjectsMask & (subjectID==valSubject);
    yt=Y(foldVal);

    assert(all(ismember([0 1],unique(Y(foldTrain)))) && ...
        all(ismember([0 1],unique(yt))), ...
        'Each fold must contain both motions.');

    % Model A: original two-feature linear SVM
    svm2 = fitcsvm(X(foldTrain,1:2),Y(foldTrain), ...
        'KernelFunction','linear','Standardize',true, ...
        'ClassNames',[0 1]);
    ySVM2=predict(svm2,X(foldVal,1:2));

    % Model B: same architecture, now all eight features
    svm8 = fitcsvm(X(foldTrain,1:8),Y(foldTrain), ...
        'KernelFunction','linear','Standardize',true, ...
        'ClassNames',[0 1]);
    ySVM8=predict(svm8,X(foldVal,1:8));

    % Model C: two-hidden-layer nonlinear neural network
    % Standardization statistics are estimated from foldTrain only.
    rng(3310+f);
    dnn = fitcnet(X(foldTrain,1:8),Y(foldTrain), ...
        'LayerSizes',hiddenLayers, ...
        'Activations','relu', ...
        'Standardize',true, ...
        'Lambda',lambdaL2, ...
        'IterationLimit',maxIterations, ...
        'ClassNames',[0 1]);
    yDNN=predict(dnn,X(foldVal,1:8));

    predictions={ySVM2,ySVM8,yDNN};
    for m=1:modelCount
        yp=double(predictions{m}(:));
        upRecall=mean(yp(yt==0)==0);
        downRecall=mean(yp(yt==1)==1);
        foldBalanced(f,m)=100*mean([upRecall,downRecall]);
        foldRaw(f,m)=100*mean(yp==yt);
    end

    fprintf(['Validation subject %2d | SVM-2 %5.1f%% | ', ...
        'SVM-8 %5.1f%% | DNN-8 %5.1f%% (balanced)\n'], ...
        valSubject,foldBalanced(f,1), ...
        foldBalanced(f,2),foldBalanced(f,3));
end

%% Summary: equal subject weights
fprintf('\nTRAINING-SUBJECT LOSO BALANCED ACCURACY:\n');
fprintf('%-22s %10s %10s %10s\n','Model','Mean','SD','Median');
for m=1:modelCount
    fprintf('%-22s %9.1f%% %9.1f%% %9.1f%%\n',modelNames{m}, ...
        mean(foldBalanced(:,m)),std(foldBalanced(:,m)), ...
        median(foldBalanced(:,m)));
end

T = array2table([trainSubjects(:),foldBalanced], ...
    'VariableNames',{'Subject','SVM_2','SVM_8','DNN_8'});
disp(T);

figure('Name','DNN versus SVM: subject-wise CV');
plot(trainSubjects,foldBalanced,'o-','LineWidth',1.3);
hold on; yline(50,'k--','Chance-level balanced accuracy');
ylim([0 105]); grid on;
xlabel('Training subject ID');
ylabel('Balanced accuracy (%)');
title('Leave-one-training-subject-out cross-validation');
legend(modelNames,'Location','best');

figure('Name','Mean subject-balanced accuracy');
bar(mean(foldBalanced,1));
hold on; errorbar(1:modelCount,mean(foldBalanced,1), ...
    std(foldBalanced,0,1),'k.','LineWidth',1.2);
xticks(1:modelCount); xticklabels(modelNames);
ylabel('Balanced accuracy (%)');
title('Across-subject performance (error bars: subject SD)');
ylim([0 105]); grid on;

%% Fit on ALL ten original training subjects
fprintf('\nFitting all training subjects for a secondary diagnostic...\n');
svm2 = fitcsvm(X(trainSubjectsMask,1:2),Y(trainSubjectsMask), ...
    'KernelFunction','linear','Standardize',true,'ClassNames',[0 1]);
svm8 = fitcsvm(X(trainSubjectsMask,1:8),Y(trainSubjectsMask), ...
    'KernelFunction','linear','Standardize',true,'ClassNames',[0 1]);
rng(3399);
dnn = fitcnet(X(trainSubjectsMask,1:8),Y(trainSubjectsMask), ...
    'LayerSizes',hiddenLayers,'Activations','relu', ...
    'Standardize',true,'Lambda',lambdaL2, ...
    'IterationLimit',maxIterations,'ClassNames',[0 1]);

if showPreviouslySeenSubjects
    actual=Y(previousMask);
    sid=subjectID(previousMask);
    allPred={predict(svm2,X(previousMask,1:2)), ...
             predict(svm8,X(previousMask,1:8)), ...
             predict(dnn,X(previousMask,1:8))};

    diagnostic = nan(numel(previouslySeenSubjects),modelCount);
    fprintf('\nDIAGNOSTIC on previously inspected subjects (NOT new test):\n');
    for m=1:modelCount
        yp=double(allPred{m}(:));
        fprintf('  %-21s: window accuracy %5.1f%%\n', ...
            modelNames{m},100*mean(yp==actual));
        for i=1:numel(previouslySeenSubjects)
            idx=sid==previouslySeenSubjects(i);
            truth=actual(idx);
            pred=yp(idx);
            diagnostic(i,m)=50*(mean(pred(truth==0)==0)+ ...
                                mean(pred(truth==1)==1));
        end
    end

    fprintf('\nPER-PERSON diagnostic balanced accuracy:\n');
    disp(array2table([previouslySeenSubjects(:),diagnostic], ...
        'VariableNames',{'Subject','SVM_2','SVM_8','DNN_8'}));

    figure('Name','DNN vs SVM: previously seen subjects');
    bar(previouslySeenSubjects,diagnostic,'grouped');
    xlabel('Previously seen subject ID');
    ylabel('Balanced accuracy (%)');
    title('Diagnostic only (these subjects were inspected previously)');
    legend(modelNames,'Location','best'); ylim([0 105]); grid on;

    figure('Name','DNN diagnostic confusion matrix');
    confusionchart(actual,allPred{3}, ...
        'Title','DNN (8 features): previously inspected subjects');
    fprintf('WARNING: do NOT call these four subjects untouched tests.\n');
end

%% Save an experimental summary, not the original IMU signals
save(fullfile(scriptFolder,'ME3310_USC_HAD_DNN_Comparison.mat'), ...
    'trainSubjects','previouslySeenSubjects','foldBalanced', ...
    'foldRaw','modelNames','hiddenLayers','lambdaL2', ...
    'maxIterations');
fprintf('\nNext question: is the bottleneck FEATURES, MODEL, or PERSON?\n');
fprintf('This model sees window summaries, NOT the original temporal waveform.\n');
