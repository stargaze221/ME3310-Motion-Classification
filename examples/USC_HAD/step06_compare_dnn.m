%% ME 3310 | Do EIGHT more frequency-band features help?
% step06_compare_dnn.m
%
% All models use the SAME experimental data and subject-wise split.
% Compare five configurations to separate the effects of feature count
% from the effects of nonlinear model complexity:
%   A. SVM:  2 original acceleration features (baseline)
%   B. SVM:  8 original IMU features
%   C. SVM: 16 features (8 original + 8 PSD band powers)
%   D. DNN:  8 original IMU features
%   E. DNN: 16 features (8 original + 8 PSD band powers)
%
% Every training/validation fold holds out a WHOLE SUBJECT.
% Subject-balanced accuracy is the primary exploratory score.
% Overlapping 3-s/0.5-s windows are NOT independent observations.
%
% The four previously viewed subjects are only a SECONDARY DIAGNOSTIC,
% NOT an untouched holdout. Do not optimize against them.
%
% DNN = 2-hidden-layer feedforward MLP via fitcnet (NOT a 1D CNN).
% Signal Processing Toolbox required by step03; Statistics and Machine
% Learning Toolbox required for fitcsvm and fitcnet.

clear; clc;
scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder), scriptFolder=pwd; end
dataPath = fullfile(scriptFolder,'ME3310_USC_HAD_WindowFeatures.mat');
assert(isfile(dataPath), ...
    'Feature file missing. Run step03_extract_window_features.m first.');
S = load(dataPath);
assert(isfield(S,'X') && isfield(S,'Y') && ...
    isfield(S,'subjectID'), 'Feature file has unexpected format.');
X = double(S.X);
Y = double(S.Y(:));
subjectID = double(S.subjectID(:));
assert(size(X,2)==16, ...
    ['Expected 16 features. Rerun the UPDATED ' ...
     'step03_extract_window_features.m after git pull.']);
assert(numel(Y)==size(X,1) && numel(subjectID)==numel(Y), ...
    'Feature, label and subject arrays have inconsistent lengths.');
assert(all(isfinite(X(:))) && isequal(unique(Y),[0;1]), ...
    'Features must be finite; labels must be exactly 0 and 1.');
assert(exist('fitcsvm','file')==2 && exist('fitcnet','file')==2, ...
    'Statistics and Machine Learning Toolbox with fitcnet is required.');

%% FIXED model settings (no tuning on previously viewed subjects)
rng(3310);
hiddenLayers = [32 16];
lambdaL2 = 1e-4;
maxIterations = 250;
showPreviouslySeenSubjects = true;

modelNames = {'SVM: 2 features','SVM: 8 features', ...
              'SVM: 16 features','DNN: 8 features', ...
              'DNN: 16 features'};
featureCols = {1:2,1:8,1:16,1:8,1:16};
usesDNN = [false false false true true];
modelCount = numel(modelNames);

fprintf('Available features: %d (8 original + 8 PSD band powers).\n', ...
    size(X,2));
if isfield(S,'featureNames')
    fprintf('Added PSD bands (features 9-16):\n');
    for j=9:16
        fprintf('  %2d: %s\n',j,S.featureNames{j});
    end
end
fprintf('DNN hidden layers: %s; ReLU; L2 lambda %.1g\n', ...
    mat2str(hiddenLayers),lambdaL2);
fprintf('Subjects, not overlapping windows, are CV units.\n');

%% Reproduce original training/previously-seen subject split
allSubjects = unique(subjectID);
allSubjects = allSubjects(randperm(numel(allSubjects)));
nTrainSubject = max(2,floor(0.75*numel(allSubjects)));
trainSubjects = sort(allSubjects(1:nTrainSubject));
previouslySeenSubjects = sort(allSubjects(nTrainSubject+1:end));
trainMask = ismember(subjectID,trainSubjects);
seenMask = ismember(subjectID,previouslySeenSubjects);
assert(numel(trainSubjects)>=3 && any(seenMask), ...
    'Need at least three train subjects and one previously seen subject.');

fprintf('\nTRAINING subjects: %s\n',mat2str(trainSubjects(:)'));
fprintf('PREVIOUSLY VIEWED subjects: %s\n', ...
    mat2str(previouslySeenSubjects(:)'));

%% Leave-one-training-subject-out cross-validation
nGroups=numel(trainSubjects);
foldBalanced=nan(nGroups,modelCount);
foldRaw=nan(nGroups,modelCount);

for fold=1:nGroups
    s=trainSubjects(fold);
    foldTrain=trainMask & subjectID~=s;
    foldVal=trainMask & subjectID==s;
    actual=Y(foldVal);
    assert(any(Y(foldTrain)==0) && any(Y(foldTrain)==1) && ...
           any(actual==0) && any(actual==1), ...
        'Each subject fold must include both motion classes.');

    for m=1:modelCount
        selected=featureCols{m};
        if usesDNN(m)
            % Same random seed per fold for DNN-8 versus DNN-16.
            rng(3310+fold);
            mdl=fitcnet(X(foldTrain,selected),Y(foldTrain), ...
                'LayerSizes',hiddenLayers, ...
                'Activations','relu', ...
                'Standardize',true, ...
                'Lambda',lambdaL2, ...
                'IterationLimit',maxIterations, ...
                'ClassNames',[0 1]);
        else
            mdl=fitcsvm(X(foldTrain,selected),Y(foldTrain), ...
                'KernelFunction','linear','Standardize',true, ...
                'ClassNames',[0 1]);
        end

        predicted=double(predict(mdl,X(foldVal,selected)));
        foldRaw(fold,m)=100*mean(predicted==actual);
        upRecall=mean(predicted(actual==0)==0);
        downRecall=mean(predicted(actual==1)==1);
        foldBalanced(fold,m)=50*(upRecall+downRecall);
    end

    fprintf('Subject %2d:',s);
    for m=1:modelCount
        fprintf(' %s %.1f%%;', ...
            strrep(modelNames{m},':',''),foldBalanced(fold,m));
    end
    fprintf('\n');
end

%% Report equal-subject-weight CV statistics
fprintf('\nTRAINING-SUBJECT LOSO BALANCED ACCURACY (exploratory)\n');
fprintf('%-23s %9s %9s %9s\n','Model','Mean','SD','Median');
for m=1:modelCount
    fprintf('%-23s %8.1f%% %8.1f%% %8.1f%%\n', ...
        modelNames{m},mean(foldBalanced(:,m)), ...
        std(foldBalanced(:,m)),median(foldBalanced(:,m)));
end
T = array2table([trainSubjects(:),foldBalanced], ...
    'VariableNames', ...
    {'Subject','SVM_2','SVM_8','SVM_16','DNN_8','DNN_16'});
disp(T);

fprintf('\nGAIN FROM EIGHT EXTRA PSD FEATURES (LOSO mean):\n');
fprintf('  Linear SVM: %+5.1f percentage points (16 - 8)\n', ...
    mean(foldBalanced(:,3)-foldBalanced(:,2)));
fprintf('  DNN:       %+5.1f percentage points (16 - 8)\n', ...
    mean(foldBalanced(:,5)-foldBalanced(:,4)));

figure('Name','LOSO: 2, 8, and 16 features');
plot(trainSubjects,foldBalanced,'o-','LineWidth',1.2);
hold on;
yline(50,'k--','50% balanced accuracy');
xlabel('Training subject ID');
ylabel('Balanced accuracy (%)');
title('Leave-one-training-subject-out validation');
ylim([0 105]); grid on;
legend(modelNames,'Location','best');

figure('Name','Mean balanced accuracy by subject');
bar(mean(foldBalanced,1));
hold on;
errorbar(1:modelCount,mean(foldBalanced,1), ...
    std(foldBalanced,0,1),'k.','LineWidth',1.2);
xticks(1:modelCount);
xticklabels(modelNames);
xtickangle(20);
ylabel('Balanced accuracy (%)');
title('Exploratory CV, equal-weighted subjects (bars: subject SD)');
ylim([0 105]); grid on;

%% Optional SECONDARY DIAGNOSTIC: these people were already inspected
if showPreviouslySeenSubjects
    fprintf('\nFitting all 10 training subjects for diagnostics...\n');
    actual=Y(seenMask);
    sid=subjectID(seenMask);
    diagnostics=nan(numel(previouslySeenSubjects),modelCount);
    windowAccuracy=nan(1,modelCount);
    allPred=cell(1,modelCount);

    for m=1:modelCount
        selected=featureCols{m};
        if usesDNN(m)
            rng(3399);
            mdl=fitcnet(X(trainMask,selected),Y(trainMask), ...
                'LayerSizes',hiddenLayers, ...
                'Activations','relu','Standardize',true, ...
                'Lambda',lambdaL2, ...
                'IterationLimit',maxIterations, ...
                'ClassNames',[0 1]);
        else
            mdl=fitcsvm(X(trainMask,selected),Y(trainMask), ...
                'KernelFunction','linear','Standardize',true, ...
                'ClassNames',[0 1]);
        end

        predicted=double(predict(mdl,X(seenMask,selected)));
        allPred{m}=predicted;
        windowAccuracy(m)=100*mean(predicted==actual);
        for j=1:numel(previouslySeenSubjects)
            idx=sid==previouslySeenSubjects(j);
            truth=actual(idx); yhat=predicted(idx);
            assert(any(truth==0) && any(truth==1), ...
                'Both classes required for subject balanced accuracy.');
            diagnostics(j,m)=50*( ...
                mean(yhat(truth==0)==0)+mean(yhat(truth==1)==1));
        end
    end

    fprintf('\nPREVIOUSLY SEEN SUBJECTS (NOT A FRESH TEST):\n');
    for m=1:modelCount
        fprintf('%-23s window accuracy %5.1f%%\n', ...
            modelNames{m},windowAccuracy(m));
    end
    disp(array2table([previouslySeenSubjects(:),diagnostics], ...
        'VariableNames', ...
        {'Subject','SVM_2','SVM_8','SVM_16','DNN_8','DNN_16'}));

    figure('Name','Previously viewed subjects: secondary diagnostic');
    bar(diagnostics,'grouped');
    xticks(1:numel(previouslySeenSubjects));
    xticklabels(string(previouslySeenSubjects));
    xlabel('Previously viewed subject ID');
    ylabel('Balanced accuracy (%)');
    title('Previously seen subjects: diagnostics, NOT fresh test');
    legend(modelNames,'Location','best'); grid on; ylim([0 105]);

    figure('Name','DNN-16: previously seen subjects');
    confusionchart(actual,allPred{5}, ...
        'Title','DNN-16: diagnostic only, subjects previously viewed');
    fprintf('WARNING: Already inspected subjects are NOT untouched tests.\n');
end

%% Keep local summary for comparison, no raw dataset saved
save(fullfile(scriptFolder,'ME3310_USC_HAD_DNN_Comparison.mat'), ...
    'trainSubjects','previouslySeenSubjects','modelNames', ...
    'foldBalanced','foldRaw','hiddenLayers','lambdaL2', ...
    'maxIterations','featureCols');
fprintf('\nEngineering question: do extra PSD bands improve cross-person accuracy?\n');
fprintf('The extra high-frequency features might also encode sensor noise.\n');
