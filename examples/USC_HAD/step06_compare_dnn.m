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
% Report BOTH window-level and trajectory-level balanced accuracy.
% One trajectory is one ORIGINAL USC-HAD trial (recordID).
% Each trajectory receives ONE label via majority vote over its windows.
% Exact vote ties are resolved by mean classifier score differences.
% Each subject is weighted equally in the summary.
% Overlapping 3-s/0.5-s windows are NOT independent observations.
% Full-trajectory voting is OFFLINE (not real-time transition detection).
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
    isfield(S,'subjectID') && isfield(S,'recordID'), ...
    'Feature file must include X, Y, subjectID and recordID.');
X = double(S.X);
Y = double(S.Y(:));
subjectID = double(S.subjectID(:));
recordID = double(S.recordID(:));
assert(size(X,2)==16, ...
    ['Expected 16 features. Rerun the UPDATED ' ...
     'step03_extract_window_features.m after git pull.']);
assert(numel(Y)==size(X,1) && numel(subjectID)==numel(Y) && ...
    numel(recordID)==numel(Y) && all(isfinite(recordID)), ...
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
fprintf('Subject-wise CV; original trial (recordID) is the trajectory.\n');
fprintf('Trajectory vote: majority of window labels, score tie-break.\n');

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
foldBalanced=nan(nGroups,modelCount);     % window-level balanced accuracy
foldRaw=nan(nGroups,modelCount);          % window-level raw accuracy
foldTrajectoryBalanced=nan(nGroups,modelCount);
foldTrajectoryRaw=nan(nGroups,modelCount);
foldVoteTies=zeros(nGroups,modelCount);
foldTrialCount=zeros(nGroups,1);
cvTrialTruth=cell(1,modelCount);
cvTrialPrediction=cell(1,modelCount);

for fold=1:nGroups
    s=trainSubjects(fold);
    foldTrain=trainMask & subjectID~=s;
    foldVal=trainMask & subjectID==s;
    actual=Y(foldVal);
    trialRecords=recordID(foldVal);
    foldTrialCount(fold)=numel(unique(trialRecords));
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

        [predicted,score]=predict(mdl,X(foldVal,selected));
        predicted=double(predicted(:));
        foldRaw(fold,m)=100*mean(predicted==actual);
        upRecall=mean(predicted(actual==0)==0);
        downRecall=mean(predicted(actual==1)==1);
        foldBalanced(fold,m)=50*(upRecall+downRecall);

        % Aggregate window decisions per ORIGINAL continuous trial.
        % The trial label is constant; validate this assumption.
        [foldTrajectoryBalanced(fold,m), ...
            foldTrajectoryRaw(fold,m), ...
            trialTruth,trialPrediction,~,foldVoteTies(fold,m)] = ...
            trajectoryVoteMetrics(actual,predicted,score,trialRecords);
        cvTrialTruth{m}=[cvTrialTruth{m};trialTruth];
        cvTrialPrediction{m}=[cvTrialPrediction{m};trialPrediction];
    end

    fprintf('Subject %2d (%d trials):\n',s,foldTrialCount(fold));
    for m=1:modelCount
        fprintf('  %-22s window %5.1f%% | trajectory %5.1f%% (balanced)\n', ...
            modelNames{m},foldBalanced(fold,m), ...
            foldTrajectoryBalanced(fold,m));
    end
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

fprintf('\nTRAINING-SUBJECT LOSO TRAJECTORY BALANCED ACCURACY:\n');
fprintf('%-23s %9s %9s %9s\n','Model','Mean','SD','Median');
for m=1:modelCount
    fprintf('%-23s %8.1f%% %8.1f%% %8.1f%%\n', ...
        modelNames{m},mean(foldTrajectoryBalanced(:,m)), ...
        std(foldTrajectoryBalanced(:,m)), ...
        median(foldTrajectoryBalanced(:,m)));
end
TT = array2table([trainSubjects(:),foldTrialCount, ...
    foldTrajectoryBalanced], ...
    'VariableNames', ...
    {'Subject','Trials','SVM_2','SVM_8','SVM_16','DNN_8','DNN_16'});
disp(TT);
fprintf('CV voting ties across all folds, each model:\n');
for m=1:modelCount
    fprintf('  %-22s %d ties\n',modelNames{m},sum(foldVoteTies(:,m)));
end

fprintf('\nGAIN FROM EIGHT EXTRA PSD FEATURES (LOSO mean):\n');
fprintf('  Linear SVM: %+5.1f percentage points (16 - 8)\n', ...
    mean(foldBalanced(:,3)-foldBalanced(:,2)));
fprintf('  DNN:       %+5.1f percentage points (16 - 8)\n', ...
    mean(foldBalanced(:,5)-foldBalanced(:,4)));
fprintf('TRAJECTORY-LEVEL CHANGE FROM 8 TO 16 FEATURES:\n');
fprintf('  Linear SVM: %+5.1f percentage points\n', ...
    mean(foldTrajectoryBalanced(:,3)-foldTrajectoryBalanced(:,2)));
fprintf('  DNN:       %+5.1f percentage points\n', ...
    mean(foldTrajectoryBalanced(:,5)-foldTrajectoryBalanced(:,4)));

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

%% Figure: paired window versus trajectory results across all five models
figure('Name','CV: window versus original-trial decisions');
bar([mean(foldBalanced,1)',mean(foldTrajectoryBalanced,1)']);
xticks(1:modelCount); xticklabels(modelNames);
xtickangle(20); ylim([0 105]); grid on;
ylabel('Balanced accuracy, mean across subjects (%)');
title('Subject-held-out CV: windows vs original trajectories');
legend({'3-s windows','Entire trajectory majority vote'}, ...
    'Location','best');

figure('Name','SVM-16: trajectory confusion matrix (cross-validation)');
confusionchart(cvTrialTruth{3},cvTrialPrediction{3}, ...
    'Title','SVM-16: original-trial predictions across LOSO folds');

%% Optional SECONDARY DIAGNOSTIC: these people were already inspected
if showPreviouslySeenSubjects
    fprintf('\nFitting all 10 training subjects for diagnostics...\n');
    actual=Y(seenMask);
    sid=subjectID(seenMask);
    diagnostics=nan(numel(previouslySeenSubjects),modelCount);
    trialDiagnostics=nan(numel(previouslySeenSubjects),modelCount);
    windowAccuracy=nan(1,modelCount);
    trajectoryAccuracy=nan(1,modelCount);
    allPred=cell(1,modelCount);
    seenRecords=recordID(seenMask);
    trialTruthDiag=cell(1,modelCount);
    trialPredDiag=cell(1,modelCount);

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

        [predicted,score]=predict(mdl,X(seenMask,selected));
        predicted=double(predicted(:));
        allPred{m}=predicted;
        windowAccuracy(m)=100*mean(predicted==actual);
        [~,trajectoryAccuracy(m),trialTruthDiag{m}, ...
            trialPredDiag{m}] = ...
            trajectoryVoteMetrics(actual,predicted,score,seenRecords);
        for j=1:numel(previouslySeenSubjects)
            idx=sid==previouslySeenSubjects(j);
            truth=actual(idx); yhat=predicted(idx);
            assert(any(truth==0) && any(truth==1), ...
                'Both classes required for subject balanced accuracy.');
            diagnostics(j,m)=50*( ...
                mean(yhat(truth==0)==0)+mean(yhat(truth==1)==1));
            [trialDiagnostics(j,m),~] = trajectoryVoteMetrics( ...
                truth,yhat,score(idx,:),seenRecords(idx));
        end
    end

    fprintf('\nPREVIOUSLY SEEN SUBJECTS (NOT A FRESH TEST):\n');
    fprintf('%-23s %11s %13s\n','Model','Window raw','Trajectory raw');
    for m=1:modelCount
        fprintf('%-23s %10.1f%% %12.1f%%\n', ...
            modelNames{m},windowAccuracy(m),trajectoryAccuracy(m));
    end
    fprintf('PER-SUBJECT WINDOW BALANCED ACCURACY (diagnostic):\n');
    disp(array2table([previouslySeenSubjects(:),diagnostics], ...
        'VariableNames', ...
        {'Subject','SVM_2','SVM_8','SVM_16','DNN_8','DNN_16'}));
    fprintf('PER-SUBJECT TRAJECTORY BALANCED ACCURACY (diagnostic):\n');
    disp(array2table([previouslySeenSubjects(:),trialDiagnostics], ...
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
        'Title','DNN-16: window diagnostic, subjects previously viewed');

    figure('Name','SVM-16: previously seen trial predictions');
    confusionchart(trialTruthDiag{3},trialPredDiag{3}, ...
        'Title','SVM-16: trajectory diagnostic, NOT fresh test');
    fprintf('WARNING: Already inspected subjects are NOT untouched tests.\n');
end

%% Keep local summary for comparison, no raw dataset saved
save(fullfile(scriptFolder,'ME3310_USC_HAD_DNN_Comparison.mat'), ...
    'trainSubjects','previouslySeenSubjects','modelNames', ...
    'foldBalanced','foldRaw','foldTrajectoryBalanced', ...
    'foldTrajectoryRaw','foldTrialCount','foldVoteTies', ...
    'cvTrialTruth','cvTrialPrediction', ...
    'hiddenLayers','lambdaL2','maxIterations','featureCols');
fprintf('\nEngineering question: is each full trajectory correctly classified?\n');
fprintf('Whole-trial voting is offline and does NOT measure transition delay.\n');

%% Local function: ONE decision for EACH original continuous trial
function [balanced,raw,trialTruth,trialPred,trialIDs,tieCount] = ...
        trajectoryVoteMetrics(truth,pred,score,records)
% Input vectors describe the windows from one or more complete trials.
% For a tie in window vote counts, use the mean class-score difference.
% No trial is ever split across training and evaluation here.
    truth=double(truth(:));
    pred=double(pred(:));
    records=double(records(:));
    assert(numel(truth)==numel(pred) && ...
        numel(pred)==numel(records) && ...
        size(score,1)==numel(pred) && size(score,2)==2, ...
        'Window labels, record IDs and scores must be aligned.');
    assert(all(ismember(pred,[0 1])), ...
        'Only binary 0/1 predictions are supported.');

    trialIDs=unique(records);
    nTrials=numel(trialIDs);
    trialTruth=zeros(nTrials,1);
    trialPred=zeros(nTrials,1);
    tieCount=0;

    for k=1:nTrials
        mask=(records==trialIDs(k));
        labels=unique(truth(mask));
        assert(numel(labels)==1, ...
            'An original trajectory cannot contain multiple labels.');
        trialTruth(k)=labels(1);
        positives=sum(pred(mask)==1);
        negatives=sum(pred(mask)==0);

        if positives>negatives
            trialPred(k)=1;
        elseif negatives>positives
            trialPred(k)=0;
        else
            tieCount=tieCount+1;
            % score columns follow ClassNames [0 1].
            % An exact secondary tie defaults deterministically to 0.
            scoreDifference=mean(score(mask,2)-score(mask,1));
            trialPred(k)=double(scoreDifference>0);
        end
    end

    assert(any(trialTruth==0) && any(trialTruth==1), ...
        'Both activity classes are needed for balanced accuracy.');
    raw=100*mean(trialPred==trialTruth);
    upRecall=mean(trialPred(trialTruth==0)==0);
    downRecall=mean(trialPred(trialTruth==1)==1);
    balanced=50*(upRecall+downRecall);
end

