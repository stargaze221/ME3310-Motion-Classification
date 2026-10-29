%% ME 3310 | FIVE-class motion recognition: SVM 8 vs 16 features
% step09_train_five_motion_svm.m
%
% Classes: Walking Forward (0), Walking Upstairs (1),
%          Walking Downstairs (2), Running Forward (3), Standing (4)
%
% Uses linear one-versus-one ECOC SVM (fitcecoc) for FIVE classes.
% Compare original 8 versus expanded 16 engineered IMU features.
% Exactly the same subject split as the earlier two-motion experiment.
%
% Training subjects: Leave-One-Subject-Out CV, one whole person per fold.
% Window score:  mean of FIVE class-specific recalls per subject.
% Trial score:   majority vote over ALL windows in ONE original trial,
%                then mean of FIVE trial-class recalls per subject.
% Subject summary: each person contributes ONE equally weighted score.
%
% A trial has only ONE true activity. This measures OFFLINE activity
% classification, NOT the onset/offset of a naturally changing motion.
% The four previously viewed people are a diagnostic, NOT a fresh test.

clear; clc;
scriptFolder=fileparts(mfilename('fullpath'));
if isempty(scriptFolder),scriptFolder=pwd;end
file=fullfile(scriptFolder,'ME3310_USC_HAD_FiveMotion_WindowFeatures.mat');
assert(isfile(file), ...
    'Missing five-motion features: run step07 and step08 first.');
S=load(file);
assert(isfield(S,'X') && isfield(S,'Y') && ...
    isfield(S,'subjectID') && isfield(S,'recordID') && ...
    isfield(S,'classNames'), 'Incomplete feature MAT file.');
X=double(S.X);
Y=double(S.Y(:));
subjectID=double(S.subjectID(:));
recordID=double(S.recordID(:));
classNames=S.classNames;
classes=(0:4)';
nClasses=numel(classes);
assert(size(X,2)==16 && numel(classNames)==5, ...
    'Expected 16 features and five classes.');
assert(numel(Y)==size(X,1) && ...
    numel(subjectID)==numel(Y) && numel(recordID)==numel(Y), ...
    'Inconsistent feature and trial metadata lengths.');
assert(all(isfinite(X(:))) && isequal(unique(Y),classes), ...
    'Invalid features or class labels.');
assert(exist('fitcecoc','file')==2 && exist('templateSVM','file')==2, ...
    'Statistics and Machine Learning Toolbox required.');

%% Fixed model options; no tuning on previously viewed subjects
modelNames={'Linear SVM: 8 features','Linear SVM: 16 features'};
featureCols={1:8,1:16};
nModels=numel(modelNames);
showPreviouslySeenSubjects=true;
svmLearner=templateSVM('KernelFunction','linear','Standardize',true);

rng(3310);
subjects=unique(subjectID);
shuffled=subjects(randperm(numel(subjects)));
nTrain=max(2,floor(0.75*numel(shuffled)));
trainSubjects=sort(shuffled(1:nTrain));
previousSubjects=sort(shuffled(nTrain+1:end));
trainMask=ismember(subjectID,trainSubjects);
previousMask=ismember(subjectID,previousSubjects);
assert(numel(trainSubjects)>=3,'Not enough subjects for CV.');

fprintf('Five classes (USC-HAD official activity labels):\n');
activityIDs=[1 4 5 6 9];
for c=1:nClasses
    fprintf('  %d: %-22s (original ID %d)\n', ...
        c-1,classNames{c},activityIDs(c));
end
fprintf('TRAINING subjects: %s\n',mat2str(trainSubjects(:)'));
fprintf('PREVIOUSLY SEEN subjects: %s\n', ...
    mat2str(previousSubjects(:)'));
fprintf('3-s windows / 0.5-s hop; original recordID = one trajectory.\n');

%% Cross-validation: hold out a WHOLE SUBJECT each time
nGroups=numel(trainSubjects);
cvWindowBalanced=nan(nGroups,nModels);
cvTrialBalanced=nan(nGroups,nModels);
cvWindowRaw=nan(nGroups,nModels);
cvTrialRaw=nan(nGroups,nModels);
cvTrialCount=zeros(nGroups,1);
cvClassRecall=nan(nGroups,nClasses,nModels);
cvWindowTruth=cell(nModels,1);
cvWindowPred=cell(nModels,1);
cvTrialTruth=cell(nModels,1);
cvTrialPred=cell(nModels,1);

for fold=1:nGroups
    s=trainSubjects(fold);
    trainIdx=trainMask & subjectID~=s;
    valIdx=trainMask & subjectID==s;
    truth=Y(valIdx);
    records=recordID(valIdx);
    cvTrialCount(fold)=numel(unique(records));

    assert(isequal(unique(Y(trainIdx)),classes) && ...
           isequal(unique(truth),classes), ...
           'Each subject fold must contain all five classes.');
    fprintf('\nValidation Subject %d: %d original trials\n', ...
        s,cvTrialCount(fold));

    for m=1:nModels
        cols=featureCols{m};
        % Uniform class prior reduces bias from different recording lengths.
        mdl=fitcecoc(X(trainIdx,cols),Y(trainIdx), ...
            'Learners',svmLearner,'Coding','onevsone', ...
            'ClassNames',classes,'Prior','uniform');

        [predicted,scores]=predict(mdl,X(valIdx,cols));
        predicted=double(predicted(:));
        cvWindowRaw(fold,m)=100*mean(predicted==truth);

        recall=zeros(nClasses,1);
        for c=1:nClasses
            recall(c)=mean(predicted(truth==classes(c))==classes(c));
        end
        cvClassRecall(fold,:,m)=recall;
        cvWindowBalanced(fold,m)=100*mean(recall);

        % ORIGINAL trial ID is the grouping key.
        [tTruth,tPred]=aggregateOriginalTrials( ...
            truth,predicted,scores,records,classes);
        cvTrialRaw(fold,m)=100*mean(tTruth==tPred);
        trialRecall=zeros(nClasses,1);
        for c=1:nClasses
            assert(any(tTruth==classes(c)), ...
                'Subject lacks a trial in one of the five classes.');
            trialRecall(c)=mean(tPred(tTruth==classes(c))==classes(c));
        end
        cvTrialBalanced(fold,m)=100*mean(trialRecall);

        cvWindowTruth{m}=[cvWindowTruth{m};truth];
        cvWindowPred{m}=[cvWindowPred{m};predicted];
        cvTrialTruth{m}=[cvTrialTruth{m};tTruth];
        cvTrialPred{m}=[cvTrialPred{m};tPred];

        fprintf('  %-24s window %5.1f%% | trial %5.1f%% (balanced)\n', ...
            modelNames{m},cvWindowBalanced(fold,m), ...
            cvTrialBalanced(fold,m));
    end
end

%% Equal-subject-weight evaluation summary
fprintf('\nFIVE-CLASS SUBJECT-HELD-OUT CV SUMMARY\n');
fprintf('Metric: macro recall (balanced accuracy), averaged over subjects.\n');
fprintf('%-25s %9s %9s %9s %9s\n', ...
    'Model','Window','Trial','Trial SD','Trials');
for m=1:nModels
    fprintf('%-25s %8.1f%% %8.1f%% %8.1f%% %9d\n', ...
        modelNames{m},mean(cvWindowBalanced(:,m)), ...
        mean(cvTrialBalanced(:,m)), ...
        std(cvTrialBalanced(:,m)),sum(cvTrialCount));
end
disp(array2table([trainSubjects(:),cvTrialCount,cvTrialBalanced], ...
    'VariableNames',{'Subject','OriginalTrials','SVM8','SVM16'}));

fprintf('\nPer-class WINDOW recall (subject-weighted, SVM-16):\n');
for c=1:nClasses
    fprintf('  %-22s %5.1f%%\n', ...
        classNames{c},100*mean(cvClassRecall(:,c,2)));
end

figure('Name','Five-motion SVM: window versus original trial');
bar([mean(cvWindowBalanced,1)',mean(cvTrialBalanced,1)']);
xticks(1:nModels); xticklabels(modelNames);
ylabel('Macro recall / balanced accuracy (%)');
ylim([0 105]); grid on;
legend({'Windows','Original trials (majority vote)'},'Location','best');
title('Five-class LOSO CV: subject-weighted scores');

figure('Name','Five-motion: trial accuracy by training subject');
bar(cvTrialBalanced,'grouped');
xticks(1:nGroups); xticklabels(string(trainSubjects));
xlabel('Validation subject ID');
ylabel('Trial-level balanced accuracy (%)');
ylim([0 105]); grid on; legend(modelNames,'Location','best');

figure('Name','Five-motion trial confusion matrix: SVM-16 CV');
trueCategories=categorical(cvTrialTruth{2},classes,classNames);
predCategories=categorical(cvTrialPred{2},classes,classNames);
confusionchart(trueCategories,predCategories, ...
    'Title','Original trial predictions: SVM-16 LOSO CV', ...
    'RowSummary','row-normalized');

%% Optional diagnostic on four previously inspected subjects
if showPreviouslySeenSubjects
    fprintf('\nDIAGNOSTIC on PREVIOUSLY SEEN subjects (NOT A FRESH TEST):\n');
    prevTruth=Y(previousMask);
    prevRecords=recordID(previousMask);
    prevSid=subjectID(previousMask);
    diagWindow=zeros(numel(previousSubjects),nModels);
    diagTrial=zeros(numel(previousSubjects),nModels);
    diagTrialTruth=cell(nModels,1);
    diagTrialPred=cell(nModels,1);

    for m=1:nModels
        cols=featureCols{m};
        mdl=fitcecoc(X(trainMask,cols),Y(trainMask), ...
            'Learners',svmLearner,'Coding','onevsone', ...
            'ClassNames',classes,'Prior','uniform');
        [predicted,scores]=predict(mdl,X(previousMask,cols));
        predicted=double(predicted(:));
        [diagTrialTruth{m},diagTrialPred{m}]=aggregateOriginalTrials( ...
            prevTruth,predicted,scores,prevRecords,classes);
        for j=1:numel(previousSubjects)
            mask=prevSid==previousSubjects(j);
            subjectTruth=prevTruth(mask);
            subjectPred=predicted(mask);
            wRecall=zeros(nClasses,1);
            for c=1:nClasses
                assert(any(subjectTruth==classes(c)), ...
                    'Diagnostic subject lacks an activity.');
                wRecall(c)=mean( ...
                    subjectPred(subjectTruth==classes(c))==classes(c));
            end
            diagWindow(j,m)=100*mean(wRecall);
            [tTruth,tPred]=aggregateOriginalTrials( ...
                subjectTruth,subjectPred,scores(mask,:), ...
                prevRecords(mask),classes);
            tRecall=zeros(nClasses,1);
            for c=1:nClasses
                tRecall(c)=mean(tPred(tTruth==classes(c))==classes(c));
            end
            diagTrial(j,m)=100*mean(tRecall);
        end
        fprintf('%-25s window %5.1f%% | trial %5.1f%% (subject-averaged)\n', ...
            modelNames{m},mean(diagWindow(:,m)),mean(diagTrial(:,m)));
    end
    disp(array2table([previousSubjects(:),diagTrial], ...
        'VariableNames',{'Subject','SVM8','SVM16'}));

    figure('Name','Five-motion SVM-16: previously viewed subjects');
    confusionchart(categorical(diagTrialTruth{2},classes,classNames), ...
        categorical(diagTrialPred{2},classes,classNames), ...
        'Title','Previously viewed people: trial diagnostic, NOT a fresh test');
    fprintf('WARNING: Reused subjects cannot supply a fresh test score.\n');
end

%% Save a small summary without redistributing any raw IMU data
summaryPath=fullfile(scriptFolder, ...
    'ME3310_USC_HAD_FiveMotion_Comparison.mat');
save(summaryPath,'classNames','classes','trainSubjects', ...
    'previousSubjects','modelNames','cvWindowBalanced', ...
    'cvTrialBalanced','cvTrialCount','cvClassRecall', ...
    'cvWindowTruth','cvWindowPred','cvTrialTruth','cvTrialPred', ...
    'diagWindow','diagTrial');
fprintf('\nSaved summary: %s\n',summaryPath);
fprintf('Trial voting uses whole single-activity recordings (OFFLINE).\n');

%% Convert window predictions into ONE label per REAL original trial
function [tTruth,tPred,tIDs]=aggregateOriginalTrials( ...
        truth,pred,scores,recordIDs,classes)
% The original trial is labeled with just ONE activity.
% Vote ties are resolved by average class score (higher = better).
% ECOC scores from predict are class-wise negative losses, not calibrated
% class probabilities; they are used only as tie breakers here.
    truth=double(truth(:));pred=double(pred(:));
    recordIDs=double(recordIDs(:));
    assert(numel(truth)==numel(pred) && ...
        numel(truth)==numel(recordIDs) && ...
        size(scores,1)==numel(truth) && ...
        size(scores,2)==numel(classes), ...
        'Misaligned trial inputs.');
    tIDs=unique(recordIDs);
    tTruth=zeros(numel(tIDs),1);
    tPred=zeros(numel(tIDs),1);
    for k=1:numel(tIDs)
        idx=recordIDs==tIDs(k);
        uniqueTruth=unique(truth(idx));
        assert(numel(uniqueTruth)==1, ...
            'One original trial unexpectedly has multiple true labels.');
        tTruth(k)=uniqueTruth;
        counts=zeros(numel(classes),1);
        for c=1:numel(classes)
            counts(c)=sum(pred(idx)==classes(c));
        end
        tied=find(counts==max(counts));
        if numel(tied)==1
            selected=tied;
        else
            meanScore=mean(scores(idx,:),1);
            [~,j]=max(meanScore(tied));
            selected=tied(j); % first class if also score-tied
        end
        tPred(k)=classes(selected);
    end
end
