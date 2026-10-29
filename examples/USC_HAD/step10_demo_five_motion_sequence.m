%% ME 3310 | Five-motion pseudo-stream classification DEMONSTRATION
% step10_demo_five_motion_sequence.m
%
% IMPORTANT:
% This script concatenates WINDOW FEATURES from SEPARATE ORIGINAL TRIALS.
% It is NOT a real continuous recording and does NOT contain measured
% physical transitions. The artificial boundaries are shown explicitly.
% It illustrates multi-class predictions and causal smoothing only.
%
% For REAL detection latency or false alarms, measure one continuous IMU
% signal and independently annotate when each motion starts/ends.
%
% Classes 0..4: Walk / Upstairs / Downstairs / Run / Stand.
% Uses the same 16 features and fixed training-person split as step09.

clear;clc;
scriptFolder=fileparts(mfilename('fullpath'));
if isempty(scriptFolder),scriptFolder=pwd;end
source=fullfile(scriptFolder,'ME3310_USC_HAD_FiveMotion_WindowFeatures.mat');
assert(isfile(source),'Run step07 and step08 first.');
S=load(source);
assert(isfield(S,'windowEndTime') && isfield(S,'recordID') && ...
    isfield(S,'subjectID') && isfield(S,'X') && ...
    isfield(S,'Y') && isfield(S,'classNames'), ...
    'Expected original trial window metadata and five-motion features.');
X=double(S.X);Y=double(S.Y(:));
subjectID=double(S.subjectID(:));
recordID=double(S.recordID(:));
windowEndTime=double(S.windowEndTime(:));
classNames=S.classNames;
classes=(0:4)';
assert(size(X,2)==16 && isequal(unique(Y),classes), ...
    'Expected five-class 16-feature data.');
assert(exist('fitcecoc','file')==2, ...
    'Statistics and Machine Learning Toolbox required.');

rng(3310);
subjects=unique(subjectID);
shuffled=subjects(randperm(numel(subjects)));
nTrain=max(2,floor(0.75*numel(shuffled)));
trainingSubjects=sort(shuffled(1:nTrain));
previousSubjects=sort(shuffled(nTrain+1:end));
trainMask=ismember(subjectID,trainingSubjects);

% Demonstrate on one person who is not in the model's training set.
demoSubject=previousSubjects(1);
sequence=[4 0 1 2 3 4]; % Stand -> Walk -> Up -> Down -> Run -> Stand
smoothingVotes=3;        % require 3 consecutive equal window predictions

learner=templateSVM('KernelFunction','linear','Standardize',true);
model=fitcecoc(X(trainMask,:),Y(trainMask), ...
    'Learners',learner,'Coding','onevsone', ...
    'ClassNames',classes,'Prior','uniform');

%% Select REAL windows from distinct original trials, in chosen sequence
streamX=zeros(0,16);
streamTruth=zeros(0,1);
streamTime=zeros(0,1);
seamTime=[];
selectedRecords=zeros(0,1);
offset=0;
hopSeconds=S.hopSeconds;

for segment=1:numel(sequence)
    thisClass=sequence(segment);
    subjectRecords=unique(recordID(subjectID==demoSubject & Y==thisClass));
    available=setdiff(subjectRecords,selectedRecords,'stable');
    assert(~isempty(available), ...
        'Not enough original trials for the requested demonstration.');
    selectedRecord=available(1);
    selectedRecords(end+1,1)=selectedRecord;

    idx=find(recordID==selectedRecord);
    [timeWithin,order]=sort(windowEndTime(idx));
    idx=idx(order);
    timeWithin=timeWithin-timeWithin(1);
    timeOnMontage=offset+timeWithin;

    if segment>1
        seamTime(end+1)=offset; %#ok<SAGROW>
    end
    streamX=[streamX;X(idx,:)]; %#ok<AGROW>
    streamTruth=[streamTruth;Y(idx)]; %#ok<AGROW>
    streamTime=[streamTime;timeOnMontage]; %#ok<AGROW>
    offset=timeOnMontage(end)+hopSeconds;
end

%% Window-level model decisions
[predicted,scores]=predict(model,streamX);
predicted=double(predicted(:));

% Simple causal 3-in-a-row rule:
% The state changes only when 3 successive window predictions agree.
% Unknown (-1) before the first stable state is established.
smoothed=-ones(numel(predicted),1);
state=-1;
for k=1:numel(predicted)
    if k>=smoothingVotes
        recent=predicted(k-smoothingVotes+1:k);
        if all(recent==recent(1))
            state=recent(1);
        end
    end
    smoothed(k)=state;
end

%% Visualize exact stitched-trial seams and predicted activities
figure('Name','FIVE motions: illustrative stitched-trial sequence');
tiledlayout(3,1,'TileSpacing','compact');
labels=[{'Unknown'},classNames];
plots={streamTruth,predicted,smoothed};
titles={'Source-trial labels (artificial montage)', ...
    'SVM-16: each window prediction', ...
    sprintf('Causal smoothing: %d consecutive matching windows', ...
        smoothingVotes)};
for p=1:3
    nexttile;hold on;
    stairs(streamTime,plots{p},'LineWidth',1.4);
    for seam=seamTime
        xline(seam,'k--','Artificial join');
    end
    ylim([-1.4 4.4]);
    yticks(-1:4);
    yticklabels(labels);
    ylabel('Activity');
    title(titles{p});grid on;
end
xlabel('Montage time (s), not a single physical measurement');
sgtitle(sprintf(['PSEUDO-STREAM only | held-out subject %d | ', ...
    'not measured motion transitions'],demoSubject));

fprintf('Demonstration subject: %d (excluded from SVM training).\n', ...
    demoSubject);
fprintf('Original trials used, in sequence: %s\n',mat2str(selectedRecords'));
fprintf('Window predictions in montage: %d\n',numel(predicted));
fprintf('Raw stitched-window agreement: %.1f%% (ILLUSTRATIVE ONLY)\n', ...
    100*mean(predicted==streamTruth));
fprintf(['WARNING: Trial joins are artificial, transitions are NOT ', ...
    'real sensor measurements.\n']);
fprintf(['Do not report stitched-stream accuracy or apparent transition ', ...
    'times as physical detection performance.\n']);

%% Next step for a real multi-motion detector:
% Record one uninterrupted IMU stream with time-stamped ground truth
% transition labels. Run 3-second causal windows, classify, smooth,
% and evaluate false alarms, missed transitions, and detection delay.
