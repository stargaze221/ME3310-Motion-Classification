function C = week11_02_feature_compare_demo(subjectNumber,featurePair)
% WEEK11_02_FEATURE_COMPARE_DEMO
% ME 3310 | Comparing MEASURED USC-HAD Upstairs and Downstairs signals.
%
% Requires the two original dataset-preparation scripts to have been run:
%   run("examples/USC_HAD/step01_prepare_up_down.m")
%   run("examples/USC_HAD/step03_extract_window_features.m")
%
% From the repository root, then:
%   addpath(fullfile(pwd,"examples","Student_IMU"))
%   C = week11_02_feature_compare_demo;            % Subject 2, [1 2]
%   C = week11_02_feature_compare_demo(2,[1 5]);    % choose two features
%   C = week11_02_feature_compare_demo("all",[1 2]);% all original subjects
%   disp(C.trialFeatures)
%
% Three figures:
%  1. Two one-dimensional feature comparisons, dots = ORIGINAL TRIALS
%  2. Two-dimensional feature scatter, dots = ORIGINAL TRIALS
%  3. Mean measured acceleration-magnitude PSD, equal weight per trial
%
% Feature values come from step03's existing 16-feature extraction;
% mean over overlapping windows WITHIN a trial (not independent trials).
% Same subject, both activities, as a controlled experimental comparison.
% No artificial signals, model fitting or optimal feature selection.
% Requires Signal Processing Toolbox for pwelch.

if nargin < 1 || isempty(subjectNumber), subjectNumber=2; end
if nargin < 2 || isempty(featurePair),featurePair=[1 2];end

scriptFolder=fileparts(mfilename('fullpath'));
dataFolder=fullfile(fileparts(scriptFolder),"USC_HAD");
rawFile=fullfile(dataFolder,"ME3310_USC_HAD_UpDown.mat");
featureFile=fullfile(dataFolder,"ME3310_USC_HAD_WindowFeatures.mat");
assert(isfile(rawFile),[ ...
    'Real USC-HAD trials not found. Run ' ...
    'run("examples/USC_HAD/step01_prepare_up_down.m") first.']);
assert(isfile(featureFile),[ ...
    'Window features not found. Run ' ...
    'run("examples/USC_HAD/step03_extract_window_features.m") first.']);
assert(exist('pwelch','file')==2, ...
    'Signal Processing Toolbox is required for the PSD comparison.');

D=load(rawFile,'records','fs','classNames');
F=load(featureFile,'X','Y','featureNames','recordID', ...
    'subjectID','trialID','fs','windowSeconds','hopSeconds');
assert(D.fs==F.fs,'Raw and feature sampling rates do not match.');
assert(size(F.X,2)==numel(F.featureNames), ...
    'Feature matrix and feature names have different widths.');
assert(isnumeric(featurePair) && numel(featurePair)==2 && ...
    all(isfinite(featurePair)) && all(featurePair==round(featurePair)) && ...
    all(featurePair>=1) && all(featurePair<=size(F.X,2)) && ...
    featurePair(1)~=featurePair(2), ...
    'Select two distinct valid feature-column indices, such as [1 2].');

if (isstring(subjectNumber) || ischar(subjectNumber)) && ...
        strcmpi(string(subjectNumber),"all")
    eligible=true(size(F.Y));
    selectionLabel="All subjects";
else
    assert(isnumeric(subjectNumber) && isscalar(subjectNumber) && ...
        isfinite(subjectNumber) && subjectNumber==round(subjectNumber), ...
        'Use an integer subject number, or "all".');
    eligible=F.subjectID==subjectNumber;
    selectionLabel=sprintf('Subject %d',subjectNumber);
end
rids=unique(F.recordID(eligible));
rids=rids(rids>=1 & rids<=numel(D.records));
assert(~isempty(rids),'No measurements available for selected subject.');
nR=numel(rids);
nFeat=size(F.X,2);
Z=zeros(nR,nFeat);
labels=zeros(nR,1);
subjects=zeros(nR,1);
trialNumbers=zeros(nR,1);
windowCounts=zeros(nR,1);
for i=1:nR
    rid=rids(i);
    mask=F.recordID==rid; % ALL overlapping windows of ONE original trial
    assert(any(mask),'Feature extraction is missing an original trial.');
    R=D.records(rid);
    assert(all(F.Y(mask)==R.label) && ...
           all(F.subjectID(mask)==R.subject) && ...
           all(F.trialID(mask)==R.trial), ...
        'Original trial identity does not match feature records.');
    Z(i,:)=mean(F.X(mask,:),1);
    labels(i)=R.label;
    subjects(i)=R.subject;
    trialNumbers(i)=R.trial;
    windowCounts(i)=nnz(mask);
end
assert(all(ismember([0 1],unique(labels))), ...
    'Both Upstairs and Downstairs original trials are required.');

C=struct;
C.subjectSelection=selectionLabel;
C.featurePair=featurePair;
C.featureNames=F.featureNames;
C.windowSeconds=F.windowSeconds;
C.hopSeconds=F.hopSeconds;
C.trialFeatures=table(subjects,trialNumbers,labels,windowCounts, ...
    Z(:,featurePair(1)),Z(:,featurePair(2)), ...
    'VariableNames',{'Subject','Trial','ClassID','WindowCount', ...
    'FeatureX','FeatureY'});
C.trialMeanOfAllFeatures=Z;
C.classNames=D.classNames;
fprintf('\nMeasured USC-HAD feature comparison: %s\n',selectionLabel);
fprintf('Window size %.1fs, hop %.1fs; each point = ONE original trial.\n', ...
    F.windowSeconds,F.hopSeconds);
fprintf('Feature %d: %s\nFeature %d: %s\n',featurePair(1), ...
    F.featureNames{featurePair(1)},featurePair(2), ...
    F.featureNames{featurePair(2)});
fprintf('Original Upstairs trials: %d; Downstairs trials: %d.\n', ...
    nnz(labels==0),nnz(labels==1));
disp(C.trialFeatures);
fprintf(['Window counts describe repeated views of a trial; ' ...
    'they are NOT independent experimental repetitions.\n']);

colors=[0.12 0.38 0.68;0.82 0.39 0.17];
classNames=D.classNames;

%% FIGURE 1: two distinct features, one point per independent trial
figure('Name','USC-HAD | Trial-level 1D feature comparisons');
tiledlayout(1,2,'TileSpacing','compact');
for k=1:2
    idx=featurePair(k);
    nexttile;hold on;
    for ci=1:2
        v=Z(labels==ci-1,idx);
        % Horizontal offsets are display-only: NOT data perturbations.
        xPosition=ci+0.11*linspace(-1,1,numel(v))';
        scatter(xPosition,v,45,'MarkerFaceColor',colors(ci,:), ...
            'MarkerEdgeColor','w','DisplayName',classNames{ci});
        errorbar(ci,mean(v),std(v,0), ...
            'k','LineWidth',1.8,'CapSize',11,'HandleVisibility','off');
        plot(ci,mean(v),'kd','MarkerFaceColor','k', ...
            'MarkerSize',7,'HandleVisibility','off');
    end
    xticks([1 2]);xticklabels(classNames);
    xlim([0.55 2.45]);grid on;
    ylabel(F.featureNames{idx},'Interpreter','none');
    title(sprintf('Feature %d: measured trial means',idx));
    if k==1,legend('Location','best');end
end
sgtitle(sprintf('%s | Dots: original trials; diamond: mean, bars: +/- 1 SD', ...
    selectionLabel));

%% FIGURE 2: pick two features, not a fitted classification boundary
figure('Name','USC-HAD | Trial-level 2D feature space');hold on;
for ci=1:2
    mask=labels==ci-1;
    scatter(Z(mask,featurePair(1)),Z(mask,featurePair(2)), ...
        60,'filled','MarkerFaceColor',colors(ci,:), ...
        'MarkerEdgeColor','w','DisplayName',classNames{ci});
    % Actual original trial IDs. For "all", many labels would clutter.
    if nR<=15
        rows=find(mask);
        for j=1:numel(rows)
            ii=rows(j);
            text(Z(ii,featurePair(1)),Z(ii,featurePair(2)), ...
                sprintf('  T%d',trialNumbers(ii)), ...
                'FontSize',8,'Color',colors(ci,:));
        end
    end
end
xlabel(F.featureNames{featurePair(1)},'Interpreter','none');
ylabel(F.featureNames{featurePair(2)},'Interpreter','none');
title(sprintf('%s | one point per measured trial',selectionLabel));
grid on;legend('Location','best');

%% FIGURE 3: average measured PSD, within windows -> trial -> class
% PSD settings MATCH step03: 3-s analysis window, 2-s Hamming Welch
% segment, 50% overlap and max(512, next power of 2) FFT size.
fs=double(D.fs);
N=round(F.windowSeconds*fs);
H=round(F.hopSeconds*fs);
welchN=min(N,round(2*fs));
welchWindow=hamming(welchN);
welchOverlap=floor(welchN/2);
nfft=max(512,2^nextpow2(welchN));
nF=floor(nfft/2)+1;
trialPSD=nan(nF,nR);
f=[];
for i=1:nR
    R=D.records(rids(i));
    a=R.acc;
    total=zeros(nF,1);
    count=0;
    for first=1:H:(size(a,1)-N+1)
        aw=a(first:first+N-1,:);
        if any(~isfinite(aw(:))),continue;end
        aMag=sqrt(sum(aw.^2,2));
        dynamic=aMag-mean(aMag);
        [P,f]=pwelch(dynamic,welchWindow,welchOverlap,nfft,fs);
        total=total+P;
        count=count+1;
    end
    if count>0
        trialPSD(:,i)=total/count;
    end
end
assert(~isempty(f),'No valid PSD window from original trials.');
meanPSD=nan(nF,2);
for ci=1:2
    valid=trialPSD(:,labels==ci-1);
    assert(all(any(isfinite(valid),1)), ...
        'A selected original trial has no valid PSD windows.');
    meanPSD(:,ci)=mean(valid,2); % EACH original trial receives equal weight
end
C.psdFrequencyHz=f;
C.meanPsdByClass=meanPSD;
C.psdByTrial=trialPSD;

figure('Name','USC-HAD | Equal-trial-weight mean PSD');hold on;
plot(f,meanPSD(:,1),'LineWidth',1.6,'Color',colors(1,:), ...
    'DisplayName',classNames{1});
plot(f,meanPSD(:,2),'LineWidth',1.6,'Color',colors(2,:), ...
    'DisplayName',classNames{2});
xline(0.8,'k--','0.8 Hz','HandleVisibility','off');
xline(3.0,'k--','3 Hz','HandleVisibility','off');
xlim([0 min(10,fs/2)]);grid on;
xlabel('Frequency (Hz)');
ylabel('Acceleration magnitude PSD (g^2/Hz)');
title(sprintf('%s | same PSD method, equal weight per trial', ...
    selectionLabel));
legend('Location','best');

fprintf(['\nAsk: Which feature shows better between-motion separation ' ...
    'relative to trial-to-trial variation?\n']);
fprintf(['Is that difference also visible in the 0.8-3 Hz PSD range? ' ...
    'Would another person behave the same way?\n']);
fprintf('No classifier was trained: these are exploratory measurements.\n');
end
