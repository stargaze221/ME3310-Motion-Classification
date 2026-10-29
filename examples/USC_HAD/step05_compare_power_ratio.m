%% ME 3310 | Test the spectral power-ratio hypothesis
% step05_compare_power_ratio.m
%
% Hypothesis: relative high/low frequency content is more transferable
% across people than absolute acceleration power.
%
% A) STD alone
% B) STD + low-band power (0.8-3 Hz; existing baseline)
% C) STD + high-band power (5-8 Hz)
% D) STD + log10(high-band / low-band power) in dB
%
% Evaluation:
% - Same 3.0-s window and 0.5-s hop for EVERY candidate.
% - Only the original TRAINING subjects are used for selecting features.
% - Leave-one-training-subject-out CV: each subject contributes one score.
% - Balanced accuracy avoids rewards for class imbalance.
% - The old four held-out subjects are OPTIONAL diagnostic data; because
%   their prior results were already inspected, they are NOT fresh tests.
%
% Requires Signal Processing Toolbox: pwelch
% Requires Statistics and Machine Learning Toolbox: fitcsvm
% MATLAB R2023b or newer recommended.

clear; clc; close all;
scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder), scriptFolder=pwd; end
dataPath = fullfile(scriptFolder,'ME3310_USC_HAD_UpDown.mat');
assert(isfile(dataPath), ...
    'Data missing. Run step01_prepare_up_down.m first.');
load(dataPath,'records','fs','classNames');
assert(exist('pwelch','file')==2,'Signal Processing Toolbox required.');
assert(exist('fitcsvm','file')==2, ...
    'Statistics and Machine Learning Toolbox required.');

%% Instructor experiment parameters
windowSeconds = 3.0;
hopSeconds = 0.5;
lowBand = [0.8 3.0];
highBand = [5.0 8.0];
powerFloor = 1e-12;      % g^2, ONLY to prevent log(0)
showPreviouslySeenSubjects = true; % exploratory diagnostic, NOT a new test

N = round(windowSeconds*fs);
H = round(hopSeconds*fs);
welchN = min(N,round(2*fs));
welchWindow = hamming(welchN);
welchOverlap = floor(welchN/2);
nfft = max(512,2^nextpow2(welchN));
assert(N>=welchN && H>=1 && highBand(2)<fs/2, ...
    'Invalid window or frequency band settings.');

modelNames = { ...
    'STD only', ...
    'STD + low-band power', ...
    'STD + high-band power', ...
    'STD + log(high/low)' ...
};
columns = {[1],[1 2],[1 3],[1 4]};
nModels = numel(modelNames);

%% Extract FOUR candidate features from exactly the SAME windows
% Columns: acceleration STD, low power, high power, log power ratio (dB).
nMax = 0;
for r = 1:numel(records)
    nMax = nMax + max(0,floor((size(records(r).acc,1)-N)/H)+1);
end
assert(nMax>0,'No valid analysis windows exist.');

X = nan(nMax,4);
Y = zeros(nMax,1);          % 0: Upstairs; 1: Downstairs
subjectID = zeros(nMax,1);
trialID = zeros(nMax,1);
recordID = zeros(nMax,1);
windowEndSec = zeros(nMax,1);
k=0;

fprintf('Computing low/high band power for 3-s windows...\n');
for r=1:numel(records)
    acc = records(r).acc;
    if size(acc,1)<N, continue; end

    for first = 1:H:(size(acc,1)-N+1)
        last = first+N-1;
        a = acc(first:last,:);
        if any(~isfinite(a(:))), continue; end

        aMag = sqrt(sum(a.^2,2));
        aDynamic = aMag-mean(aMag);

        [Pxx,f] = pwelch(aDynamic,welchWindow,welchOverlap,nfft,fs);
        lowIdx = (f>=lowBand(1)) & (f<=lowBand(2));
        highIdx = (f>=highBand(1)) & (f<=highBand(2));

        lowPower = trapz(f(lowIdx),Pxx(lowIdx));
        highPower = trapz(f(highIdx),Pxx(highIdx));
        logRatio = 10*log10(max(highPower,powerFloor) / ...
                            max(lowPower,powerFloor));

        k=k+1;
        X(k,:) = [std(aMag),lowPower,highPower,logRatio];
        Y(k) = records(r).label;
        subjectID(k) = records(r).subject;
        trialID(k) = records(r).trial;
        recordID(k) = r;
        windowEndSec(k) = (last-1)/fs;
    end
end

X=X(1:k,:); Y=Y(1:k);
subjectID=subjectID(1:k); trialID=trialID(1:k);
recordID=recordID(1:k); windowEndSec=windowEndSec(1:k);
assert(all(isfinite(X(:))) && numel(unique(Y))==2, ...
    'Missing/invalid computed features.');
nFloored = sum(X(:,2)<=powerFloor | X(:,3)<=powerFloor);
fprintf('Valid windows: %d; near-zero power windows: %d (%.2f%%)\n', ...
    k,nFloored,100*nFloored/k);

%% Reproduce the training/held-out split used by step04
subjects = unique(subjectID);
rng(3310);
shuffled = subjects(randperm(numel(subjects)));
nTrain = max(2,floor(0.75*numel(shuffled)));
trainingSubjects = sort(shuffled(1:nTrain));
previouslySeenSubjects = sort(shuffled(nTrain+1:end));
trainingMask = ismember(subjectID,trainingSubjects);

fprintf('TRAINING subjects: %s\n',mat2str(trainingSubjects));
fprintf('PREVIOUSLY SEEN held-out subjects: %s\n', ...
    mat2str(previouslySeenSubjects));
assert(numel(trainingSubjects)>=3, ...
    'Need at least 3 subjects for subject-wise CV.');

%% Leave-one-training-subject-out CV for ALL candidates
% Note: these frequency bands were suggested by viewing the same training
% subjects. This is EXPLORATORY CV, not a fully nested unbiased evaluation.
nTrainSubjects = numel(trainingSubjects);
cvBalanced = nan(nTrainSubjects,nModels);
cvRaw = nan(nTrainSubjects,nModels);

for si=1:nTrainSubjects
    s = trainingSubjects(si);
    foldTrain = trainingMask & subjectID~=s;
    foldVal = trainingMask & subjectID==s;
    assert(all(ismember([0 1],Y(foldTrain))) && ...
           all(ismember([0 1],Y(foldVal))), ...
           'Both motion classes must appear in each subject fold.');

    for mi=1:nModels
        features=columns{mi};
        mdl=fitcsvm(X(foldTrain,features),Y(foldTrain), ...
            'KernelFunction','linear','Standardize',true, ...
            'ClassNames',[0 1]);
        predicted=predict(mdl,X(foldVal,features));
        truth=Y(foldVal);
        upRecall = mean(predicted(truth==0)==0);
        downRecall = mean(predicted(truth==1)==1);
        cvBalanced(si,mi)=100*mean([upRecall downRecall]);
        cvRaw(si,mi)=100*mean(predicted==truth);
    end
end

fprintf('\nLEAVE-ONE-TRAINING-SUBJECT-OUT: BALANCED ACCURACY\n');
fprintf('(Each subject counts equally; this is exploratory CV.)\n');
for mi=1:nModels
    fprintf('%-28s  mean %5.1f%%  SD %5.1f%%  median %5.1f%%\n', ...
        modelNames{mi},mean(cvBalanced(:,mi)), ...
        std(cvBalanced(:,mi)),median(cvBalanced(:,mi)));
end

fprintf('\nPer-subject CV balanced accuracy (percent):\n');
T = array2table(cvBalanced,'VariableNames', ...
    {'STD_only','STD_Low','STD_High','STD_LogRatio'});
T = addvars(T,trainingSubjects(:),'Before',1, ...
    'NewVariableNames','SubjectID');
disp(T);

%% Diagnostic figure: variation across people matters more than windows
figure('Name','Subject-wise CV feature comparison');
plot(trainingSubjects,cvBalanced,'o-','LineWidth',1.2);
hold on; yline(50,'k--','Chance-level balanced accuracy');
ylim([0 105]); grid on;
xlabel('Training subject ID');
ylabel('Held-out-subject balanced accuracy (%)');
title('Leave-one-training-subject-out CV (exploratory)');
legend(modelNames,'Location','best');

%% Pairwise feature-space comparison, ONLY training subjects
% The same 2D linear SVM is fitted for display of each candidate boundary.
figure('Name','Feature space: raw power versus log ratio');
tiledlayout(1,3,'TileSpacing','compact');
idx0=find(trainingMask & Y==0); idx0=idx0(1:3:end);
idx1=find(trainingMask & Y==1); idx1=idx1(1:3:end);

for pi=1:3
    mi=pi+1; % omit 1D STD-only model from this 2D display
    feat=columns{mi};
    nexttile; hold on;
    scatter(X(idx0,feat(1)),X(idx0,feat(2)),8,'filled', ...
        'DisplayName','Upstairs');
    scatter(X(idx1,feat(1)),X(idx1,feat(2)),8,'filled', ...
        'DisplayName','Downstairs');

    displayModel=fitcsvm(X(trainingMask,feat),Y(trainingMask), ...
        'KernelFunction','linear','Standardize',true, ...
        'ClassNames',[0 1]);
    xVals=linspace(min(X(trainingMask,feat(1))), ...
                    max(X(trainingMask,feat(1))),80);
    yVals=linspace(min(X(trainingMask,feat(2))), ...
                    max(X(trainingMask,feat(2))),80);
    [A,B]=meshgrid(xVals,yVals);
    [~,score]=predict(displayModel,[A(:),B(:)]);
    contour(A,B,reshape(score(:,2),size(A)),[0 0], ...
        'k','LineWidth',1.5);

    xlabel('Accel magnitude STD (g)');
    if mi==2
        ylabel('0.8-3 Hz power (g^2)');
    elseif mi==3
        ylabel('5-8 Hz power (g^2)');
    else
        ylabel('10log10(P5-8 / P0.8-3) (dB)');
    end
    title(modelNames{mi});
    grid on;
    legend('Location','best');
end

%% Another direct physical check: Does the ratio distinguish people?
medUp=nan(nTrainSubjects,1); medDown=nan(nTrainSubjects,1);
for si=1:nTrainSubjects
    s=trainingSubjects(si);
    medUp(si)=median(X(subjectID==s & Y==0,4));
    medDown(si)=median(X(subjectID==s & Y==1,4));
end
figure('Name','Log ratio subject-by-subject');
plot(trainingSubjects,medUp,'o-','LineWidth',1.5); hold on;
plot(trainingSubjects,medDown,'s-','LineWidth',1.5);
xlabel('Training subject ID');
ylabel('Median high/low band power ratio (dB)');
title('Do relative spectral features transfer across subjects?');
legend(classNames,'Location','best'); grid on;
fprintf('\nMedian log-ratio (Downstairs > Upstairs) for %d of %d training subjects.\n', ...
    sum(medDown>medUp),nTrainSubjects);

%% Optional: reuse the FOUR previously viewed subjects as a diagnostic
% The original 72%% result and individual subject scores were already viewed.
% Results here MUST NOT be called an untouched independent test, and MUST
% NOT be used for repeated feature cherry-picking.
if showPreviouslySeenSubjects
    seenMask=ismember(subjectID,previouslySeenSubjects);
    fprintf('\nDIAGNOSTIC: PREVIOUSLY SEEN SUBJECTS (NOT A FRESH TEST)\n');
    fprintf('%-28s %10s %10s\n','Model','WindowAcc','SubjBalAcc');
    for mi=1:nModels
        feat=columns{mi};
        mdl=fitcsvm(X(trainingMask,feat),Y(trainingMask), ...
            'KernelFunction','linear','Standardize',true, ...
            'ClassNames',[0 1]);
        predicted=predict(mdl,X(seenMask,feat));
        truth=Y(seenMask);
        windowAcc=100*mean(predicted==truth);
        seenIDs=subjectID(seenMask);
        subjBalanced=zeros(numel(previouslySeenSubjects),1);
        for si=1:numel(previouslySeenSubjects)
            mask=seenIDs==previouslySeenSubjects(si);
            pred=predicted(mask);
            actual=truth(mask);
            assert(any(actual==0) && any(actual==1), ...
                'Both activities needed for balanced accuracy.');
            subjBalanced(si)=50*(mean(pred(actual==0)==0)+ ...
                                  mean(pred(actual==1)==1));
        end
        fprintf('%-28s %9.1f%% %10.1f%%\n', ...
            modelNames{mi},windowAcc,mean(subjBalanced));
    end
    fprintf('WARNING: These four subjects have been inspected before.\n');
    fprintf('Do not report this comparison as a new independent test.\n');
end

%% Save exploratory summary, NOT raw dataset
summaryPath=fullfile(scriptFolder, ...
    'ME3310_USC_HAD_PowerRatioComparison.mat');
save(summaryPath,'modelNames','cvBalanced','cvRaw', ...
    'trainingSubjects','previouslySeenSubjects','windowSeconds', ...
    'hopSeconds','lowBand','highBand','medUp','medDown');
fprintf('\nSaved summary: %s\n',summaryPath);

%% Take-home question:
% Does relative high-frequency power add INFORMATION rather than just
% increase the apparent number of features? A new person/recording is
% required for a final trustworthy generalization statement.
