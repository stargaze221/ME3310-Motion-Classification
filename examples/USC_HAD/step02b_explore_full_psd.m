%% ME 3310 | Explore the FULL PSD before selecting a frequency band
% step02b_explore_full_psd.m
%
% Engineering question:
% Does walking upstairs versus downstairs change the frequency CONTENT,
% or mainly the total magnitude of the acceleration?
%
% Plots:
%  Figure 1. Full 0-50 Hz average PSD: accel magnitude and 3 gyro axes.
%  Figure 2. Acceleration PSD zoom (0-10 Hz) and Down/Up PSD ratio.
%  Figure 3. Subject-by-subject Down/Up band-power ratios for candidate
%            bands. This is EXPLORATION, not classifier accuracy.
%
% Data integrity:
% - Uses the SAME 3-s windows, 0.5-s hop, and inner Welch settings as
%   step03_extract_window_features.m.
% - Average window PSDs WITHIN trial, then trials WITHIN each subject.
%   Subjects are weighted equally instead of long recordings dominating.
% - To avoid feature-selection leakage, plots and band summaries ONLY
%   use the TRAINING subjects selected by the same seeded split as
%   step04_train_linear_svm.m.
% - An overlapping window is not an independent experimental trial.
%
% Requires Signal Processing Toolbox (pwelch).

clear; clc;
scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder), scriptFolder = pwd; end
source = fullfile(scriptFolder,'ME3310_USC_HAD_UpDown.mat');
if ~isfile(source)
    error('Data file missing. Run step01_prepare_up_down.m first.');
end
load(source,'records','fs','classNames');
assert(exist('pwelch','file')==2, ...
    'This experiment requires Signal Processing Toolbox (pwelch).');

%% Parameters: identical to step03_extract_window_features.m
windowSeconds = 3.0;
hopSeconds = 0.5;
existingBandHz = [0.8 3.0];
zoomMaxHz = 10;
candidateBandsHz = [0.3 0.8; 0.8 1.5; 1.5 3.0; ...
                    3.0 5.0; 5.0 8.0; 8.0 12.0];

N = round(fs*windowSeconds);
H = round(fs*hopSeconds);
welchN = min(N,round(2*fs));
welchWindow = hamming(welchN);
welchOverlap = floor(welchN/2);
nfft = max(512,2^nextpow2(welchN));
frequency = (0:(nfft/2))'*fs/nfft;
nFreq = numel(frequency);

assert(N>1 && H>=1 && max(candidateBandsHz(:))<fs/2, ...
    'Window, hop, or frequency-band parameters are invalid.');

channelNames = {'Acceleration magnitude','Gyro X','Gyro Y','Gyro Z'};
units = {'g^2/Hz','(deg/s)^2/Hz','(deg/s)^2/Hz', ...
         '(deg/s)^2/Hz'};

%% The SAME held-out-subject split as the classifier script
allSubjects = unique([records.subject]);
rng(3310);
subjectOrder = allSubjects(randperm(numel(allSubjects)));
nTrain = max(2,floor(0.75*numel(subjectOrder)));
trainSubjects = sort(subjectOrder(1:nTrain));
testSubjects = sort(subjectOrder(nTrain+1:end));

fprintf('PSD exploration uses training subjects ONLY: %s\n', ...
    mat2str(trainSubjects));
fprintf('Held-out subjects NOT used: %s\n',mat2str(testSubjects));
fprintf('Window length %.1f s; hop %.1f s; sampling rate %.0f Hz\n', ...
    windowSeconds,hopSeconds,fs);

%% Average spectra: window -> trial -> subject (both activities)
% PSD (frequency x channel x class x subject)
nSubjects = numel(allSubjects);
subjectPSD = nan(nFreq,4,2,nSubjects);
nTrialsUsed = zeros(nSubjects,2);

for si = 1:nSubjects
    s = allSubjects(si);
    for ci = 1:2
        indices = find([records.subject]==s & [records.label]==ci-1);
        trialPSD = nan(nFreq,4,numel(indices));

        for ri = 1:numel(indices)
            R = records(indices(ri));
            ns = size(R.acc,1);
            total = zeros(nFreq,4);
            validCount = 0;

            for first = 1:H:(ns-N+1)
                last = first+N-1;
                a = R.acc(first:last,:);
                g = R.gyro(first:last,:);
                if any(~isfinite(a(:))) || any(~isfinite(g(:)))
                    continue;
                end

                aMag = sqrt(sum(a.^2,2));
                signals = [aMag,g]; % N-by-4
                signals = signals-mean(signals,1); % remove each DC offset
                P = pwelch(signals,welchWindow, ...
                    welchOverlap,nfft,fs); % freq-by-4
                total = total + P;
                validCount = validCount+1;
            end

            if validCount>0
                trialPSD(:,:,ri) = total/validCount;
                nTrialsUsed(si,ci) = nTrialsUsed(si,ci)+1;
            end
        end

        % Equal weight for each recorded trial within the same subject.
        if nTrialsUsed(si,ci)>0
            subjectPSD(:,:,ci,si) = mean(trialPSD,3,'omitnan');
        end
    end
end

isTraining = ismember(allSubjects,trainSubjects);
assert(any(isTraining),'No training subjects available.');

% Mean of subject-level spectra, equal weight for each training subject.
meanPSD = nan(nFreq,4,2);
for ci = 1:2
    for ch = 1:4
        tmp = reshape(subjectPSD(:,ch,ci,isTraining),nFreq,[]);
        meanPSD(:,ch,ci) = mean(tmp,2,'omitnan');
    end
end
fprintf('Subject-balanced PSDs computed using original trials.\n');

%% Figure 1: The FULL 0-to-Nyquist PSD for all four signal channels
figure('Name','Full PSD: training subjects only');
tiledlayout(2,2,'TileSpacing','compact');
for ch = 1:4
    nexttile; hold on;
    pUp = meanPSD(:,ch,1);
    pDown = meanPSD(:,ch,2);
    plot(frequency,10*log10(pUp+eps), ...
        'LineWidth',1.4,'DisplayName',classNames{1});
    plot(frequency,10*log10(pDown+eps), ...
        'LineWidth',1.4,'DisplayName',classNames{2});
    xlim([0 fs/2]);
    grid on;
    title(channelNames{ch});
    xlabel('Frequency (Hz)');
    ylabel(sprintf('PSD (dB re 1 %s)',units{ch}));
    legend('Location','best');
end
sgtitle(sprintf('Mean PSD across %d training subjects (equal subject weights)', ...
    nnz(isTraining)));

%% Figure 2: Zoom in on the acceleration spectrum and the PSD ratio
aUp = meanPSD(:,1,1);
aDown = meanPSD(:,1,2);

figure('Name','Acceleration PSD: full-band exploration');
tiledlayout(2,1,'TileSpacing','compact');

nexttile; hold on;
plot(frequency,10*log10(aUp+eps),'LineWidth',1.7);
plot(frequency,10*log10(aDown+eps),'LineWidth',1.7);
xline(existingBandHz(1),'k--','0.8 Hz');
xline(existingBandHz(2),'k--','3 Hz');
xlim([0 zoomMaxHz]);
xlabel('Frequency (Hz)');
ylabel('PSD (dB re 1 g^2/Hz)');
title('Acceleration magnitude: 0-10 Hz zoom');
legend(classNames,'Location','best');
grid on;

nexttile; hold on;
ratioDb = 10*log10((aDown+eps)./(aUp+eps));
plot(frequency,ratioDb,'LineWidth',1.5);
yline(0,'k--');
xline(existingBandHz(1),'k--');
xline(existingBandHz(2),'k--');
xlim([0 zoomMaxHz]);
xlabel('Frequency (Hz)');
ylabel('10 log10(P_{down}/P_{up}) (dB)');
title('Positive: more downstairs power; negative: more upstairs power');
grid on;

%% Figure 3: Does the direction of the difference agree across subjects?
% Every row is ONE training subject, not thousands of windows.
nBands = size(candidateBandsHz,1);
bandPower = nan(nSubjects,2,nBands); % subject x class x band

for b = 1:nBands
    mask = frequency>=candidateBandsHz(b,1) & ...
           frequency<=candidateBandsHz(b,2);
    for si = 1:nSubjects
        for ci = 1:2
            p = subjectPSD(:,1,ci,si);
            if all(isfinite(p(mask)))
                bandPower(si,ci,b) = trapz(frequency(mask),p(mask));
            end
        end
    end
end

logBandRatio = 10*log10((reshape(bandPower(:,2,:), ...
    nSubjects,nBands)+eps)./(reshape(bandPower(:,1,:), ...
    nSubjects,nBands)+eps));
trainingRatios = logBandRatio(isTraining,:);

bandLabels = cell(1,nBands);
for b = 1:nBands
    bandLabels{b} = sprintf('%.1f-%.1f', ...
        candidateBandsHz(b,1),candidateBandsHz(b,2));
end

figure('Name','Subject consistency: candidate frequency bands');
imagesc(trainingRatios);
colormap(parula);
cb = colorbar;
ylabel(cb,'Downstairs / Upstairs band power (dB)');
xticks(1:nBands);
xticklabels(bandLabels);
yticks(1:nnz(isTraining));
yticklabels(string(allSubjects(isTraining)));
xlabel('Frequency band (Hz)');
ylabel('Training subject ID');
title('Does the frequency difference persist across people?');
grid on;

fprintf('\nCandidate frequency-band comparison (TRAINING ONLY):\n');
for b = 1:nBands
    v = trainingRatios(:,b);
    v = v(isfinite(v));
    if ~isempty(v)
        fprintf('%10s Hz: median Down/Up = %+6.2f dB; ' ...
            + "positive for %d of %d subjects\n", ...
            bandLabels{b},median(v),sum(v>0),numel(v));
    end
end

%% Interpretation notes
% - A large average PSD difference does NOT guarantee accurate classification.
% - If a band differs in opposite directions across subjects, it may not
%   generalize. The heatmap makes such heterogeneity visible.
% - Large changes in PSD power may be explained by overall variance.
%   To test SPECTRAL SHAPE, inspect normalized PSD (divide by total power).
% - Candidate bands explored here have been selected using TRAIN subjects.
%   Never tune using the held-out subjects and then call them a clean test.
% - To use a different band in the actual classifier, change "bandHz"
%   in step03_extract_window_features.m, rerun 03, then rerun 04.
