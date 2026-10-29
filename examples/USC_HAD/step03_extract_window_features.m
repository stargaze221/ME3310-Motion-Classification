%% ME 3310 | Project Week 2 | Sliding-window feature extraction
% step03_extract_window_features.m
%
% One trial -> many fixed-duration windows -> one feature vector per window.
% We keep subject/trial identity so overlapping windows do NOT leak between
% train and test. Requires Signal Processing Toolbox for pwelch.

clear; clc;
scriptFolder = fileparts(mfilename('fullpath'));
if isempty(scriptFolder), scriptFolder=pwd; end
source = fullfile(scriptFolder,'ME3310_USC_HAD_UpDown.mat');
if ~isfile(source)
    error('Missing data. Run step01_prepare_up_down.m or copy the MAT file here.');
end
load(source,'records','fs','classNames');

%% STUDENT EXPERIMENT PARAMETERS
windowSeconds = 3.0;     % How many seconds are used to make one decision?
hopSeconds = 0.5;        % How often is a new decision requested?
bandHz = [0.8 3.0];      % ORIGINAL feature #2 (retain for comparison)
% EIGHT additional acceleration-magnitude PSD bands; exploratory choices.
% High-frequency bands may include impacts and sensor noise.
% The 0.3-0.8 Hz band is coarse with a 2-s Welch segment.
extraBandsHz = [0.3 0.8; 0.8 1.5; 1.5 3.0; 3.0 5.0; ...
                5.0 8.0; 8.0 12.0; 12.0 20.0; 20.0 40.0];
N = round(windowSeconds*fs);
H = round(hopSeconds*fs);
assert(N>1 && H>=1 && bandHz(2)<fs/2 && ...
    all(extraBandsHz(:,1)>0) && ...
    all(extraBandsHz(:,1)<extraBandsHz(:,2)) && ...
    all(extraBandsHz(:,2)<fs/2), ...
    'Check window, hop and frequency-band settings.');
assert(exist('pwelch','file')==2, ...
    'Signal Processing Toolbox (pwelch) is required.');

% To preserve physical meaning, all features use the same units:
% acceleration magnitude in g; gyroscope in degrees/second.
featureNames = { ...
    'Accel magnitude STD (g)', ...
    'Accel PSD band power (g^2)', ...
    'Gyro X RMS (deg/s)', ...
    'Gyro Y RMS (deg/s)', ...
    'Gyro Z RMS (deg/s)', ...
    'Gyro X mean (deg/s)', ...
    'Gyro Y mean (deg/s)', ...
    'Gyro Z mean (deg/s)' ...
};
% Keep existing indices 1-8 UNCHANGED for reproducible SVM comparisons.
% New PSD band-power features become columns 9-16.
for b=1:size(extraBandsHz,1)
    featureNames{end+1} = sprintf( ...
        'Accel PSD %.1f-%.1f Hz power (g^2)', ...
        extraBandsHz(b,1),extraBandsHz(b,2));
end

% Sufficient PSD segment duration for a short analysis window.
welchN = min(N,round(2*fs));
welchWindow = hamming(welchN);
welchOverlap = floor(welchN/2);
nfft = max(512,2^nextpow2(welchN));

%% Count the windows before filling the arrays
nW = 0;
for r = 1:numel(records)
    nSamples = size(records(r).acc,1);
    nW = nW + max(0,floor((nSamples-N)/H)+1);
end
assert(nW>0,'No windows: try a shorter windowSeconds.');

X = zeros(nW,numel(featureNames));
Y = zeros(nW,1);                  % 0=Upstairs, 1=Downstairs
subjectID = zeros(nW,1);
trialID = zeros(nW,1);
recordID = zeros(nW,1);
windowEndTime = zeros(nW,1);

%% Extract features from EACH original trial separately
k = 0;
for r = 1:numel(records)
    acc = records(r).acc;
    gyro = records(r).gyro;
    nSamples = size(acc,1);
    for first = 1:H:(nSamples-N+1)
        last = first+N-1;
        a = acc(first:last,:);
        g = gyro(first:last,:);

        if any(~isfinite(a(:))) || any(~isfinite(g(:)))
            continue; % do not silently treat missing data as zero
        end

        aMag = sqrt(sum(a.^2,2));
        aDynamic = aMag-mean(aMag);

        [Pxx,f] = pwelch(aDynamic,welchWindow, ...
            welchOverlap,nfft,fs);
        keepBand = f>=bandHz(1) & f<=bandHz(2);
        bandPower = trapz(f(keepBand),Pxx(keepBand));

        % Eight EXTRA frequency-band powers, all from the SAME PSD.
        extraPower = zeros(1,size(extraBandsHz,1));
        for b=1:size(extraBandsHz,1)
            keep = f>=extraBandsHz(b,1) & f<=extraBandsHz(b,2);
            extraPower(b) = trapz(f(keep),Pxx(keep));
        end

        k = k+1;
        X(k,:) = [std(aMag),bandPower, ...
            sqrt(mean(g.^2,1)),mean(g,1),extraPower];
        Y(k) = records(r).label;
        subjectID(k) = records(r).subject;
        trialID(k) = records(r).trial;
        recordID(k) = r;
        windowEndTime(k) = (last-1)/fs;
    end
end

% Remove unused preallocation if invalid windows were skipped
X = X(1:k,:); Y = Y(1:k);
subjectID = subjectID(1:k); trialID = trialID(1:k);
recordID = recordID(1:k); windowEndTime = windowEndTime(1:k);
assert(any(Y==0) && any(Y==1),'Both activities must have valid windows.');

outPath = fullfile(scriptFolder,'ME3310_USC_HAD_WindowFeatures.mat');
save(outPath,'X','Y','featureNames','classNames','subjectID', ...
    'trialID','recordID','windowEndTime','windowSeconds', ...
    'hopSeconds','bandHz','extraBandsHz','fs');
fprintf('Feature count: %d (%d original + %d additional PSD bands).\n', ...
    size(X,2),8,size(extraBandsHz,1));
fprintf('Extracted %d windows from %d original trials.\n', ...
    size(X,1),numel(unique(recordID)));
fprintf('Upstairs windows: %d, Downstairs windows: %d\n', ...
    sum(Y==0),sum(Y==1));
fprintf('Saved: %s\n',outPath);

%% Visualize TWO features, before fitting any classifier
% The next script lets you change these feature indices too.
featureX = 1;    % Acceleration magnitude standard deviation
featureY = 2;    % Acceleration PSD band power
figure('Name','Window-level feature space'); hold on;
scatter(X(Y==0,featureX),X(Y==0,featureY),10,'filled', ...
    'DisplayName',classNames{1});
scatter(X(Y==1,featureX),X(Y==1,featureY),10,'filled', ...
    'DisplayName',classNames{2});
xlabel(featureNames{featureX}); ylabel(featureNames{featureY});
title(sprintf('Feature space: %.1f-s window, %.1f-s hop', ...
    windowSeconds,hopSeconds));
legend('Location','best'); grid on;

%% Engineering questions
% 1. Are these TWO features genuinely discriminative for Up vs Down?
% 2. Compare the original 8 features with all 16. Does more power
%    spectrum information improve performance across NEW subjects?
% 3. Some bands overlap original feature 2; correlation and noise matter.
% 4. Compare windowSeconds = 1.5, 3, and 5. Is PSD useful for all?
% 5. Does creating more overlapping windows create more INDEPENDENT data?
