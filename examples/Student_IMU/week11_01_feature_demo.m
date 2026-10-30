function R = week11_01_feature_demo(section,dataSource)
% WEEK11_01_FEATURE_DEMO | ME 3310 Week 11, Canvas Page 01
% MEASURED recordings only: actual USC-HAD upstairs/downstairs trials,
% or a local student's actual MAT file.
%
% First-time setup from repository root:
% run("examples/USC_HAD/step01_prepare_up_down.m")
% addpath("examples/Student_IMU")
%
% week11_01_feature_demo("signal")                 % real USC-HAD Upstairs
% week11_01_feature_demo("signal","downstairs")    % real USC-HAD Downstairs
% R = week11_01_feature_demo("time","upstairs")        % WHOLE trial
% R = week11_01_feature_demo("timecompare","upstairs") % whole vs first 3 s
% R = week11_01_feature_demo("timecompare","downstairs")
% R = week11_01_feature_demo("fft","upstairs")
% R = week11_01_feature_demo("psd","downstairs")
% R = week11_01_feature_demo("bands","upstairs")
% R = week11_01_feature_demo("windows","upstairs")
% R = week11_01_feature_demo("all","upstairs")
% R = week11_01_feature_demo("all","C:\path\to\trial_T01.mat")
%
% Source data: USC-HAD 100 Hz acceleration (g) and gyro (deg/s).
% For comparing two motions, the default picks Subject 2 Trial 1
% (if present) for BOTH upstairs and downstairs.
% Actual student MAT files may have other fs/channels.
% No missing sensor channels are fabricated.
% Signal Processing Toolbox required for 'psd','bands','all'.

if nargin<1 || isempty(section),section="all";end
if nargin<2 || isempty(dataSource),dataSource="upstairs";end
section=lower(string(section));
dataSource=string(dataSource);
assert(ismember(section,["signal","time","timecompare","fft","psd", ...
    "bands","windows","all"]),'Unknown demo section.');

%% Select one ORIGINAL USC-HAD continuous trial or one student MAT file.
if any(strcmpi(dataSource,["upstairs","downstairs"]))
    scriptFolder=fileparts(mfilename('fullpath'));
    dataset=fullfile(fileparts(scriptFolder),"USC_HAD", ...
        "ME3310_USC_HAD_UpDown.mat");
    assert(isfile(dataset), ...
        ['USC-HAD data missing. From repository root, first run ' ...
         'run("examples/USC_HAD/step01_prepare_up_down.m").']);
    D=load(dataset,'records','fs','classNames');
    label=double(strcmpi(dataSource,"downstairs")); % Upstairs 0, Downstairs 1
    idx=find([D.records.subject]==2 & [D.records.trial]==1 & ...
             [D.records.label]==label,1);
    if isempty(idx)
        idx=find([D.records.label]==label,1); % fallback: real record only
    end
    assert(~isempty(idx),'No valid trial for requested USC-HAD motion.');
    original=D.records(idx);
    fs=double(D.fs);
    t=double(original.t(:));
    a=double(original.acc);
    g=double(original.gyro);
    assert(numel(t)==size(a,1) && numel(t)==size(g,1), ...
        'USC-HAD trial channel lengths do not match.');
    name=sprintf('USC-HAD: %s | Subject %d | Trial %d', ...
        D.classNames{label+1},original.subject,original.trial);
else
    assert(isfile(dataSource), ...
        ['Unknown source: choose "upstairs", "downstairs", ' ...
         'or an existing student MAT-file path.']);
    T=load_student_trial(char(dataSource));
    fs=T.fs;t=T.t;a=T.acc;g=T.gyro;
    name="Student measurement: "+dataSource;
end
n=numel(t);
assert(n>=8 && fs>0 && all(isfinite(t)), ...
    'Too few samples or invalid time/sampling rate.');
assert((isempty(a) || all(isfinite(a(:)))) && ...
       (isempty(g) || all(isfinite(g(:)))), ...
    'Measurement channels must contain finite values.');
assert(~isempty(a) || ~isempty(g), ...
    'No acceleration or gyroscope channel was recorded.');
if ~isempty(a)
    raw=sqrt(sum(a.^2,2));
    measured="Acceleration magnitude";
else
    raw=sqrt(sum(g.^2,2));
    measured="Angular-rate magnitude";
end
x=raw-mean(raw); % remove DC for spectral analysis, not for time features
R=struct('source',name,'fs',fs,'time',t, ...
    'signal',raw,'signalDemeaned',x,'nSamples',n);
fprintf('\n=== Week 11 / 01: %s ===\n',section);
fprintf('%s\n',name);
fprintf('N=%d, fs=%.3f Hz, Nyquist=%.3f Hz\n',n,fs,fs/2);
if isempty(g),fprintf('No gyroscope measured: no gyro features computed.\n');end

%% 1. Signal in the time domain
if section=="signal" || section=="all"
    figure('Name','01 | Time-series channels');
    nPanels=1+~isempty(g);
    subplot(nPanels,1,1);
    if ~isempty(a)
        plot(t,a,'LineWidth',1.1);
        ylabel('Acceleration (raw units)');
        legend('a_x','a_y','a_z','Location','best');
    else
        plot(t,g,'LineWidth',1.1);
        ylabel('Angular rate (raw units)');
        legend('w_x','w_y','w_z','Location','best');
    end
    grid on;xlabel('Time (s)');
    title(name,'Interpreter','none');
    if nPanels==2
        subplot(2,1,2);plot(t,g,'LineWidth',1.1);
        legend('w_x','w_y','w_z','Location','best');
        ylabel('Angular rate (raw units)');xlabel('Time (s)');
        title('Gyroscope XYZ');grid on;
    end
    fprintf('Median sample interval %.4f s.\n',median(diff(t)));
end

%% 2. Time-domain features: WHOLE original trial
% This mode intentionally does NOT divide the record into 3-s windows.
if section=="time" || section=="all"
    [R.timeFeatures,R.jerkRMS]=summarizeTimeChannels(a,g,t);
    fprintf('WHOLE original trial (%d samples, %.2f s span):\n', ...
        n,t(end)-t(1));
    disp(R.timeFeatures);
    if ~isempty(a)
        fprintf('Whole-trial jerk RMS: %.5f [accel unit]/s.\n',R.jerkRMS);
    end
    fprintf('Within-trial STD is NOT between-independent-trial STD.\n');
end

%% 2b. Same measured trial: WHOLE trial vs FIRST 3-s feature window
% Keep identical feature definitions so the observed difference is caused
% by the analysis interval, not a different computation or normalization.
if section=="timecompare" || section=="all"
    windowSeconds=3;
    windowN=round(windowSeconds*fs); % USC-HAD: 300 samples at 100 Hz
    assert(windowN>=2 && n>=windowN, ...
        'Need at least 3 s of measured data for this comparison.');
    windowIdx=1:windowN;
    if isempty(a),firstA=[];else,firstA=a(windowIdx,:);end
    if isempty(g),firstG=[];else,firstG=g(windowIdx,:);end
    [wholeFeatures,wholeJerk]=summarizeTimeChannels(a,g,t);
    [firstFeatures,firstJerk]=summarizeTimeChannels( ...
        firstA,firstG,t(windowIdx));
    assert(isequal(wholeFeatures.Channel,firstFeatures.Channel), ...
        'Channel definitions do not match.');
    R.wholeTrialTimeFeatures=wholeFeatures;
    R.firstWindowTimeFeatures=firstFeatures;
    R.wholeTrialJerkRMS=wholeJerk;
    R.firstWindowJerkRMS=firstJerk;
    R.firstWindowSampleIndices=windowIdx;
    R.firstWindowDurationSeconds=windowSeconds;
    R.timeComparison=table( ...
        wholeFeatures.Channel,wholeFeatures.Mean,firstFeatures.Mean, ...
        wholeFeatures.SampleSTD,firstFeatures.SampleSTD, ...
        wholeFeatures.RMS,firstFeatures.RMS, ...
        wholeFeatures.PeakAbs,firstFeatures.PeakAbs, ...
        'VariableNames',{'Channel','WholeMean','First3sMean', ...
        'WholeSTD','First3sSTD','WholeRMS','First3sRMS', ...
        'WholePeakAbs','First3sPeakAbs'});
    fprintf('\nWHOLE trial (%d samples) vs FIRST 3-s window (%d samples):\n', ...
        n,windowN);
    disp(R.timeComparison);
    if ~isempty(a)
        fprintf('Jerk RMS: whole %.4f vs first 3-s %.4f [accel unit]/s.\n', ...
            wholeJerk,firstJerk);
    end

    % FIGURE: measured first-window interval and time features computed
    % on exactly these two analysis intervals.
    nPanels=2+(~isempty(g) && ~isempty(a));
    figure('Name','01 | Whole trial versus first 3-second window');
    tiledlayout(nPanels,1,'TileSpacing','compact');
    nexttile;hold on;
    plot(t,raw,'Color',[0.22 0.42 0.68],'LineWidth',1.2);
    yl=ylim;
    xline(t(windowN),'--','End of first 3-s window', ...
        'LabelVerticalAlignment','middle','Color',[0.75 0.32 0.18]);
    xlabel('Time (s)');
    ylabel(measured+" (raw units)");
    title(name,'Interpreter','none');grid on;

    if ~isempty(a)
        row=find(wholeFeatures.Channel=="|a|",1);
        y1=[wholeFeatures.SampleSTD(row),firstFeatures.SampleSTD(row)];
        measureTitle='Acceleration magnitude STD';
        yLabel='STD (acceleration units)';
    else
        row=find(wholeFeatures.Channel=="|w|",1);
        y1=[wholeFeatures.SampleSTD(row),firstFeatures.SampleSTD(row)];
        measureTitle='Gyroscope magnitude STD';
        yLabel='STD (angular-rate units)';
    end
    nexttile;bar(y1,'FaceColor',[0.24 0.53 0.68]);
    xticks([1 2]);xticklabels({'Whole trial','First 3 s'});
    ylabel(yLabel);title(measureTitle);grid on;

    if ~isempty(g)&&~isempty(a)
        row=find(wholeFeatures.Channel=="|w|",1);
        y2=[wholeFeatures.RMS(row),firstFeatures.RMS(row)];
        nexttile;bar(y2,'FaceColor',[0.77 0.44 0.22]);
        xticks([1 2]);xticklabels({'Whole trial','First 3 s'});
        ylabel('RMS (angular-rate units)');
        title('Gyroscope magnitude RMS');grid on;
    end
    fprintf(['These are two summaries of the SAME original measured ' ...
        'trial, NOT independent repetitions.\n']);
end

%% 3. FFT -> one-sided amplitude and rectangular-window periodogram
if any(section==["fft","psd","bands","all"])
    X=fft(x);
    k=(0:floor(n/2))';
    ff=k*fs/n;
    ampl=abs(X(1:numel(k)))/n;
    p2=abs(X).^2/(n*fs);      % correct two-sided PSD normalization
    p1=p2(1:numel(k));
    if rem(n,2)==0
        interior=2:(numel(k)-1);  % DC and Nyquist not doubled
    else
        interior=2:numel(k);
    end
    ampl(interior)=2*ampl(interior);
    p1(interior)=2*p1(interior);
    R.fFFT=ff;R.fftAmplitude=ampl;R.periodogramPSD=p1;
    if section=="fft" || section=="all"
        figure('Name','02 | FFT versus PSD');
        subplot(2,1,1);plot(ff,ampl,'LineWidth',1.2);
        xlim([0 fs/2]);grid on;
        ylabel('One-sided FFT amplitude');
        title('FFT coefficients contain amplitude AND phase');
        subplot(2,1,2);plot(ff,p1,'LineWidth',1.2);
        xlim([0 fs/2]);grid on;
        ylabel('PSD (signal units^2/Hz)');xlabel('Frequency (Hz)');
        title('Periodogram = FFT power with PSD normalization');
        fprintf('Native FFT spacing fs/N = %.4f Hz.\n',fs/n);
        fprintf('Parseval check: mean(x^2)=%.6f, sum(PSD)*df=%.6f\n', ...
            mean(x.^2),sum(p1)*fs/n);
    end
end

%% 4. Welch PSD; 5. selected low/high frequency-band power
if any(section==["psd","bands","all"])
    assert(exist('pwelch','file')==2, ...
        'pwelch requires Signal Processing Toolbox.');
    segmentN=min(n,max(8,round(2*fs)));
    nFFT=max(512,2^nextpow2(segmentN));
    [P,f]=pwelch(x,hamming(segmentN),floor(segmentN/2),nFFT,fs);
    R.fWelch=f;R.welchPSD=P;
    fprintf('Welch: %d-sample segments (%.2f s); native 1/T=%.3f Hz.\n', ...
        segmentN,segmentN/fs,fs/segmentN);
    fprintf('Zero padding refines the grid, NOT the true resolution.\n');
    if section=="psd" || section=="all"
        figure('Name','03 | Periodogram versus Welch PSD');
        plot(ff,p1,'Color',[0.65 0.65 0.65],'LineWidth',1);
        hold on;plot(f,P,'Color',[0.06 0.39 0.71],'LineWidth',1.4);
        grid on;xlim([0 fs/2]);xlabel('Frequency (Hz)');
        ylabel('PSD (signal units^2/Hz)');
        legend('Full-trial periodogram','Welch PSD','Location','best');
        title(measured);
    end
    if section=="bands" || section=="all"
        edges=[0.5 3;3 8];
        if fs/2<=8
            warning('Nyquist <=8 Hz; skip bands that reach Nyquist.');
            edges=edges(edges(:,2)<fs/2,:);
        end
        bp=zeros(size(edges,1),1);
        for j=1:size(edges,1)
            use=f>=edges(j,1)&f<=edges(j,2);
            assert(nnz(use)>=2,'Too few PSD points for band integration.');
            bp(j)=trapz(f(use),P(use));
            fprintf('Band %.1f-%.1f Hz: %.6f [signal unit]^2\n', ...
                edges(j,1),edges(j,2),bp(j));
        end
        R.bandEdgesHz=edges;
        R.bandPowers=bp;
        figure('Name','04 | Frequency-band power');
        plot(f,P,'LineWidth',1.3);grid on;hold on;
        bandColors=lines(size(edges,1));
        for j=1:size(edges,1)
            use=f>=edges(j,1)&f<=edges(j,2);
            area(f(use),P(use),'FaceColor',bandColors(j,:), ...
                'FaceAlpha',0.20,'EdgeColor','none');
        end
        xlabel('Frequency (Hz)');ylabel('PSD (signal units^2/Hz)');
        xlim([0 fs/2]);
        title('Area under PSD = frequency band-power feature');
    end
end

%% 6. Overlapping windows do not generate independent experiments
if section=="windows" || section=="all"
    windowN=round(3*fs);
    hopN=round(0.5*fs);
    assert(windowN>=2&&hopN>=1&&n>=windowN, ...
        'At least 3 seconds of data required for window demo.');
    starts=1:hopN:(n-windowN+1);
    wSTD=zeros(numel(starts),1);
    wEnd=zeros(numel(starts),1);
    for j=1:numel(starts)
        idx=starts(j):(starts(j)+windowN-1);
        wSTD(j)=std(raw(idx));
        wEnd(j)=t(idx(end));
    end
    R.windowEndTime=wEnd;R.windowStd=wSTD;
    fprintf('%d overlapping windows from ONE original trial.\n', ...
        numel(starts));
    figure('Name','05 | Overlapping windows');
    subplot(2,1,1);plot(t,raw,'LineWidth',1.2);
    grid on;hold on;
    yl=ylim;cc=lines(3);
    for j=1:min(3,numel(starts))
        startT=t(starts(j));
        endT=t(starts(j)+windowN-1);
        plot([startT startT],yl,'--','Color',cc(j,:));
        plot([endT endT],yl,':','Color',cc(j,:));
    end
    xlabel('Time (s)');ylabel('Signal magnitude (raw units)');
    title('First 3 analysis windows share most of their samples');
    subplot(2,1,2);plot(wEnd,wSTD,'o-','LineWidth',1.3);
    grid on;xlabel('Window end time (s)');ylabel('Window STD');
    title('Many summaries from one trial, NOT independent repetitions');
end
fprintf('Next: compare three independent Target vs three Confusing trials.\n');
end

function [features,jerkRMS]=summarizeTimeChannels(a,g,t)
% Same time-domain feature definitions for ANY contiguous interval.
t=double(t(:));
traces=[];labels={};
if ~isempty(a)
    traces=[traces,a,sqrt(sum(a.^2,2))];
    labels=[labels,{'a_x','a_y','a_z','|a|'}];
end
if ~isempty(g)
    traces=[traces,g,sqrt(sum(g.^2,2))];
    labels=[labels,{'w_x','w_y','w_z','|w|'}];
end
assert(~isempty(traces) && size(traces,1)==numel(t), ...
    'Time and measured channels do not match.');
m=mean(traces,1)';
sd=std(traces,0,1)';
rmsValue=sqrt(mean(traces.^2,1))';
absPeak=max(abs(traces),[],1)';
rangeValue=(max(traces,[],1)-min(traces,[],1))';
features=table(string(labels(:)),m,sd,rmsValue, ...
    absPeak,rangeValue,'VariableNames', ...
    {'Channel','Mean','SampleSTD','RMS','PeakAbs','Range'});
jerkRMS=NaN;
if ~isempty(a)
    dt=diff(t);
    assert(all(dt>0),'Timestamps must increase to calculate jerk.');
    jerk=diff(a)./dt;
    jerkRMS=sqrt(mean(sum(jerk.^2,2)));
end
end
