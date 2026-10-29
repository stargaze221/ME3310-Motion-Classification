function R = week11_01_feature_demo(section,studentMatFile)
% WEEK11_01_FEATURE_DEMO | ME 3310 Week 11, Canvas Page 01
% A MATLAB command for each lesson: signal, time, fft, psd, bands, windows.
% Run 'all' to execute all six sections.
%
% From repository root:
% addpath("examples/Student_IMU");
% week11_01_feature_demo("signal")
% R = week11_01_feature_demo("time")
% R = week11_01_feature_demo("fft")
% R = week11_01_feature_demo("psd")
% R = week11_01_feature_demo("bands")
% R = week11_01_feature_demo("windows")
% R = week11_01_feature_demo("all")
% R = week11_01_feature_demo("all","C:\path\to\trial_T01.mat")
%
% If studentMatFile is absent, data are clearly labeled SYNTHETIC 20-Hz
% six-axis instructional data. Never report them as experimental evidence.
% Actual files may be accel only; missing gyro is NEVER filled with zero.
% 'psd', 'bands', 'all' require Signal Processing Toolbox (pwelch).

if nargin<1 || isempty(section),section="all";end
if nargin<2,studentMatFile="";end
section=lower(string(section));
assert(ismember(section,["signal","time","fft","psd", ...
    "bands","windows","all"]),'Unknown demo section.');

%% Load a student MAT file or make a REPRODUCIBLE synthetic example.
synthetic=strlength(string(studentMatFile))==0;
if synthetic
    fs=20;
    t=(0:139)'/fs;    % 7 seconds of teaching signal
    rng(3310);
    a=[0.09*sin(2*pi*1.7*t)+0.012*randn(size(t)), ...
       0.06*sin(2*pi*1.7*t+0.6)+0.012*randn(size(t)), ...
       1+0.26*sin(2*pi*1.7*t)+0.07*sin(2*pi*4.5*t) ...
       +0.012*randn(size(t))];
    g=[8*sin(2*pi*1.7*t),13*sin(2*pi*1.7*t+0.8), ...
       5*sin(2*pi*1.7*t+1.3)];
    name="SYNTHETIC 20-Hz six-axis signal (NOT student data)";
else
    T=load_student_trial(char(studentMatFile));
    fs=T.fs;t=T.t;a=T.acc;g=T.gyro;
    name="Actual file: "+string(studentMatFile);
end
n=numel(t);
assert(n>=8 && fs>0,'Too few samples or invalid sampling rate.');
if ~isempty(a)
    raw=sqrt(sum(a.^2,2));
    measured="Acceleration magnitude";
else
    raw=sqrt(sum(g.^2,2));
    measured="Angular-rate magnitude";
end
x=raw-mean(raw); % remove DC for FFT/PSD investigations
R=struct('source',name,'fs',fs,'time',t, ...
    'signal',raw,'signalDemeaned',x,'nSamples',n);
fprintf('\n=== Week 11 / 01: %s ===\n',section);
fprintf('%s\n',name);
fprintf('N=%d, fs=%.3f Hz, Nyquist=%.3f Hz\n',n,fs,fs/2);
if isempty(g),fprintf('NO GYRO measured. No gyro features fabricated.\n');end

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

%% 2. Mean, sample STD, RMS, absolute peak, range; optional jerk
if section=="time" || section=="all"
    traces=[];labels={};
    if ~isempty(a)
        traces=[traces,a,sqrt(sum(a.^2,2))];
        labels=[labels,{'a_x','a_y','a_z','|a|'}];
    end
    if ~isempty(g)
        traces=[traces,g,sqrt(sum(g.^2,2))];
        labels=[labels,{'w_x','w_y','w_z','|w|'}];
    end
    m=mean(traces,1)';
    sd=std(traces,0,1)';
    rmsValue=sqrt(mean(traces.^2,1))';
    absPeak=max(abs(traces),[],1)';
    rangeValue=(max(traces,[],1)-min(traces,[],1))';
    R.timeFeatures=table(string(labels(:)),m,sd,rmsValue, ...
        absPeak,rangeValue,'VariableNames', ...
        {'Channel','Mean','SampleSTD','RMS','PeakAbs','Range'});
    disp(R.timeFeatures);
    if ~isempty(a)
        jerk=diff(a)./diff(t);
        R.jerkRMS=sqrt(mean(sum(jerk.^2,2)));
        fprintf('Jerk RMS: %.5f [raw accel unit]/s.\n',R.jerkRMS);
    else
        R.jerkRMS=NaN;
    end
    fprintf('Within-trial STD is NOT between-independent-trial STD.\n');
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
