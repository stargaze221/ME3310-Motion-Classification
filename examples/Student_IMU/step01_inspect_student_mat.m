%% ME 3310 | Inspect actual student MAT files before classification
% step01_inspect_student_mat.m
%
% Choose one or more student trial_T##.mat files in the file dialog.
% Handles accel-only, gyro-only, or combined Nx3 measurements.
% Prints sample rate, recorded duration, sensor channels and optional
% ground-truth labels if present in meta.EventLabel.
%
% Never reports class accuracy from only one or two trials. It shows
% measurement quality and what features may be physically meaningful.
clear;clc;
scriptFolder=fileparts(mfilename('fullpath'));
if isempty(scriptFolder),scriptFolder=pwd;end
addpath(scriptFolder); % Makes load_student_trial available

[fileNames,folder]=uigetfile('*.mat', ...
    'Select student MAT trials (e.g. trial_T01.mat and trial_T02.mat)', ...
    'MultiSelect','on');
if isequal(fileNames,0)
    fprintf('No files selected.\n');
    return;
end
if ischar(fileNames),fileNames={fileNames};end

fprintf('ME 3310 student measurement inspection\n');
fprintf('No motion labels are assumed from filenames.\n\n');

for i=1:numel(fileNames)
    path=fullfile(folder,fileNames{i});
    tr=load_student_trial(path);
    n=numel(tr.t);
    duration=tr.t(end)-tr.t(1);
    nWindow=0;
    if round(3*tr.fs)<=n
        nWindow=floor((n-round(3*tr.fs))/max(1,round(0.5*tr.fs)))+1;
    end

    fprintf('\n=== %s ===\n',fileNames{i});
    fprintf('Samples: %d | fs (from time): %.3f Hz | duration %.3f s\n', ...
        n,tr.fs,duration);
    fprintf('3.0-s / 0.5-s hop: %d correlated windows\n',nWindow);
    fprintf('Nyquist frequency: %.3f Hz\n',tr.fs/2);
    if ~isempty(tr.id),fprintf('Metadata trial ID: %s\n',tr.id);end
    if ~isempty(tr.label)
        fprintf('Metadata event label: %s\n',tr.label);
    else
        fprintf('Metadata event label: absent/unreadable. Verify in MATLAB.\n');
    end

    if ~isempty(tr.acc)
        fprintf('Accelerometer: Nx3 (reported units: %s)\n', ...
            emptyAsUnknown(tr.units.accel));
        a=tr.acc;
        amag=sqrt(sum(a.^2,2));
        [peak,peakIndex]=max(amag);
        fprintf(['  Axis means:  [%8.4f %8.4f %8.4f]\n', ...
            '  Axis STD:    [%8.4f %8.4f %8.4f]\n'], ...
            mean(a,1),std(a,0,1));
        fprintf('  Magnitude mean %.4f, STD %.4f, peak %.4f at %.2f s\n', ...
            mean(amag),std(amag),peak,tr.t(peakIndex));

        figure('Name',['Student acceleration: ' fileNames{i}]);
        tiledlayout(3,1,'TileSpacing','compact');
        nexttile;
        plot(tr.t,a,'LineWidth',1);
        grid on;xlabel('Time (s)');ylabel('Acceleration (raw units)');
        legend('X','Y','Z','Location','best');
        title(['Acceleration XYZ: ' fileNames{i}],'Interpreter','none');
        nexttile;
        plot(tr.t,amag,'LineWidth',1);
        hold on;plot(tr.t(peakIndex),peak,'ro');
        grid on;xlabel('Time (s)');ylabel('Magnitude (raw units)');
        title('Acceleration magnitude (including gravity, if present)');

        nexttile;
        if exist('pwelch','file')==2 && n>=8
            dynamic=amag-mean(amag);
            segLength=min(n,max(8,round(2*tr.fs)));
            nfft=max(512,2^nextpow2(segLength));
            [Pxx,f]=pwelch(dynamic,hamming(segLength), ...
                floor(segLength/2),nfft,tr.fs);
            plot(f,Pxx,'LineWidth',1);grid on;
            xlim([0 tr.fs/2]);
            ylabel('PSD (raw units^2/Hz)');
            title('Welch PSD; only frequencies below Nyquist');
        else
            text(0.1,0.5,'Signal Processing Toolbox unavailable', ...
                'Units','normalized');
            axis off;
        end
        xlabel('Frequency (Hz)');
    end

    if ~isempty(tr.gyro)
        fprintf('Gyroscope: Nx3 (reported units: %s)\n', ...
            emptyAsUnknown(tr.units.gyro));
        figure('Name',['Student gyro: ' fileNames{i}]);
        plot(tr.t,tr.gyro,'LineWidth',1);
        grid on; xlabel('Time (s)');ylabel('Angular rate (raw units)');
        legend('X','Y','Z','Location','best');
        title(['Gyroscope XYZ: ' fileNames{i}],'Interpreter','none');
    end
end

fprintf('\nNEXT: group multiple independent, labeled original trials ');
fprintf('before training/testing any classifier.\n');
fprintf('Do NOT split overlapping windows of the same trial across groups.\n');

function out=emptyAsUnknown(x)
    if isempty(x),out='not specified';else,out=x;end
end
