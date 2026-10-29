%% ME 3310 | Prepare FIVE original USC-HAD motions
% step07_prepare_five_motions.m
%
% Official activity IDs:
%    1 Walking Forward      4 Walking Upstairs
%    5 Walking Downstairs    6 Running Forward
%    9 Standing
%
% Every record is one REAL original single-activity trial.
% This script does NOT concatenate trials, synthesize transitions, or
% redistribute source data. Data are downloaded from USC-SIPI as needed.
% Input file/folder follows the same layout as step01_prepare_up_down.m.
%
% Source: https://sipi.usc.edu/had/
% Zhang and Sawchuk, USC-HAD, SAGAware (2012).
clear; clc;
scriptFolder=fileparts(mfilename('fullpath'));
if isempty(scriptFolder),scriptFolder=pwd;end

activityIDs=[1 4 5 6 9];
classNames={'Walking Forward','Walking Upstairs', ...
    'Walking Downstairs','Running Forward','Standing'};
fs=100;
sourceFolder=scriptFolder;
if ~isfile(fullfile(sourceFolder,'USC-HAD.zip')) && ...
        ~isfolder(fullfile(sourceFolder,'USC_HAD_raw'))
    parentFolder=fileparts(scriptFolder);
    if isfile(fullfile(parentFolder,'USC-HAD.zip')) || ...
            isfolder(fullfile(parentFolder,'USC_HAD_raw'))
        sourceFolder=parentFolder;
    end
end
zipURL='https://sipi.usc.edu/had/USC-HAD.zip';
zipPath=fullfile(sourceFolder,'USC-HAD.zip');
rawDir=fullfile(sourceFolder,'USC_HAD_raw');
outPath=fullfile(scriptFolder,'ME3310_USC_HAD_FiveMotions.mat');

%% Download/extract only if selected activity files are not accessible
needUnzip=false;
for i=1:numel(activityIDs)
    listing=dir(fullfile(rawDir,'**',sprintf('a%dt*.mat',activityIDs(i))));
    if isempty(listing),needUnzip=true;end
end
if needUnzip
    if ~isfile(zipPath)
        fprintf('Downloading USC-HAD from the official provider...\n');
        websave(zipPath,zipURL,weboptions('Timeout',180));
    else
        fprintf('Reusing existing archive: %s\n',zipPath);
    end
    if ~isfolder(rawDir),mkdir(rawDir);end
    unzip(zipPath,rawDir);
end

%% Extract ACTUAL original single-activity trials
records=struct('subject',{},'trial',{},'activityID',{}, ...
    'label',{},'activity',{},'acc',{},'gyro',{},'t',{}, ...
    'sourceFile',{});
for c=1:numel(activityIDs)
    aid=activityIDs(c);
    found=dir(fullfile(rawDir,'**',sprintf('a%dt*.mat',aid)));
    fprintf('Class %d | %s | discovered %d files\n', ...
        c-1,classNames{c},numel(found));
    if isempty(found)
        error('No files for activity %d after extraction.',aid);
    end
    for i=1:numel(found)
        src=fullfile(found(i).folder,found(i).name);
        S=load(src,'sensor_readings');
        if ~isfield(S,'sensor_readings') || ...
                ~isnumeric(S.sensor_readings) || ...
                size(S.sensor_readings,2)<6 || size(S.sensor_readings,1)<2
            warning('Skipping malformed sensor recording: %s',src);
            continue;
        end
        normalized=strrep(src,'\','/');
        subTok=regexp(normalized,'Subject([0-9]+)', ...
            'tokens','once','ignorecase');
        trTok=regexp(found(i).name,'^a([0-9]+)t([0-9]+)\.mat$', ...
            'tokens','once','ignorecase');
        if isempty(subTok)||isempty(trTok)
            warning('Unexpected subject or trial filename: %s',src);
            continue;
        end
        A=double(S.sensor_readings(:,1:6));
        r.subject=str2double(subTok{1});
        r.trial=str2double(trTok{2});
        r.activityID=aid;
        r.label=c-1;                   % 0,1,2,3,4
        r.activity=classNames{c};
        r.acc=A(:,1:3);                % g
        r.gyro=A(:,4:6);               % degrees/s
        r.t=(0:size(A,1)-1)'/fs;
        r.sourceFile=src;
        records(end+1)=r; %#ok<SAGROW>
    end
end

assert(~isempty(records),'No valid records were found.');
for c=0:numel(classNames)-1
    assert(any([records.label]==c), ...
        'At least one selected activity has no valid trials.');
end
[~,order]=sortrows([[records.subject]',[records.label]', ...
    [records.trial]']);
records=records(order);

save(outPath,'records','fs','classNames','activityIDs', ...
    'zipURL','-v7.3');
fprintf('\nSaved %d original trials: %s\n',numel(records),outPath);
fprintf('Participating subjects: %s\n',mat2str(unique([records.subject])));
for c=0:4
    fprintf('  %s: %d original trials\n', ...
        classNames{c+1},sum([records.label]==c));
end
fprintf('No artificial motion transitions were created.\n');
