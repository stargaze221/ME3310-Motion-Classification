function trial = load_student_trial(filePath)
% LOAD_STUDENT_TRIAL  Read a ME 3310 student MAT trial without inventing data.
%
% Expected measurement fields (as seen in T01/T02):
%   t       Nx1 time [seconds]
%   accel   Nx3 acceleration (UNIT MUST be checked in meta)
%   gyro    Nx3 angular velocity (optional)
%   meta    struct with optional ID, EventLabel, Units, etc.
%
% This adapter deliberately does NOT:
% - fill missing sensor channels with zeros;
% - convert units without reliable metadata;
% - infer a ground-truth class from a file name;
% - concatenate measurements into artificial trajectories.
%
% Output:
%   trial.t, trial.fs, trial.acc, trial.gyro, trial.meta,
%   trial.label, trial.id, trial.units, trial.path

    arguments
        filePath (1,:) char
    end
    assert(isfile(filePath),'Trial file does not exist: %s',filePath);
    vars=whos('-file',filePath);
    available={vars.name};
    assert(ismember('t',available),'Missing t (elapsed seconds).');
    assert(ismember('accel',available) || ismember('gyro',available), ...
        'Expected at least one Nx3 sensor field: accel or gyro.');

    requested=intersect({'t','accel','gyro','meta'}, ...
        available,'stable');
    S=load(filePath,requested{:});
    t=double(S.t(:));
    assert(numel(t)>=3 && all(isfinite(t)) && ...
        all(diff(t)>0),'Time must be finite and strictly increasing.');

    dt=diff(t);
    nominalDt=median(dt);
    trial.fs=1/nominalDt;
    if any(abs(dt-nominalDt)>0.05*nominalDt)
        warning('Time intervals vary by more than 5%% in %s.',filePath);
    end
    trial.t=t-t(1);
    n=numel(t);
    trial.acc=[];
    trial.gyro=[];

    if isfield(S,'accel')
        trial.acc=double(S.accel);
        assert(size(trial.acc,1)==n && size(trial.acc,2)==3 ...
            && all(isfinite(trial.acc(:))), ...
            'Acceleration must be finite Nx3 aligned with time.');
    end
    if isfield(S,'gyro')
        trial.gyro=double(S.gyro);
        assert(size(trial.gyro,1)==n && size(trial.gyro,2)==3 ...
            && all(isfinite(trial.gyro(:))), ...
            'Gyroscope must be finite Nx3 aligned with time.');
    end

    trial.meta=struct;
    if isfield(S,'meta') && isstruct(S.meta)
        trial.meta=S.meta;
    end
    trial.label='';
    trial.id='';
    if isfield(trial.meta,'EventLabel')
        trial.label=textFromMeta(trial.meta.EventLabel);
    end
    if isfield(trial.meta,'ID')
        trial.id=textFromMeta(trial.meta.ID);
    end

    trial.units=struct('accel','','gyro','','t','s');
    if isfield(trial.meta,'Units') && isstruct(trial.meta.Units)
        if isfield(trial.meta.Units,'accel')
            trial.units.accel=textFromMeta(trial.meta.Units.accel);
        end
        if isfield(trial.meta.Units,'gyro')
            trial.units.gyro=textFromMeta(trial.meta.Units.gyro);
        end
    end
    trial.path=filePath;
end

function value=textFromMeta(v)
    value='';
    try
        if ischar(v) || isstring(v) || iscategorical(v)
            value=char(string(v));
        elseif isnumeric(v) && isscalar(v)
            value=num2str(v);
        end
    catch
        value='';
    end
end
