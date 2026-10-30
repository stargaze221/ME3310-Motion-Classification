function U = week11_05_uncertainty_demo(section)
% WEEK11_05_UNCERTAINTY_DEMO | ME 3310 Week 11, Canvas Page 05
% REAL USC-HAD IMU recordings and EXISTING five-class SVM results only.
%
% After running step07, step08, step09 (Page 04), from repository root:
%   addpath(fullfile(pwd,"examples","Student_IMU"))
%   U = week11_05_uncertainty_demo("features");
%   U = week11_05_uncertainty_demo("subjects");
%   U = week11_05_uncertainty_demo("errors");
%   U = week11_05_uncertainty_demo("all");
%   disp(U.featureIntervals)
%   disp(U.subjectReport)
%   disp(U.classRecall)
%
% Figures:
%  1. Five independent trials of each motion: feature mean +/- 95% t CI
%  2. 8-feature vs 16-feature LOSO trial accuracy BY validation subject
%  3. 5-class TRIAL confusion matrix and class-specific recalls
%
% CI relates ONLY to mean feature across repeated original trials for a
% fixed USC-HAD subject; it is NOT a 95% confidence interval on model
% accuracy or reliability. The t interval assumes independent,
% approximately normally distributed original-trial feature summaries.
% Overlapping windows are NOT independent repetitions.
%
% Saved SVM predictions are subject-held-out LOSO on development people;
% those folds use overlapping training sets, so do NOT treat fold scores
% as independent trials for a simple t confidence interval on accuracy.
% Previously viewed diagnostic people are NOT part of the 10-fold CV.
% Requires Statistics and Machine Learning Toolbox (tinv).

if nargin<1 || isempty(section),section="all";end
section=lower(string(section));
assert(any(section==["all","features","subjects","errors"]), ...
    'Use section "features", "subjects", "errors" or "all".');
assert(exist('tinv','file')==2, ...
    'Statistics and Machine Learning Toolbox (tinv) required.');

here=fileparts(mfilename('fullpath'));
folder=fullfile(fileparts(here),"USC_HAD");
featPath=fullfile(folder,"ME3310_USC_HAD_FiveMotion_WindowFeatures.mat");
cvPath=fullfile(folder,"ME3310_USC_HAD_FiveMotion_Comparison.mat");
assert(isfile(featPath), ...
    'Missing five-motion features: run step07 then step08 in examples/USC_HAD.');
assert(isfile(cvPath), ...
    'Missing five-motion CV results: run step09_train_five_motion_svm.m.');

F=load(featPath,'X','Y','subjectID','recordID','trialID', ...
    'featureNames','classNames');
S=load(cvPath,'cvTrialBalanced','cvTrialTruth','cvTrialPred', ...
    'trainSubjects','cvTrialCount','modelNames','classNames');
assert(isequal(F.classNames,S.classNames), ...
    'Feature labels and saved CV labels disagree.');
assert(numel(S.trainSubjects)==size(S.cvTrialBalanced,1) && ...
    size(S.cvTrialBalanced,2)==2 && ...
    numel(S.cvTrialTruth)==2 && numel(S.cvTrialPred)==2 && ...
    numel(S.classNames)==5,'Unexpected saved SVM data structure.');
assert(all(isfinite(F.X(:))),'Nonfinite feature measurements.');
U=struct;
U.classNames=S.classNames;
U.subjectReport=table(S.trainSubjects(:), ...
    S.cvTrialCount(:),S.cvTrialBalanced(:,1),S.cvTrialBalanced(:,2), ...
    S.cvTrialBalanced(:,2)-S.cvTrialBalanced(:,1), ...
    'VariableNames',{'Subject','OriginalTrials','SVM8_pct', ...
    'SVM16_pct','Change_pct_points'});
fprintf('\nPage 05: Uncertainty and Reliability, measured USC-HAD only.\n');
fprintf(['The class count is five; the saved outcome is an OFFLINE ' ...
    'prediction for each original single-activity trial.\n']);
fprintf('Development subjects (LOSO CV): %s\n', ...
    mat2str(S.trainSubjects(:)'));

%% A. Repeated original trials -> feature mean and uncertainty interval.
if any(section==["all","features"])
    subjectNumber=2;                 % observed person, not a new dataset
    labels=[1 2];                    % five-class: upstairs=1 downstairs=2
    featureIdx=[1 2];               % accel magnitude STD, 0.8-3 Hz power
    recordIDs=unique(F.recordID(F.subjectID==subjectNumber),'stable');
    nR=numel(recordIDs);
    trials=zeros(nR,1); classes=zeros(nR,1);
    rawMean=nan(nR,numel(featureIdx));
    for k=1:nR
        mask=F.recordID==recordIDs(k);
        trials(k)=F.trialID(find(mask,1));
        classes(k)=F.Y(find(mask,1));
        rawMean(k,:)=mean(F.X(mask,featureIdx),1);
        assert(all(F.subjectID(mask)==subjectNumber) && ...
            all(F.Y(mask)==classes(k)) && ...
            all(F.trialID(mask)==trials(k)), ...
            'Some original-trial windows have inconsistent metadata.');
    end
    R=cell(numel(labels)*numel(featureIdx),6);
    line=0;
    figure('Name','Page 05 | Repeatability and 95% feature-mean CI');
    tiledlayout(1,2,'TileSpacing','compact');
    colors=[0.15 0.38 0.71;0.82 0.40 0.15];
    for j=1:numel(featureIdx)
        nexttile;hold on;
        for ci=1:numel(labels)
            values=rawMean(classes==labels(ci),j);
            assert(numel(values)>=3, ...
                'At least 3 independent ORIGINAL trials per class needed.');
            n=numel(values);
            avg=mean(values);s=std(values,0);
            sem=s/sqrt(n);
            halfWidth=tinv(0.975,n-1)*sem;
            xpoints=ci+0.12*linspace(-1,1,n)';
            scatter(xpoints,values,54,'filled', ...
                'MarkerFaceColor',colors(ci,:), ...
                'MarkerEdgeColor','white');
            errorbar(ci,avg,halfWidth,'k','LineWidth',1.7, ...
                'CapSize',12);
            plot(ci,avg,'kd','MarkerFaceColor','black','MarkerSize',7);
            line=line+1;
            R(line,:)={S.classNames{labels(ci)+1}, ...
                featureIdx(j),n,avg,s,halfWidth};
        end
        xticks([1 2]);xticklabels(S.classNames(2:3));
        xlim([0.65 2.35]);grid on;
        ylabel(F.featureNames{featureIdx(j)},'Interpreter','none');
        title(sprintf('Feature %d | Independent trial measurements', ...
            featureIdx(j)));
    end
    sgtitle('USC-HAD Subject 2 | 95% t CI for each CLASS MEAN FEATURE');
    U.featureIntervals=cell2table(R,'VariableNames', ...
        {'Motion','FeatureIndex','IndependentTrials','Mean', ...
        'SampleSTD','CI_HalfWidth'});
    disp(U.featureIntervals);
    fprintf(['Feature CIs assume approximately independent, normal ' ...
        'TRIAL summaries, not independent overlapping windows.\n']);
else
    U.featureIntervals=table();
end

%% B. Substantial participant-to-participant variation
if any(section==["all","subjects"])
    figure('Name','Page 05 | Participant differences in LOSO trial score');
    hold on;
    x=1:numel(S.trainSubjects);
    for i=1:numel(x)
        plot([x(i)-0.1 x(i)+0.1],S.cvTrialBalanced(i,:), ...
            '-','Color',[.70 .73 .78],'LineWidth',1.2);
    end
    scatter(x-0.10,S.cvTrialBalanced(:,1),58,'filled', ...
        'MarkerFaceColor',[.16 .43 .69],'DisplayName','8 features');
    scatter(x+0.10,S.cvTrialBalanced(:,2),58,'filled', ...
        'MarkerFaceColor',[.84 .42 .17],'DisplayName','16 features');
    xticks(x);xticklabels(string(S.trainSubjects(:)));
    xlim([0.4 numel(x)+0.6]);ylim([0 105]);
    ylabel('Held-out subject, balanced original-trial accuracy (%)');
    xlabel('Validation subject ID');grid on;
    legend('Location','best');
    title('Same real experiment, different people and feature sets');
    fprintf('Subject-weighted mean LOSO balanced trial scores:\n');
    fprintf('SVM-8: %.1f%%; SVM-16: %.1f%%\n', ...
        mean(S.cvTrialBalanced(:,1)), ...
        mean(S.cvTrialBalanced(:,2)));
    fprintf('Between-subject SD, SVM-16: %.1f percentage points\n', ...
        std(S.cvTrialBalanced(:,2),0));
    fprintf(['Do NOT interpret SD between folds as a confidence ' ...
        'interval for generalization accuracy.\n']);
    disp(U.subjectReport);
end

%% C. Error TYPES by true class; 5x5 from real held-out trials
if any(section==["all","errors"])
    classes=(0:4)';
    nC=numel(classes);
    U.confusionCounts16=zeros(nC,nC);
    U.classRecall=table(string(S.classNames(:)),zeros(nC,1), ...
        zeros(nC,1),'VariableNames',{'Motion', ...
        'SVM8_TrialRecall_pct','SVM16_TrialRecall_pct'});
    for m=1:2
        y=double(S.cvTrialTruth{m}(:));
        p=double(S.cvTrialPred{m}(:));
        assert(numel(y)==numel(p) && ...
            all(ismember(y,classes)) && all(ismember(p,classes)), ...
            'Saved trial predictions are malformed.');
        cm=zeros(nC,nC);
        for row=1:nC
            for column=1:nC
                cm(row,column)=nnz(y==classes(row) & p==classes(column));
            end
        end
        denom=sum(cm,2);
        assert(all(denom>0),'Every class must occur among trials.');
        recall=100*diag(cm)./denom;
        if m==1
            U.classRecall.SVM8_TrialRecall_pct=recall;
        else
            U.classRecall.SVM16_TrialRecall_pct=recall;
            U.confusionCounts16=cm;
        end
    end
    figure('Name','Page 05 | Wrong predictions and per-class reliability');
    tiledlayout(1,2,'TileSpacing','compact');
    nexttile;
    imagesc(U.confusionCounts16);
    axis square; colormap(gca,parula);colorbar;
    xticks(1:nC);yticks(1:nC);
    xticklabels({'Walk','Upstairs','Downstairs','Run','Stand'});
    yticklabels({'Walk','Upstairs','Downstairs','Run','Stand'});
    xtickangle(25);
    xlabel('Predicted motion');ylabel('Measured true motion');
    title('SVM-16 original-trial CV confusion counts');
    for i=1:nC
        for j=1:nC
            text(j,i,num2str(U.confusionCounts16(i,j)), ...
                'FontWeight','bold','HorizontalAlignment','center', ...
                'Color','k');
        end
    end
    nexttile;bar([U.classRecall.SVM8_TrialRecall_pct, ...
        U.classRecall.SVM16_TrialRecall_pct],'grouped');
    xticks(1:nC);xticklabels({'Walk','Upstairs','Downstairs', ...
        'Run','Stand'});xtickangle(25);
    ylim([0 105]);ylabel('Original-trial recall (%)');grid on;
    legend({'SVM 8','SVM 16'},'Location','best');
    title('Which original motion is most often missed?');
    sgtitle('Measured USC-HAD | 10 development subjects, held out one at a time');
    disp(U.classRecall);
    fprintf(['These are conditional observed error rates by class. ' ...
        'They are not detection-latency or false-alarm-rate tests.\n']);
end

fprintf(['\nEngineering judgment: state what is supported, what is ' ...
    'uncertain, and the next independent measurement needed.\n']);
end
