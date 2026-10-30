function M = week11_03_classifiers_demo(mode,featurePair)
% WEEK11_03_CLASSIFIERS_DEMO | ME 3310 Week 11 Canvas Page 03
% Logistic regression vs LINEAR SVM using REAL USC-HAD trial features.
% One ORIGINAL trial = one observation; overlapping 3-second windows are
% averaged WITHIN the trial. This demo classifies completed trials, not
% continuous transitions or real-time detection.
%
% From repository root after running step01 and step03:
%  addpath(fullfile(pwd,"examples","Student_IMU"));
%  M = week11_03_classifiers_demo("manual",[1 2]);
%  M = week11_03_classifiers_demo("manual",[1 3]);
%  M = week11_03_classifiers_demo("search");
%  disp(M.diagnosticReport);
%  disp(M.candidateResults);        % available in "search" mode
%
% No fabricated observations or automatic hyperparameter tuning.
% Development subjects: 2 3 4 7 8 9 10 11 12 14
% Diagnostic subjects:  1 5 6 13
% The diagnostic subjects were PREVIOUSLY EXAMINED in this project:
% results are useful for teaching but NOT a pristine final test.
% Feature screening, when selected, uses ONLY development subjects with
% leave-one-subject-out CV. Standardization is fit inside each training
% fold. Validation and diagnostic scores are TRIAL-level.
%
% Requires Statistics and Machine Learning Toolbox (fitglm, fitcsvm).

if nargin<1 || isempty(mode),mode="manual";end
if nargin<2 || isempty(featurePair),featurePair=[1 2];end
mode=lower(string(mode));
assert(any(mode==["manual","search"]), ...
    'mode must be "manual" or "search".');
assert(exist('fitglm','file')==2 && exist('fitcsvm','file')==2, ...
    'Statistics and Machine Learning Toolbox (fitglm and fitcsvm) required.');

here=fileparts(mfilename('fullpath'));
source=fullfile(fileparts(here),"USC_HAD", ...
    "ME3310_USC_HAD_WindowFeatures.mat");
assert(isfile(source), ...
    ['Run run("examples/USC_HAD/step01_prepare_up_down.m") then ' ...
     'run("examples/USC_HAD/step03_extract_window_features.m").']);
S=load(source,'X','Y','subjectID','trialID','recordID', ...
    'featureNames','windowSeconds','hopSeconds');
assert(all(isfinite(S.X(:))) && all(ismember(unique(S.Y),[0 1])), ...
    'Missing/invalid measured features or unexpected class labels.');
assert(isnumeric(featurePair) && numel(featurePair)==2 && ...
    all(featurePair==round(featurePair)) && all(featurePair>=1) && ...
    all(featurePair<=size(S.X,2)) && featurePair(1)~=featurePair(2), ...
    'Choose two different feature indices e.g. [1 2].');

% Trial-level analysis: preserve original trial identities.
ids=unique(S.recordID,'stable');
n=numel(ids);
X=zeros(n,size(S.X,2));Y=zeros(n,1);
subject=zeros(n,1);trial=zeros(n,1);nWindows=zeros(n,1);
for k=1:n
    mask=S.recordID==ids(k);
    X(k,:)=mean(S.X(mask,:),1);
    Y(k)=S.Y(find(mask,1));
    subject(k)=S.subjectID(find(mask,1));
    trial(k)=S.trialID(find(mask,1));
    nWindows(k)=nnz(mask);
    assert(all(S.Y(mask)==Y(k)) && ...
        all(S.subjectID(mask)==subject(k)) && ...
        all(S.trialID(mask)==trial(k)), ...
        'Trial windows unexpectedly have inconsistent metadata.');
end
development=[2 3 4 7 8 9 10 11 12 14];
diagnostic=[1 5 6 13];
train=ismember(subject,development);
test=ismember(subject,diagnostic);
assert(all(ismember(development,unique(subject))) && ...
    all(ismember(diagnostic,unique(subject))) && ...
    nnz(train)>0 && nnz(test)>0, ...
    'All 14 expected USC-HAD subjects are required for this lesson.');
assert(~any(train&test),'Subject leakage between sets.');
assert(all(ismember([0 1],unique(Y(train)))) && ...
    all(ismember([0 1],unique(Y(test)))), ...
    'Both motions required in development and diagnostic sets.');

fprintf('\nME 3310 Week 11 / Page 03: REAL USC-HAD trials\n');
fprintf('Original trials: %d; development %d; diagnostic %d.\n', ...
    n,nnz(train),nnz(test));
fprintf('Development subjects: %s\n',mat2str(development));
fprintf('Diagnostic subjects: %s (previously viewed; NOT pristine)\n', ...
    mat2str(diagnostic));
fprintf('ONE original trial = ONE model observation.\n');

M=struct;
M.mode=mode; M.classNames={'Walking Upstairs','Walking Downstairs'};
M.developmentSubjects=development;
M.diagnosticSubjects=diagnostic;
M.trialData=table(ids,subject,trial,Y,nWindows, ...
    'VariableNames',{'RecordID','Subject','Trial','ClassID','WindowCount'});
M.candidateResults=table();

if mode=="search"
    candidates=[1 2;1 3;1 5;1 9;2 12;9 12];
    scores=zeros(size(candidates,1),1);
    % Choose using development data only; never evaluate candidate pairs
    % on diagnostic subjects until the choice is locked.
    devSubjects=unique(subject(train))';
    for j=1:size(candidates,1)
        pair=candidates(j,:);
        cvPrediction=nan(n,1);
        for sid=devSubjects
            foldTest=train & subject==sid;
            foldTrain=train & subject~=sid;
            [zt,zv]=standardizeTrainOnly(X(foldTrain,pair), ...
                X(foldTest,pair));
            mdl=fitglm(zt,Y(foldTrain),'linear', ...
                'Distribution','binomial','Link','logit');
            cvPrediction(foldTest)=double(predict(mdl,zv)>=0.5);
        end
        scores(j)=balancedRate(Y(train),cvPrediction(train));
    end
    M.candidateResults=table(candidates(:,1),candidates(:,2),scores, ...
        'VariableNames',{'Feature1','Feature2', ...
        'LOSO_Dev_BalancedAccuracy_pct'});
    disp(M.candidateResults);
    [~,best]=max(scores); % first wins a tie: deliberately simple
    featurePair=candidates(best,:);
    fprintf(['Selected from SIX predefined pairs using ' ...
        'development-subject CV only: [%d %d]\n'], ...
        featurePair(1),featurePair(2));
    figure('Name','Page 03 | Development-only feature search');
    bar(scores,'FaceColor',[.25 .47 .66]);
    ylim([0 100]);grid on;
    xticks(1:numel(scores));xticklabels(compose('[%d %d]', ...
        candidates(:,1),candidates(:,2)));
    ylabel('Dev LOSO trial balanced accuracy (%)');
    xlabel('Candidate feature pair');
    title('Logistic model: CV feature screening, development subjects ONLY');
    fprintf(['Selection among six candidates is NOT proof of a ' ...
        'global optimum. CV scores can still be optimistic after search.\n']);
end

M.featurePair=featurePair;
M.featureNames=S.featureNames(featurePair);
fprintf('Selected features [%d %d]:\n  %s\n  %s\n', ...
    featurePair(1),featurePair(2),M.featureNames{1},M.featureNames{2});

% Standardization learned ONLY from development trials.
[Ztr,Zte,mu,sd]=standardizeTrainOnly(X(train,featurePair), ...
    X(test,featurePair));
M.trainingMean=mu;
M.trainingStd=sd;
logistic=fitglm(Ztr,Y(train),'linear', ...
    'Distribution','binomial','Link','logit');
svm=fitcsvm(Ztr,Y(train),'KernelFunction','linear', ...
    'Standardize',false,'ClassNames',[0 1]);
probTrain=predict(logistic,Ztr);
probDiag=predict(logistic,Zte);
predLogTrain=double(probTrain>=0.5);
predLogDiag=double(probDiag>=0.5);
predSVMTrain=predict(svm,Ztr);
predSVMDiag=predict(svm,Zte);

M.logisticModel=logistic;
M.svmModel=svm;
M.diagnosticLabels=Y(test);
M.diagnosticProbDownstairs=probDiag;
M.diagnosticPredLogistic=predLogDiag;
M.diagnosticPredSVM=predSVMDiag;
M.trainingReport=table( ...
    [balancedRate(Y(train),predLogTrain); ...
     balancedRate(Y(train),predSVMTrain)], ...
    'RowNames',{'Logistic','LinearSVM'}, ...
    'VariableNames',{'TrialBalancedAccuracy_pct'});
M.diagnosticReport=table( ...
    [balancedRate(Y(test),predLogDiag); ...
     balancedRate(Y(test),predSVMDiag)], ...
    'RowNames',{'Logistic','LinearSVM'}, ...
    'VariableNames',{'TrialBalancedAccuracy_pct'});
disp('Training scores (same original trials used for fitting):');
disp(M.trainingReport);
disp('Diagnostic subject scores (subjects excluded from this fitting):');
disp(M.diagnosticReport);

%% Figure 1: both learned LINEAR decision boundaries in raw units
rawTr=X(train,featurePair);rawTe=X(test,featurePair);
combined=[rawTr;rawTe];
range=rangePad(combined);
[g1,g2]=meshgrid(linspace(range(1,1),range(1,2),150), ...
                 linspace(range(2,1),range(2,2),150));
gridRaw=[g1(:),g2(:)];
gridZ=(gridRaw-mu)./sd;
logGrid=predict(logistic,gridZ);
[~,svmScores]=predict(svm,gridZ);

figure('Name','Page 03 | Logistic vs Linear SVM boundaries');
tiledlayout(1,2,'TileSpacing','compact');
for panel=1:2
    nexttile;hold on;
    if panel==1
        contour(g1,g2,reshape(logGrid,size(g1)),[0.5 0.5], ...
            'k','LineWidth',2,'DisplayName','P(Downstairs)=0.5');
        title('Logistic regression | probability threshold 0.5');
    else
        contour(g1,g2,reshape(svmScores(:,2),size(g1)),[0 0], ...
            'k','LineWidth',2,'DisplayName','SVM margin score = 0');
        title('Linear SVM | margin boundary');
    end
    drawPoints(rawTr,Y(train),'o',true,7);
    drawPoints(rawTe,Y(test),'o',false,12);
    xlabel(M.featureNames{1},'Interpreter','none');
    ylabel(M.featureNames{2},'Interpreter','none');
    grid on;
    if panel==1
        legend({'Boundary','Upstairs: development', ...
            'Downstairs: development','Upstairs: diagnostic', ...
            'Downstairs: diagnostic'},'Location','best');
    end
end
sgtitle('Original TRIAL means | diagnostic subjects not used to fit models');

%% Figure 2: diagnostic confusion matrices (ROWS: true, COLUMNS: predicted)
confLog=counts2x2(Y(test),predLogDiag);
confSVM=counts2x2(Y(test),predSVMDiag);
M.confusionLogistic=confLog;M.confusionSVM=confSVM;
figure('Name','Page 03 | Diagnostic trial confusion matrices');
tiledlayout(1,2,'TileSpacing','compact');
nexttile;showCounts(confLog,'Logistic regression');
nexttile;showCounts(confSVM,'Linear SVM');
sgtitle('Real diagnostic trials | subjects excluded from model fitting');

fprintf(['\nInterpretation: logistic returns model-estimated P(Downstairs), ' ...
    'not a guaranteed true probability.\n']);
fprintf(['SVM margin scores are NOT probabilities without separate ' ...
    'calibration.\n']);
fprintf(['Repeated tuning against diagnostic subjects invalidates ' ...
    'their use as a clean final test.\n']);
fprintf(['Project next step: collect NEW independent trial recordings ' ...
    'under stated measurement conditions.\n']);
end

function [ztr,zte,mu,sd]=standardizeTrainOnly(a,b)
mu=mean(a,1);sd=std(a,0,1);
sd(sd<1e-12)=1;
ztr=(a-mu)./sd;
zte=(b-mu)./sd;
end

function a=balancedRate(truth,pred)
up=truth==0;down=truth==1;
assert(any(up) && any(down),'Both motion classes are required.');
a=50*(mean(pred(up)==0)+mean(pred(down)==1));
end

function lim=rangePad(X)
lo=min(X,[],1); hi=max(X,[],1);
pad=max((hi-lo)*0.07,1e-9);
lim=[lo(1)-pad(1),hi(1)+pad(1); ...
     lo(2)-pad(2),hi(2)+pad(2)];
end

function drawPoints(X,y,style,isTrain,markerSize)
c=[0.12 0.39 0.69; 0.82 0.38 0.17];
for ci=0:1
    rows=y==ci;
    if isTrain
        scatter(X(rows,1),X(rows,2),markerSize^2,style, ...
            'filled','MarkerFaceColor',c(ci+1,:), ...
            'MarkerFaceAlpha',0.58,'MarkerEdgeColor','none');
    else
        scatter(X(rows,1),X(rows,2),markerSize^2,style, ...
            'MarkerEdgeColor',c(ci+1,:),'LineWidth',1.4);
    end
end
end

function cm=counts2x2(y,p)
cm=[sum(y==0&p==0),sum(y==0&p==1); ...
    sum(y==1&p==0),sum(y==1&p==1)];
end

function showCounts(cm,modelName)
imagesc(cm);axis equal tight;
colormap(gca,parula);colorbar;
xticks([1 2]);yticks([1 2]);
xticklabels({'Upstairs','Downstairs'});
yticklabels({'Upstairs','Downstairs'});
xlabel('Predicted');ylabel('Actual');
title(modelName);
for i=1:2
    for j=1:2
        text(j,i,num2str(cm(i,j)),'HorizontalAlignment','center', ...
            'FontSize',14,'FontWeight','bold','Color','k');
    end
end
end
