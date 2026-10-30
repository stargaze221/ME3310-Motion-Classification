function D = week11_02_fisher_lda_demo(C, featurePair)
% ME3310 WEEK 11 | 02 - Feature Comparison, Fisher Score and LDA
%
% Work with REAL USC-HAD original trials summarized by the companion
% week11_02_feature_compare_demo function. No generated measurement data.
% Run from the repository root after preparing USC-HAD and window features:
%
%   addpath(fullfile(pwd,"examples","Student_IMU"));
%   C = week11_02_feature_compare_demo(2,[1 2]);
%   D = week11_02_fisher_lda_demo(C,[1 2]);
%   disp(D.fisherTable)
%   D = week11_02_fisher_lda_demo(C,[1 3]);
%
% Comparison is descriptive. Original trials are independent experimental
% repetitions; overlapping windows WITHIN a trial are not repetitions.
% Fisher Score is univariate. The LDA direction is learned on the plotted
% data, so apparent separation is NOT held-out classification accuracy.
% No Statistics and Machine Learning Toolbox required for Fisher/LDA.

if nargin<1 || isempty(C)
    error(['First run: C = week11_02_feature_compare_demo(2,[1 2]); ' ...
           'This function requires real measured USC-HAD trial features.']);
end
if nargin<2 || isempty(featurePair), featurePair=[1 2];end
assert(isstruct(C) && isfield(C,'trialMeanOfAllFeatures') && ...
    isfield(C,'trialFeatures') && isfield(C,'featureNames'), ...
    'C must be the output of week11_02_feature_compare_demo.');
X = double(C.trialMeanOfAllFeatures); % one row per original trial
y = C.trialFeatures.ClassID;
[~,nFeat] = size(X);
assert(all(isfinite(X(:))) && all(ismember(unique(y),[0 1])) && ...
    any(y==0) && any(y==1), 'Invalid feature values or missing class.');
assert(isnumeric(featurePair) && numel(featurePair)==2 && ...
    all(featurePair==round(featurePair)) && ...
    all(featurePair>=1) && all(featurePair<=nFeat) && ...
    featurePair(1)~=featurePair(2), ...
    'featurePair must contain two different feature indices.');
nUp=nnz(y==0);nDown=nnz(y==1);
assert(nUp>=3 && nDown>=3, ...
    'Need at least 3 original trials per activity for this exploratory demo.');

%% 1. Each feature: distance between class means vs repeat variability
up=X(y==0,:);down=X(y==1,:);
muUp=mean(up,1);muDown=mean(down,1);
sUp=std(up,0,1);sDown=std(down,0,1);
within=sUp.^2+sDown.^2;
fisher=nan(1,nFeat);
valid=within>0;
fisher(valid)=((muUp(valid)-muDown(valid)).^2)./within(valid);
% When BOTH variances are zero, the score is undefined. Do not invent it.
if any(~valid)
    fprintf(['Some features have zero observed within-class variation; ' ...
        'their Fisher Scores are undefined.\n']);
end
D=struct;
D.subjectSelection=C.subjectSelection;
D.featurePair=featurePair;
D.nUpstairs=nUp;
D.nDownstairs=nDown;
D.fisherTable=table((1:nFeat)',string(C.featureNames(:)), ...
    muUp',sUp',muDown',sDown',fisher', ...
    'VariableNames',{'Index','Feature','UpMean','UpStd', ...
    'DownMean','DownStd','FisherScore'});
D.selectedFisherScores=fisher(featurePair);
disp(D.fisherTable(:,{'Index','Feature','FisherScore'}));
fprintf('Each Fisher Score uses %d upstairs and %d downstairs original trials.\n', ...
    nUp,nDown);

figure('Name','ME3310 | Fisher Score for measured trial features');
bar(1:nFeat,fisher,'FaceColor',[.24 .48 .66]);hold on;
for j=1:2
    idx=featurePair(j);
    plot(idx,fisher(idx),'o','MarkerFaceColor',[.9 .44 .14], ...
         'MarkerEdgeColor','k','MarkerSize',9);
end
xticks(1:nFeat);xlim([.5 nFeat+.5]);grid on;
xlabel('Feature index (names printed in MATLAB table)');
ylabel('Fisher Score (dimensionless)');
title(sprintf('%s | Exploratory separability, not test accuracy', ...
    C.subjectSelection),'Interpreter','none');

%% 2. Feature redundancy: examine class-conditional correlation
p=featurePair;
A=X(:,p);
for ci=0:1
    V=A(y==ci,:);
    r=corrcoef(V(:,1),V(:,2));
    if numel(r)>=4 && isfinite(r(1,2))
        fprintf('Class %d correlation between features %d and %d: %.3f\n', ...
            ci,p(1),p(2),r(1,2));
    else
        fprintf('Class %d feature correlation not estimable.\n',ci);
    end
end
fprintf('Correlated features can provide overlapping information.\n');

%% 3. Two-feature Fisher Linear Discriminant: supervised 1D projection
% Standardization makes the two feature units comparable for plotting.
% Small samples can make covariance ill-conditioned; apply a fixed ridge.
% This is NOT a held-out validated classifier.
center=mean(A,1);
scale=std(A,0,1);
if any(scale<=0)
    error('One selected feature has zero observed scale. Choose another pair.');
end
Q=(A-center)./scale;
qUp=Q(y==0,:);qDown=Q(y==1,:);
meanUp=mean(qUp,1);meanDown=mean(qDown,1);
S=cov(qUp)+cov(qDown);    % pooled within-class covariance, up to scaling
shrinkage=0.10;           % stabilizer, not a fitted hyperparameter
ridge=shrinkage*max(trace(S)/2,eps);
w=(S+ridge*eye(2))\(meanDown-meanUp)';
assert(all(isfinite(w)) && norm(w)>0, ...
    'Cannot estimate LDA direction for the chosen feature pair.');
w=w/norm(w);
if dot((meanDown-meanUp),w')<0,w=-w;end
score=Q*w;
D.ldaDirectionStandardized=w;
D.ldaProjection=score;
D.ldaRegularization=ridge;
D.ldaFeatureMeans=center;
D.ldaFeatureScales=scale;
D.ldaMeans=[mean(score(y==0));mean(score(y==1))];

figure('Name','ME3310 | Two-feature LDA exploration');
tiledlayout(1,2,'TileSpacing','compact');
colors=[.11 .40 .67;.80 .36 .19];
nexttile;hold on;
for ci=0:1
    g=Q(y==ci,:);
    scatter(g(:,1),g(:,2),62,'filled', ...
        'MarkerFaceColor',colors(ci+1,:), ...
        'MarkerEdgeColor','w', ...
        'DisplayName',C.classNames{ci+1});
end
mid=(meanUp+meanDown)/2;
span=max(1,max(abs(Q(:))));
pts=[mid(:)-span*w,mid(:)+span*w];
plot(pts(1,:),pts(2,:),'k--','LineWidth',1.6, ...
    'DisplayName','LDA projection direction');
quiver(mid(1),mid(2),0.55*span*w(1),0.55*span*w(2),0, ...
    'Color','k','LineWidth',1.9,'MaxHeadSize',.8, ...
    'HandleVisibility','off');
plot(meanUp(1),meanUp(2),'kx','MarkerSize',11,'LineWidth',1.8, ...
    'HandleVisibility','off');
plot(meanDown(1),meanDown(2),'kx','MarkerSize',11,'LineWidth',1.8, ...
    'HandleVisibility','off');
xlabel(['Standardized: ' C.featureNames{p(1)}],'Interpreter','none');
ylabel(['Standardized: ' C.featureNames{p(2)}],'Interpreter','none');
title('Two original features and LDA direction');grid on;
legend('Location','best');

nexttile;hold on;
for ci=0:1
    vals=score(y==ci);
    horizontal=ci+1+0.10*linspace(-1,1,numel(vals))';
    scatter(horizontal,vals,55,'filled', ...
        'MarkerFaceColor',colors(ci+1,:),'MarkerEdgeColor','w');
    plot(ci+1,mean(vals),'kd','MarkerSize',8,'MarkerFaceColor','k');
end
xticks([1 2]);xticklabels(C.classNames);xlim([.5 2.5]);
ylabel('LDA projection score (standardized feature space)');
title('Project the same measured trials onto one axis');grid on;
sgtitle(sprintf('%s | Fitted and viewed on SAME trials', ...
    C.subjectSelection),'Interpreter','none');

fprintf(['LDA uses labels to FIND a separating direction; ' ...
    'PCA would look for variation without class labels.\n']);
fprintf(['The LDA plot is an in-sample visualization, NOT proof of ' ...
    'performance on a new person or new trial.\n']);
fprintf(['Next experiment: hold out a new original trial or subject ' ...
    'BEFORE choosing features or estimating LDA.\n']);
end
