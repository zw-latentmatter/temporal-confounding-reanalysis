spm('defaults','fmri');
spm_get_defaults('cmdline',true);

job_file = getenv('DCM_PEB_JOB');
if isempty(job_file)
    error('Required environment variable DCM_PEB_JOB is empty');
end
if ~exist(job_file,'file')
    error('Missing PEB job MAT: %s',job_file);
end
job = load(job_file);

subjects = job.subjects(:);
input_paths = job.input_paths;
condition_sessions = job.condition_sessions(:);
condition_tasks = job.condition_tasks(:);
branch = strtrim(char(job.branch));
task = strtrim(char(job.task));
analysis_name = strtrim(char(job.analysis_name));
baseline_session = strtrim(char(job.baseline_session));
drug_session = strtrim(char(job.drug_session));
output_file = strtrim(char(job.output_file));
plain_output_file = strtrim(char(job.plain_output_file));
subject_peb_dir = strtrim(char(job.subject_peb_dir));
effect_column = double(job.effect_column(1));
effect_prefix = strtrim(char(job.effect_prefix));
group_Q = strtrim(char(job.group_Q));
bmc_rng_seed = double(job.bmc_rng_seed(1));
expected_roi_names = job.expected_roi_names(:);
for i = 1:numel(expected_roi_names)
    expected_roi_names{i} = strtrim(char(expected_roi_names{i}));
end
for i = 1:numel(condition_sessions)
    condition_sessions{i} = strtrim(char(condition_sessions{i}));
end
for i = 1:numel(condition_tasks)
    condition_tasks{i} = strtrim(char(condition_tasks{i}));
end

n_subjects = numel(subjects);
if size(input_paths,1) ~= n_subjects || size(input_paths,2) ~= 2 || numel(condition_sessions) ~= 2 || numel(condition_tasks) ~= 2
    error('Single-task PEB jobs require two condition rows per participant');
end
if numel(expected_roi_names) ~= 6
    error('The job must specify six ROI names');
end
if strcmp(baseline_session,drug_session)
    error('Baseline and drug session labels must differ');
end
if ~exist(fileparts(output_file),'dir')
    mkdir(fileparts(output_file));
end
if ~exist(subject_peb_dir,'dir')
    mkdir(subject_peb_dir);
end

valid_subjects = cell(0,1);
valid_dcms = cell(0,2);
failed_subjects = cell(0,1);
failed_sessions = cell(0,1);
failed_error_types = cell(0,1);
required_fields = {'M','Ep','Cp','F','a','options'};
for i = 1:n_subjects
    subject = strtrim(char(subjects{i}));
    participant_dcms = cell(2,1);
    participant_ok = true;
    paths = cell(2,1);
    for k = 1:2
        paths{k} = strtrim(char(input_paths{i,k}));
    end
    if strcmp(paths{1},paths{2})
        participant_ok = false;
        failed_subjects{end+1,1} = subject;
        failed_sessions{end+1,1} = 'both';
        failed_error_types{end+1,1} = 'duplicate_condition_path';
    end
    for k = 1:2
        try
            if ~exist(paths{k},'file')
                error('Missing DCM MAT file');
            end
            loaded_dcm = load(paths{k},'DCM');
            if ~isfield(loaded_dcm,'DCM') || ~isstruct(loaded_dcm.DCM)
                error('MAT file does not contain a DCM structure');
            end
            DCM = loaded_dcm.DCM;
            if ~all(isfield(DCM,required_fields)) || ~isfield(DCM.M,'pE') || ~isfield(DCM.M,'pC')
                error('Estimated DCM fields are incomplete');
            end
            if ~isstruct(DCM.Ep) || ~isfield(DCM.Ep,'A') || ~isequal(size(DCM.Ep.A),[6 6])
                error('Expected a 6x6 posterior A matrix');
            end
            if ~isequal(size(DCM.a),[6 6]) || any(double(DCM.a(:)) ~= 1)
                error('Expected a fully connected six-node model');
            end
            covariance = full(double(DCM.Cp));
            if ~isscalar(DCM.F) || ~isfinite(double(DCM.F)) || any(~isfinite(double(DCM.Ep.A(:)))) || any(~isfinite(covariance(:)))
                error('Non-finite DCM posterior or free energy');
            end
            if size(covariance,1) ~= size(covariance,2) || any(diag(covariance) < -1e-8)
                error('Invalid DCM posterior covariance');
            end
            qA = spm_find_pC(DCM,{'A'});
            if numel(qA) ~= 36
                error('Expected 36 estimable A parameters');
            end
            if ~isfield(DCM.options,'analysis') || ~strcmpi(strtrim(char(DCM.options.analysis)),'CSD')
                error('Expected CSD analysis');
            end
            if ~isfield(DCM,'Y') || ~isfield(DCM.Y,'dt') || abs(double(DCM.Y.dt) - 0.91) > 1e-9
                error('Expected repetition time 0.91 seconds');
            end
            if ~isfield(DCM,'xY') || numel(DCM.xY) ~= 6 || ~all(isfield(DCM.xY,'name'))
                error('Expected six named regions');
            end
            for r = 1:6
                if ~strcmp(strtrim(char(DCM.xY(r).name)),expected_roi_names{r})
                    error('Region order mismatch');
                end
            end
            if isfield(DCM,'reanalysis') && isfield(DCM.reanalysis,'branch') && ~strcmp(strtrim(char(DCM.reanalysis.branch)),branch)
                error('DCM branch metadata mismatch');
            end
            participant_dcms{k} = DCM;
        catch ME
            participant_ok = false;
            failed_subjects{end+1,1} = subject;
            failed_sessions{end+1,1} = condition_sessions{k};
            if isempty(ME.identifier)
                failed_error_types{end+1,1} = 'matlab_error';
            else
                failed_error_types{end+1,1} = ME.identifier;
            end
        end
    end
    if participant_ok
        valid_subjects{end+1,1} = subject;
        valid_dcms(end+1,:) = reshape(participant_dcms,1,[]);
    end
end

if numel(valid_subjects) < 2
    error('Fewer than two valid paired participants remain');
end
included_subjects = valid_subjects;
X = [];
Xnames = cell(0,1);
participant_X = [];
participant_Xnames = cell(0,1);

if strcmp(analysis_name,'author_stacked_01') || strcmp(analysis_name,'stacked_centered_pm05')
    coding = double(job.coding(:));
    if numel(coding) ~= 2
        error('Stacked analyses require two coding values');
    end
    n_valid = numel(valid_subjects);
    GCM = cell(2*n_valid,1);
    X = zeros(2*n_valid,2);
    for i = 1:n_valid
        baseline_row = 2*i - 1;
        drug_row = 2*i;
        GCM{baseline_row} = valid_dcms{i,1};
        GCM{drug_row} = valid_dcms{i,2};
        X(baseline_row,:) = [1 coding(1)];
        X(drug_row,:) = [1 coding(2)];
    end
    if strcmp(analysis_name,'author_stacked_01')
        Xnames = {'baseline_intercept','drug_psilocybin_minus_baseline'};
    else
        Xnames = {'grand_mean','drug_psilocybin_minus_baseline'};
    end
    M = struct('X',X,'Q',group_Q,'noplot',true);
    M.Xnames = Xnames;
    PEB = spm_dcm_peb(GCM,M,{'A'});
elseif strcmp(analysis_name,'paired_peb_of_pebs')
    participant_X = double(job.participant_X);
    participant_Xnames = job.participant_Xnames(:)';
    for i = 1:numel(participant_Xnames)
        participant_Xnames{i} = strtrim(char(participant_Xnames{i}));
    end
    if size(participant_X,1) ~= 2 || size(participant_X,2) ~= numel(participant_Xnames) || rank(participant_X) ~= size(participant_X,2)
        error('Invalid participant PEB design');
    end
    PEBs = cell(0,1);
    paired_subjects = cell(0,1);
    for i = 1:numel(valid_subjects)
        subject = valid_subjects{i};
        try
            Msub = struct('X',participant_X,'Q','none','noplot',true);
            Msub.Xnames = participant_Xnames;
            GCM_subject = reshape(valid_dcms(i,:),[],1);
            PEB_subject = spm_dcm_peb(GCM_subject,Msub,{'A'});
            safe_subject = regexprep(subject,'[^A-Za-z0-9_.-]','_');
            subject_file = fullfile(subject_peb_dir,[safe_subject '_PEB.mat']);
            save(subject_file,'PEB_subject','subject','branch','task','participant_X','participant_Xnames','condition_sessions','condition_tasks','-v7');
            PEBs{end+1,1} = PEB_subject;
            paired_subjects{end+1,1} = subject;
        catch ME
            failed_subjects{end+1,1} = subject;
            failed_sessions{end+1,1} = 'both';
            if isempty(ME.identifier)
                failed_error_types{end+1,1} = 'matlab_error';
            else
                failed_error_types{end+1,1} = ME.identifier;
            end
        end
    end
    if numel(PEBs) < 2
        error('Fewer than two participant PEBs remain');
    end
    included_subjects = paired_subjects;
    X = ones(numel(PEBs),1);
    Xnames = {'group_mean'};
    Mgroup = struct('X',X,'Q',group_Q,'noplot',true);
    Mgroup.Xnames = Xnames;
    PEB = spm_dcm_peb(PEBs,Mgroup);
else
    error('Unsupported analysis: %s',analysis_name);
end

rng(bmc_rng_seed,'twister');
[BMA,BMR] = spm_dcm_peb_bmc(PEB);
if ~isfield(BMA,'Pp')
    error('BMA output lacks parameter probabilities');
end
roi_names = expected_roi_names;
save(output_file,'PEB','BMA','BMR','included_subjects','roi_names','X','Xnames','participant_X','participant_Xnames','condition_sessions','condition_tasks','branch','task','analysis_name','baseline_session','drug_session','group_Q','failed_subjects','failed_sessions','failed_error_types','bmc_rng_seed','-v7.3');

PEB_Ep = full(PEB.Ep);
PEB_Cp = full(PEB.Cp);
BMA_Ep = full(BMA.Ep);
BMA_Cp = full(BMA.Cp);
BMA_Pp = full(BMA.Pp);
Pnames = PEB.Pnames;
if ischar(Pnames)
    Pnames = cellstr(Pnames);
end
Pnames = Pnames(:);
save(plain_output_file,'PEB_Ep','PEB_Cp','BMA_Ep','BMA_Cp','BMA_Pp','Pnames','Xnames','roi_names','effect_column','effect_prefix','included_subjects','failed_subjects','failed_sessions','failed_error_types','bmc_rng_seed','X','participant_X','participant_Xnames','condition_sessions','condition_tasks','baseline_session','drug_session','group_Q','-v7');
exit(0);
