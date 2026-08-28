spm('defaults','fmri');
spm_get_defaults('cmdline',true);

input_file = getenv('DCM_INPUT_MAT');
output_file = getenv('DCM_OUTPUT_MAT');
branch = getenv('DCM_BRANCH');
if isempty(input_file) || isempty(output_file) || isempty(branch)
    error('Required environment variables: DCM_INPUT_MAT, DCM_OUTPUT_MAT, DCM_BRANCH');
end
if ~exist(input_file,'file')
    error('Missing DCM input MAT: %s',input_file);
end

loaded = load(input_file,'time_series','ROI_names','tr_seconds');
if ~isfield(loaded,'time_series') || ~isfield(loaded,'ROI_names') || ~isfield(loaded,'tr_seconds')
    error('Input MAT lacks time_series, ROI_names or tr_seconds');
end
time_series = double(loaded.time_series);
ROI_names = loaded.ROI_names;
TR = double(loaded.tr_seconds(1));
if ~iscell(ROI_names)
    error('ROI_names must be a MATLAB cell array');
end
ROI_names = reshape(ROI_names,1,[]);
if ndims(time_series) ~= 2 || size(time_series,2) ~= 6
    error('Expected time_series to have six columns, got %s',mat2str(size(time_series)));
end
if numel(ROI_names) ~= 6 || any(~isfinite(time_series(:))) || ~isfinite(TR) || TR <= 0
    error('Invalid DCM input values');
end

[output_dir,output_stem,output_ext] = fileparts(output_file);
if isempty(output_ext)
    output_ext = '.mat';
    output_file = fullfile(output_dir,[output_stem output_ext]);
end
if ~exist(output_dir,'dir')
    mkdir(output_dir);
end
partial_file = [tempname(output_dir) '.partial.mat'];

n = size(time_series,2);
v = size(time_series,1);
DCM.Y.y = time_series;
DCM.name = [output_stem output_ext];
for i = 1:n
    DCM.xY(i).name = ROI_names{i};
end
DCM.v = v;
DCM.n = n;
DCM.Y.name = ROI_names;
DCM.Y.dt = TR;
DCM.Y.X0 = zeros(v,1);
DCM.Y.Q = spm_Ce(ones(1,n)*v);
DCM.delays = repmat(DCM.Y.dt,DCM.n,1);
DCM.U.u = zeros(v,1);
DCM.U.name = {'null'};
DCM.a = ones(n,n);
DCM.b = zeros(n,n,0);
DCM.c = zeros(n,0);
DCM.d = zeros(n,n,0);
DCM.options.stochastic = 0;
DCM.options.nonlinear = 0;
DCM.options.two_state = 0;
DCM.options.analysis = 'CSD';
DCM.options.induced = 1;
DCM.options.maxnodes = n;
DCM.options.maxit = 256;
DCM.options.nograph = 1;
DCM.reanalysis.branch = branch;

save(partial_file,'DCM');
spm_dcm_fmri_csd(partial_file);
estimated = load(partial_file,'DCM','F','Ep','Cp');
if ~isfield(estimated,'DCM') || ~isfield(estimated,'F') || ~isfield(estimated,'Ep') || ~isfield(estimated,'Cp')
    error('SPM output is incomplete');
end
if ~isscalar(estimated.F) || ~isfinite(estimated.F) || isempty(estimated.Cp) || any(~isfinite(nonzeros(estimated.Cp))) || ~isfield(estimated.Ep,'A') || ~isfield(estimated.DCM,'version')
    error('SPM output failed validity checks');
end
[moved,message] = movefile(partial_file,output_file,'f');
if ~moved
    error('Could not publish DCM output: %s',message);
end
exit(0);
