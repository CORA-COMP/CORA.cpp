function exportBenchmarkData(instances, varargin)
% exportBenchmarkData - writes the ARCH-COMP AFF instances of MATLAB CORA to benchmarks/data
%
% Each original script (examples/ARCHcompetition/linear/benchmark_linear_verify*_ARCH23_*.m)
% runs up to its verify call, on a copy in a temporary folder; the sys, params, options and spec
% it passes become <instance>.json (metadata, specifications, expected results) and
% <instance>.bin (matrices, exact doubles). The format is documented in benchmarks/README.md.
%
% Syntax:
%    exportBenchmarkData()
%    exportBenchmarkData(instances, 'coraRoot', coraRoot, 'expected', true, 'folder', folder)
%
% Inputs:
%    instances - (optional) cell array of instance names, default: all
%    coraRoot - (optional) MATLAB CORA folder, default: environment variable CORA_ROOT
%    expected - (optional) also run verify and store its result, iterations, nrSteps, timeStep
%    folder - (optional) output folder, default: benchmarks/data

% ------------------------------ BEGIN CODE -------------------------------

% benchmark name, instance, label printed by the benchmarks, script of the instance
table = { ...
    'Spacestation', 'ISSC01_ISS02', 'ISSC01-ISS02', 'verifyFast_ARCH23_iss_ISSC01_ISS02'; ...
    'Spacestation', 'ISSC01_ISU02', 'ISSC01-ISU02', 'verifyFast_ARCH23_iss_ISSC01_ISU02'; ...
    'Spacestation', 'ISSF01_ISS01', 'ISSF01-ISS01', 'verifyFast_ARCH23_iss_ISSF01_ISS01'; ...
    'Spacestation', 'ISSF01_ISU01', 'ISSF01-ISU01', 'verifyFast_ARCH23_iss_ISSF01_ISU01'; ...
    'Beam', 'CBC01', 'CBC01', 'verifyFast_ARCH23_beam_CBC01'; ...
    'Beam', 'CBC02', 'CBC02', 'verifyFast_ARCH23_beam_CBC02'; ...
    'Beam', 'CBC03', 'CBC03', 'verifyFast_ARCH23_beam_CBC03'; ...
    'Beam', 'CBF01', 'CBF01', 'verifyFast_ARCH23_beam_CBF01'; ...
    'Beam', 'CBF02', 'CBF02', 'verifyFast_ARCH23_beam_CBF02'; ...
    'Beam', 'CBF03', 'CBF03', 'verifyFast_ARCH23_beam_CBF03'; ...
    'Heat3D', 'HEAT01', 'HEAT01', 'verifyFast_ARCH23_heat3D_HEAT01'; ...
    'Heat3D', 'HEAT02', 'HEAT02', 'verifyFast_ARCH23_heat3D_HEAT02'; ...
    'Random', 'RAND01', 'RAND01', 'verify_ARCH23_rand_RAND01'; ...
    'Random', 'RAND02', 'RAND02', 'verify_ARCH23_rand_RAND02'};

p = inputParser;
addParameter(p, 'coraRoot', getenv('CORA_ROOT'));
addParameter(p, 'expected', true);
addParameter(p, 'folder', fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data'));
parse(p, varargin{:});
if nargin < 1 || isempty(instances)
    instances = table(:, 2)';
end
aux_addCora(p.Results.coraRoot);
if ~isfolder(p.Results.folder)
    mkdir(p.Results.folder);
end

for name = cellstr(instances)
    row = table(strcmp(table(:, 2), name{1}), :);
    if isempty(row)
        error('benchmarks:unknownInstance', 'Unknown instance: %s', name{1});
    end
    fprintf('%s ...\n', row{2});
    [text, rec] = aux_runScript(['benchmark_linear_' row{4}], p.Results.expected);
    aux_write(row, rec, text, p.Results.folder);
end

end


% Auxiliary functions -----------------------------------------------------

function aux_addCora(coraRoot)
% puts the repository, without its .git folders, on the path

if isempty(coraRoot) || ~isfolder(coraRoot)
    error('benchmarks:noCora', 'Pass coraRoot or set CORA_ROOT to the MATLAB CORA folder.');
end
paths = strsplit(genpath(coraRoot), pathsep);
addpath(paths{~contains(paths, [filesep '.git']) & ~cellfun(@isempty, paths)});

end

function [text, rec] = aux_runScript(script, runVerify)
% runs a copy of the original script that saves the verify arguments (and results) to a file

source = which(script);
if isempty(source)
    error('benchmarks:noScript', 'Script %s not found on the path.', script);
end
lines = splitlines(fileread(source));
at = find(contains(lines, 'verify(sys,params,options,spec);'));
if numel(at) ~= 1
    error('benchmarks:noVerify', 'Expected one verify call in %s.', script);
end

% the script's own verify line stays as is, between the two saves
folder = tempname;
mkdir(folder);
cleanup = onCleanup(@() aux_remove(folder));
recfile = fullfile(folder, 'rec.mat');
saveArgs = ['save(''' recfile ''',''sys'',''params'',''options'',''spec'');'];
if ~runVerify
    lines{at} = [saveArgs ' return;'];
elseif contains(lines{at}, '[res,fals,savedata]')
    saveResult = ['save(''' recfile ''',''savedata'',''-append'');'];
    lines{at} = [saveArgs newline lines{at} newline saveResult];
else
    lines{at} = [saveArgs newline lines{at}];
end
fid = fopen(fullfile(folder, [script '.m']), 'w');
fwrite(fid, strjoin(lines, newline));
fclose(fid);
addpath(folder, '-begin');
clear(script);

text = '';
if runVerify
    text = feval(script);
else
    feval(script);
end
rec = load(recfile);
if ~isfield(rec, 'savedata')
    rec.savedata = struct();
end

end

function aux_remove(folder)
% removes the temporary folder from the path and the disk
rmpath(folder);
rmdir(folder, 's');
end

function aux_write(row, rec, text, folder)
% writes <instance>.bin and <instance>.json

fid = fopen(fullfile(folder, [row{2} '.bin']), 'w');
closer = onCleanup(@() fclose(fid));
sys = rec.sys;
params = rec.params;

% the sets as center and generators; an absent input set is {0}
R0 = zonotope(params.R0);
if isfield(params, 'U')
    U = zonotope(params.U);
else
    U = zonotope(zeros(size(sys.B, 2), 1));
end
B = sys.B;
if isscalar(B) && size(sys.A, 1) > 1
    % a scalar B scales the input set (the beam scripts), i.e. B*I
    B = B * eye(size(sys.A, 1));
end
names = {'A', 'B', 'C', 'R0c', 'R0G', 'Uc', 'UG'};
values = {sys.A, B, sys.C, center(R0), generators(R0), center(U), generators(U)};
entries = cell(size(names));
for i = 1:numel(names)
    entries{i} = sprintf('    "%s": %s', names{i}, aux_matrix(fid, values{i}));
end

% the specifications as halfspaces {x | A x <= b}
specs = cell(1, numel(rec.spec));
for i = 1:numel(rec.spec)
    specs{i} = aux_spec(rec.spec(i), params.tFinal);
end

% the results of this run, without timings (those differ per machine)
expected = 'null';
if ~isempty(text)
    parts = strsplit(text, ',');
    expected = sprintf('{"verified": %d', str2double(parts{3}));
    for f = {'iterations', 'nrSteps', 'timeStep'}
        if isfield(rec.savedata, f{1})
            value = aux_num(rec.savedata.(f{1}));
            expected = [expected sprintf(', "%s": %s', f{1}, value)]; %#ok<AGROW>
        end
    end
    expected = [expected '}'];
end

json = sprintf(['{\n  "format": "cora-benchmark-1",\n  "benchmark": "%s",\n' ...
    '  "instance": "%s",\n  "label": "%s",\n  "verifyAlg": "%s",\n  "tFinal": %s,\n' ...
    '  "data": "%s.bin",\n  "matrices": {\n%s\n  },\n  "specs": [\n%s\n  ],\n' ...
    '  "expected": %s\n}\n'], row{1}, row{2}, row{3}, rec.options.verifyAlg, ...
    aux_num(params.tFinal), row{2}, strjoin(entries, sprintf(',\n')), ...
    strjoin(specs, sprintf(',\n')), expected);
jid = fopen(fullfile(folder, [row{2} '.json']), 'w');
fwrite(jid, json);
fclose(jid);

end

function str = aux_matrix(fid, M)
% appends M to the binary file and returns its json entry: coordinates (int32 rows, int32
% cols, float64 values; 0-based) if sparse enough, else float64 values row by row

M = full(M);
[r, c] = size(M);
[i, j, v] = find(M);
offset = ftell(fid);
if numel(v) * 16 < r * c * 8
    storage = 'coo';
    nnz = numel(v);
    fwrite(fid, i - 1, 'int32');
    fwrite(fid, j - 1, 'int32');
    fwrite(fid, v, 'double');
else
    storage = 'dense';
    nnz = r * c;
    fwrite(fid, M.', 'double');
end
str = sprintf('{"shape": [%d, %d], "storage": "%s", "offset": %d, "nnz": %d}', ...
    r, c, storage, offset, nnz);

end

function str = aux_spec(spec, tFinal)
% a specification as halfspaces: safeSet (inside all) or unsafeSet (avoid their intersection)

if ~ismember(spec.type, {'safeSet', 'unsafeSet'})
    error('benchmarks:specType', 'Unsupported specification type %s.', spec.type);
end
if ~isempty(spec.time) && ~(infimum(spec.time) == 0 && supremum(spec.time) >= tFinal)
    error('benchmarks:specTime', 'Specifications on a time subinterval are not supported.');
end
P = polytope(spec.set);
if ~isempty(P.Ae)
    error('benchmarks:specEquality', 'Equality constraints are not supported.');
end
rows = cell(1, size(P.A, 1));
for k = 1:size(P.A, 1)
    rows{k} = ['[' strjoin(arrayfun(@aux_num, P.A(k, :), 'UniformOutput', false), ', ') ']'];
end
str = sprintf('    {"type": "%s", "A": [%s], "b": [%s]}', spec.type, strjoin(rows, ', '), ...
    strjoin(arrayfun(@aux_num, P.b(:)', 'UniformOutput', false), ', '));

end

function str = aux_num(x)
% a number that reads back as the same double
str = sprintf('%.17g', x);
end

% ------------------------------ END OF CODE ------------------------------
