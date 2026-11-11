function export_network_data(varargin)
% export_network_data - writes the networks of the ARCH-COMP AINNCS instances of MATLAB CORA
%
% Each network is built as in its example script (examples/contDynamics/neurNetContrSys/
% example_neuralNet_reach_*.m) and written as <name>.nn.json and <name>.nn.bin to
% examples/benchmarks/data/networks: the layers (weights W, bias b, activation) as JSON, the numbers as
% exact doubles in the binary file (little-endian, W row-major then b). A linear layer takes the
% activation layer that follows it (identity if none); an elementwise affine layer is written as
% the layer with W = diag(scale), so that the arithmetic stays that of CORA.
%
% Syntax:
%    export_network_data()
%    export_network_data('coraRoot', coraRoot, 'folder', folder)
%
% Inputs:
%    coraRoot - (optional) MATLAB CORA folder, default: environment variable CORA_ROOT
%    folder - (optional) output folder, default: examples/benchmarks/data/networks

% Authors:       Claude Sonnet 5.5
% Written:       11-November-2026
% Last update:   ---
% Reviewed:      ---

% ------------------------------ BEGIN CODE -------------------------------

p = inputParser;
addParameter(p, 'coraRoot', getenv('CORA_ROOT'));
addParameter(p, 'folder', fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', 'benchmarks', 'data', 'networks'));
parse(p, varargin{:});
if ~isfolder(p.Results.coraRoot)
    error('benchmarks:noCora', 'Pass coraRoot or set CORA_ROOT to the MATLAB CORA folder.');
end
paths = strsplit(genpath(p.Results.coraRoot), pathsep);
addpath(paths{~contains(paths, [filesep '.git']) & ~cellfun(@isempty, paths)});

% name, network as the example script builds it
nets = { ...
    'unicycle', @() aux_cell('controllerUnicycle.mat', 'ReLU', []); ...
    'acc', @() aux_acc(); ...
    'tora_relu', @() aux_cell('controllerTORA.mat', 'ReLU', []); ...
    'tora_tanh', @() aux_cell('nn_tora_relu_tanh.mat', {'ReLU', 'ReLU', 'ReLU', 'tanh'}, []); ...
    'tora_sigmoid', @() aux_cell('nn_tora_sigmoid.mat', 'sigmoid', []); ...
    'single_pendulum', @() neuralNetwork.readONNXNetwork('controller_single_pendulum.onnx'); ...
    'double_pendulum_less_robust', @() neuralNetwork.readONNXNetwork('controller_double_pendulum_less_robust.onnx'); ...
    'double_pendulum_more_robust', @() neuralNetwork.readONNXNetwork('controller_double_pendulum_more_robust.onnx'); ...
    'airplane', @() neuralNetwork.readONNXNetwork('controller_airplane.onnx'); ...
    'attitude_control', @() neuralNetwork.readONNXNetwork('attitude_control_3_64_torch.onnx')};

if ~isfolder(p.Results.folder)
    mkdir(p.Results.folder);
end
for i = 1:size(nets, 1)
    fprintf('%s ...\n', nets{i, 1});
    aux_write(nets{i, 1}, nets{i, 2}(), p.Results.folder);
end

end


% Auxiliary functions -----------------------------------------------------

function nn = aux_cell(file, actFun, ~)
% a network from the cell arrays W, b of a mat file
data = load(file);
nn = neuralNetwork.getFromCellArray(data.W, data.b, actFun);
end

function nn = aux_acc()
% the ACC network: the affine map of the state to the inputs of the network, then the controller
T_gap = 1.4; v_set = 30;
C = [0, 0, 0, 0, 0, 0; 0, 0, 0, 0, 0, 0; 0, 0, 0, 0, 1, 0; 1, 0, 0, -1, 0, 0; 0, 1, 0, 0, -1, 0];
k = [v_set; T_gap; 0; 0; 0];
data = load('controllerACC.mat');
actFun = [{'identity'}, repmat({'ReLU'}, [1, length(data.W)])];
nn = neuralNetwork.getFromCellArray([{C}, data.W], [{k}, data.b], actFun);
end

function aux_write(name, nn, folder)
% writes <name>.nn.json and <name>.nn.bin

layers = {};   % each: W, b, activation
for k = 1:numel(nn.layers)
    layer = nn.layers{k};
    switch class(layer)
        case 'nnLinearLayer'
            layers{end+1} = struct('W', double(layer.W), 'b', double(layer.b), 'act', 'identity'); %#ok<AGROW>
        case 'nnElementwiseAffineLayer'
            n = size(layers{end}.W, 1);
            scale = double(layer.scale) .* ones(n, 1);
            offset = double(layer.offset) .* ones(n, 1);
            layers{end+1} = struct('W', diag(scale), 'b', offset, 'act', 'identity'); %#ok<AGROW>
        case 'nnReLULayer'
            layers{end}.act = 'relu';
        case 'nnTanhLayer'
            layers{end}.act = 'tanh';
        case 'nnSigmoidLayer'
            layers{end}.act = 'sigmoid';
        case 'nnIdentityLayer'
            layers{end}.act = 'identity';
        otherwise
            error('benchmarks:layer', 'Layer %s of %s is not supported.', class(layer), name);
    end
end

fid = fopen(fullfile(folder, [name '.nn.bin']), 'w');
closer = onCleanup(@() fclose(fid));
offset = 0;
text = sprintf('{\n  "name": "%s",\n  "layers": [\n', name);
for k = 1:numel(layers)
    L = layers{k};
    [rows, cols] = size(L.W);
    fwrite(fid, L.W.', 'double');
    offW = offset; offset = offset + rows * cols;
    fwrite(fid, L.b(:), 'double');
    offB = offset; offset = offset + rows;
    sep = ','; if k == numel(layers), sep = ''; end
    text = [text sprintf('    {"out": %d, "in": %d, "activation": "%s", "W": %d, "b": %d}%s\n', ...
        rows, cols, L.act, offW, offB, sep)]; %#ok<AGROW>
end
text = [text sprintf('  ]\n}\n')];
fid2 = fopen(fullfile(folder, [name '.nn.json']), 'w');
fwrite(fid2, text);
fclose(fid2);

end

% ------------------------------ END OF CODE ------------------------------
