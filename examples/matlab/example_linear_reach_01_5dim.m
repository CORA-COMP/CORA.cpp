% example_linear_reach_01_5dim - reachability of a five-dimensional linear system in CORA.cpp,
%    converted into CORA (MATLAB) for plotting and for comparison with its own reach
%
% Needs CORA (MATLAB) on the path, on the branch with toPy/fromPy, and the python module of
% CORA.cpp (preset 'python'); MATLAB and CORA.cpp have to use the same python (see pyenv).
%
% Syntax:
%    pyenv(Version='<path to python>');   % once per MATLAB session, before the first python call
%    example_linear_reach_01_5dim
%
% Inputs:
%    -
%
% Outputs:
%    completed - true/false
%
% See also: examples/python/example_linear_reach_01_5dim.py

% ------------------------------ BEGIN CODE -------------------------------

% folder of the python module: <CORA.cpp>/build/python unless set otherwise
if isempty(getenv('CORACPP_PYTHONPATH'))
    setenv('CORACPP_PYTHONPATH',fullfile(fileparts(mfilename('fullpath')),'..','..','build','python'));
end

% Parameters --------------------------------------------------------------

params.tFinal = 5;
params.R0 = zonotope(ones(5,1),0.1*eye(5));

% Reachability Settings ---------------------------------------------------

options.timeStep = 0.02;
options.taylorTerms = 4;
options.zonotopeOrder = 50;

% System Dynamics ---------------------------------------------------------

A = [-1 -4 0 0 0; 4 -1 0 0 0; 0 0 -3 1 0; 0 0 -1 -3 0; 0 0 0 0 -2];
fiveDimSys = linearSys(A,zeros(5,1));

% Reachability Analysis ---------------------------------------------------

% CORA.cpp: toPy converts the system and the initial set, fromPy the result
sys_py = fiveDimSys.toPy();
tic
R_py = sys_py.reach(params.R0.toPy(),options.timeStep,params.tFinal,int32(options.taylorTerms));
tComp = toc;
R_cpp = reachSet.fromPy(R_py,options.timeStep);
disp("computation time of CORA.cpp: " + tComp + " s");

% CORA
tic
R_cora = reach(fiveDimSys,params,options);
disp("computation time of CORA: " + toc + " s");

% Visualization -----------------------------------------------------------

% both results are reachSet objects: CORA plots them
for projDims = {[1 2],[3 4]}
    figure; hold on;
    plot(R_cora,projDims{1},'DisplayName','CORA');
    plot(R_cpp,projDims{1},'DisplayName','CORA.cpp','Filled',false);
    plot(params.R0,projDims{1},'DisplayName','Initial set');
    legend(Location='northwest');
end

% compare the final time-interval sets
I_cpp = interval(R_cpp.timeInterval.set{end});
I_cora = interval(R_cora.timeInterval.set{end});
disp("largest difference of the final interval hulls: " + ...
    max(abs([I_cpp.inf - I_cora.inf; I_cpp.sup - I_cora.sup])));

% example completed
completed = true;

% ------------------------------ END OF CODE ------------------------------
