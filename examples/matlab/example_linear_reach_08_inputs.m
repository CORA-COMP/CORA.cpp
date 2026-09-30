% example_linear_reach_08_inputs - reachability of a linear system with an input and an output in
%    CORA.cpp, converted into CORA (MATLAB) for plotting and for comparison with its own reach
%
% A mass-spring-damper x' = A x + B u, y = C x: a force u(t) in U acts on the mass, the position
% is measured. Setup as in example_linear_reach_01_5dim (python module of CORA.cpp, same python).
%
% Syntax:
%    pyenv(Version='<path to python>');   % once per MATLAB session, before the first python call
%    example_linear_reach_08_inputs
%
% Inputs:
%    -
%
% Outputs:
%    completed - true/false
%
% See also: examples/python/example_linear_reach_08_inputs.py

% ------------------------------ BEGIN CODE -------------------------------

% python module of CORA.cpp (folder set once in the macro or environment variable CORACPP_PYTHONPATH)
setUpCORAcpp();

% Parameters --------------------------------------------------------------

params.tFinal = 5;
params.R0 = zonotope([0;0],0.2*eye(2));
params.U = zonotope(0.5,0.25);

% Reachability Settings ---------------------------------------------------

options.timeStep = 0.05;
options.taylorTerms = 6;
options.zonotopeOrder = 20;

% System Dynamics ---------------------------------------------------------

A = [0 1; -4 -0.4]; B = [0; 1]; C = [1 0];
sys = linearSys(A,B,[],eye(2));   % states, to plot
sysY = linearSys(A,B,[],C);       % output: the position

% Reachability Analysis ---------------------------------------------------

% CORA.cpp: toPy converts the sets; the system takes its matrices as tensors (B and C are not
% covered by linearSys.toPy yet), fromPy the result
cora = coracpp();
sys_py = cora.linearSys(toTensor(A),toTensor(B),toTensor(C).reshape(int32(1),int32(2)));
tic
R_py = sys_py.reach(params.R0.toPy(),options.timeStep,params.tFinal,int32(options.taylorTerms), ...
    "standard",params.U.toPy(),int32(options.zonotopeOrder));
tComp = toc;
R_cpp = reachSet.fromPy(R_py,options.timeStep);
Y_cpp = reachSet.fromPy(sys_py.outputSet(R_py),options.timeStep);
disp("computation time of CORA.cpp: " + tComp + " s");

% CORA
tic
R_cora = reach(sys,params,options);
disp("computation time of CORA: " + toc + " s");
Y_cora = reach(sysY,params,options);

% Visualization -----------------------------------------------------------

% both results are reachSet objects: CORA plots them
figure; hold on;
plot(R_cora,[1 2],'DisplayName','CORA');
plot(R_cpp,[1 2],'DisplayName','CORA.cpp','Filled',false);
plot(params.R0,[1 2],'DisplayName','Initial set');
legend(Location='northwest');

% compare the final time-interval sets of the output
I_cpp = interval(Y_cpp.timeInterval.set{end});
I_cora = interval(Y_cora.timeInterval.set{end});
disp("largest difference of the final output hulls: " + ...
    max(abs([I_cpp.inf - I_cora.inf; I_cpp.sup - I_cora.sup])));

% example completed
completed = true;

% ------------------------------ END OF CODE ------------------------------
