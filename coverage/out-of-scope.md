# Out of scope

CORA methods CORA.cpp does not port, one row each with its reason (`group/class/method`). The
coverage tables count them separately; a method that a source tags cannot be listed.

| Row | Reason |
|---|---|
| `contSet/contSet/initEmptySet` | deprecated in CORA: replaced by contSet.empty |
| `contSet/contSet/isConHyperplane` | deprecated in CORA: replaced by representsa(S,'conHyperplane') |
| `contSet/contSet/isHyperplane` | deprecated in CORA: replaced by representsa(S,'hyperplane') |
| `contSet/contSet/isInterval` | deprecated in CORA: replaced by representsa(S,'interval') |
| `contSet/contSet/isParallelotope` | deprecated in CORA: replaced by representsa(S,'parallelotope') |
| `contSet/contSet/isPolytope` | deprecated in CORA: replaced by representsa(S,'polytope') |
| `contSet/contSet/isZero` | deprecated in CORA: replaced by representsa(S,'origin') |
| `contSet/contSet/isZonotope` | deprecated in CORA: replaced by representsa(S,'zonotope') |
| `contSet/contSet/isempty` | deprecated in CORA: replaced by representsa(S,'emptySet') |
| `contSet/polygon/simplify` | deprecated in CORA: replaced by compact(polygon,'douglasPeucker',tol) |
| `contSet/conZonotope/deleteZeros` | deprecated in CORA: replaced by compact(cZ,'zeros') |
| `contSet/polyZonotope/deleteZeros` | deprecated in CORA: replaced by compact(pZ,'zeros') |
| `contSet/zonotope/deleteZeros` | deprecated in CORA: replaced by compact(Z,'zeros') |
| `contSet/zonotope/deleteAligned` | deprecated in CORA: replaced by compact(Z,'aligned') |
| `nn/layers/instantiateFromString` | MATLAB string-to-layer dispatch for the legacy constructors; CORA.cpp builds the layer classes directly (and the Python front end lowers networks through `cora.OnnxNetwork.load`) |
| `nn/layers/nnActLayerFromHandle` | activation layer from a MATLAB function handle: its derivatives come from MATLAB symbolic differentiation of the handle, which CORA.cpp does not have; CORA.cpp has the activations it is used for as their own layers (`Activation::sin`, `Activation::cos`, and the GroupSort layer) |
| `nn/layers/nnGCNLayer` | graph-neural-network layer on MATLAB graph objects: CORA.cpp ports no GNN |
| `nn/layers/nnGNNGlobalPoolingLayer` | graph-neural-network layer on MATLAB graph objects: CORA.cpp ports no GNN |
| `nn/layers/nnGNNLayer` | abstract base of the graph-neural-network layers: CORA.cpp ports no GNN |
| `nn/layers/nnGNNLinearLayer` | graph-neural-network layer on MATLAB graph objects: CORA.cpp ports no GNN |
| `nn/layers/nnGNNProjectionLayer` | graph-neural-network layer on MATLAB graph objects: CORA.cpp ports no GNN |
| `nn/layers/nnVisualizationLayer` | plotting layer of the MATLAB visualization tools: display only |
| `nn/neuralNetwork/addVisualizationLayers` | inserts plotting layers: display only |
| `nn/neuralNetwork/clearVisualizationLayerPlots` | clears MATLAB figures: display only |
| `nn/neuralNetwork/convertDLToolboxNetwork` | converts a MATLAB Deep Learning Toolbox network; CORA.cpp's Python front end loads networks with `cora.OnnxNetwork.load` instead |
| `nn/neuralNetwork/convertToDLToolboxNetwork` | converts to a MATLAB Deep Learning Toolbox network; CORA.cpp exports no networks (it loads them with `cora.OnnxNetwork.load`) |
| `nn/neuralNetwork/display` | prints the network in MATLAB: display only |
| `nn/neuralNetwork/enumerateLayers` | assigns running layer ids for the MATLAB bookkeeping of derivative caches: plumbing, CORA.cpp indexes its layers directly |
| `nn/neuralNetwork/exportAsJSON` | writer of a MATLAB-specific JSON format; CORA.cpp exports no networks (it loads them with `cora.OnnxNetwork.load`) |
| `nn/neuralNetwork/exportAsStruct` | writer to a MATLAB struct; CORA.cpp exports no networks |
| `nn/neuralNetwork/exportNetworkAsCellArray` | writer to MATLAB cell arrays; CORA.cpp exports no networks |
| `nn/neuralNetwork/exportONNXNetwork` | writer through the MATLAB Deep Learning Toolbox; CORA.cpp exports no networks (it loads them with `cora.OnnxNetwork.load`) |
| `nn/neuralNetwork/generateFromString` | MATLAB string-to-network constructor for the legacy layer names; CORA.cpp builds the layer classes directly |
| `nn/neuralNetwork/getFromCellArray` | legacy constructor from MATLAB cell arrays of weights and biases; CORA.cpp's NeuralNetwork takes its layers directly |
| `nn/neuralNetwork/getNumMessagePassingSteps` | counts GNN message-passing layers: CORA.cpp ports no GNN |
| `nn/neuralNetwork/importFromJSON` | reader of a MATLAB-specific JSON format; CORA.cpp's Python front end loads networks with `cora.OnnxNetwork.load` |
| `nn/neuralNetwork/importFromStruct` | reader of a MATLAB struct; CORA.cpp's Python front end loads networks with `cora.OnnxNetwork.load` |
| `nn/neuralNetwork/readGNNdata` | reader of GNN datasets: CORA.cpp ports no GNN |
| `nn/neuralNetwork/readGNNetwork` | reader of GNN networks: CORA.cpp ports no GNN |
| `nn/neuralNetwork/readGNNnetwork` | reader of GNN networks: CORA.cpp ports no GNN |
| `nn/neuralNetwork/readJSONNetwork` | reader of a MATLAB-specific JSON format; CORA.cpp's Python front end loads networks with `cora.OnnxNetwork.load` |
| `nn/neuralNetwork/readNNetNetwork` | reader of the .nnet format; CORA.cpp's Python front end loads networks with `cora.OnnxNetwork.load` (the benchmarks read theirs through `bench::load_network`) |
| `nn/neuralNetwork/readNetwork` | dispatches on the file extension to the MATLAB readers; CORA.cpp's Python front end loads networks with `cora.OnnxNetwork.load` |
| `nn/neuralNetwork/readONNXNetwork` | reader through the MATLAB Deep Learning Toolbox; CORA.cpp loads ONNX files with `cora.OnnxNetwork.load` |
| `nn/neuralNetwork/readSherlockNetwork` | reader of the Sherlock text format; CORA.cpp's Python front end loads networks with `cora.OnnxNetwork.load` |
| `nn/neuralNetwork/readYMLNetwork` | reader of the YML format; CORA.cpp's Python front end loads networks with `cora.OnnxNetwork.load` |
| `nn/neuralNetwork/reduceGNNForNode` | reduces a GNN to the neighborhood of a node of a MATLAB graph: CORA.cpp ports no GNN |
| `nn/neuralNetwork/reset` | clears the MATLAB-side caches of the layers (stored bounds and derivative handles): plumbing, CORA.cpp keeps no such caches |
| `nn/neuralNetwork/resetBounds` | clears the MATLAB-side stored bounds of the layers: plumbing, CORA.cpp keeps no such state |
| `nn/neuralNetwork/resetGNN` | resets GNN layer state: CORA.cpp ports no GNN |
| `nn/neuralNetwork/setInputSize` | sets the MATLAB-side input size used by the visualization and GNN layers: plumbing |
| `nn/neuralNetwork/train` | training with the MATLAB Deep Learning Toolbox: CORA.cpp verifies trained networks and does not train |
| `nn/neuralNetwork/visLossLandscape` | plots a loss landscape: display only |

## Not comparable with CORALean

CORA methods that CORA.cpp implements but that CORALean will not formalize, one row each with its
reason. Their status stays as their tags give it; the benchmark report (`benchmarks.py`) leaves them
out of the bit-exact numerator and denominator, as it does the rows that CORALean marks out of scope.

| Row | Reason |
|---|---|
| `contDynamics/nonlinearSys/simulate` | numerical ODE integration (ode45-like) with no set to compare: CORALean has no ODE solver, and marks `contDynamics/contDynamics/simulateRandom` out of scope |
| `contDynamics/neurNetContrSys/simulate` | numerical ODE integration of the closed loop with no set to compare: CORALean has no ODE solver (only the theorem that a simulation lies in the reachable set), and marks `contDynamics/contDynamics/simulateRandom` out of scope |
| `contSet/contSet/plus` | the abstract class's sum dispatches on the operand classes and constructs nothing of its own: CORALean has the sums of the concrete sets, and no operation of the base class to compare |
| `nn/neuralNetwork/copyNeuralNetwork` | the copy constructor: a network is a value in CORA.cpp, so a copy computes nothing and there is no operation to compare |
