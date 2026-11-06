// cora - every public header of CORA.cpp in one include
//
// A program that uses the library includes this, or the headers of what it uses. libtorch's
// backend is part of it when the library was built with libtorch (CORACPP_TORCH).
//
// Syntax:   #include "cora/cora.h"
// See also: docs/architecture.md, CMake: find_package(CORA) and target CORA::cora

#pragma once

#include "cora/config.h"

// dynamic systems, their problems and algorithms
#include "cora/cont_dynamics/algorithms.h"
#include "cora/cont_dynamics/cont_dynamics.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
// sets, the type-erased Set and the binary registry
#include "cora/cont_set/capsule/capsule.h"
#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/cont_set/zonotope/zonotope.h"
// matrix sets
#include "cora/matrix_set/interval_matrix/interval_matrix.h"
// tensors, backends, errors, settings, plotting and the CORALean oracle
#include "cora/global/backend/eigen.h"
#include "cora/global/backend/lean.h"
#include "cora/global/error.h"
#include "cora/global/expr.h"
#include "cora/global/linprog.h"
#include "cora/global/options.h"
#include "cora/global/oracle/empty_set.h"
#include "cora/global/oracle/interval_matrix.h"
#include "cora/global/oracle/linear_sys.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/request.h"
#include "cora/global/plot/color.h"
#include "cora/global/plot/figure.h"
#include "cora/global/plot/plot.h"
#include "cora/global/rng.h"
#include "cora/global/settings.h"
#include "cora/global/tensor/tensor.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/threads.h"
#include "cora/global/within_tol.h"
// neural networks and specifications
#include "cora/nn/neural_network/neural_network.h"
#include "cora/specification/specification.h"
#ifdef CORACPP_TORCH // NOLINT(backend): the umbrella header includes the torch backend
#include "cora/global/backend/torch.h"
#endif

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ---------------------------------------  END OF CODE  ---------------------------------------- //
