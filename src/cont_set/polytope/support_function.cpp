// support_function - the support function of a polytope, as CORA's polytope.supportFunc_
//
// A vertex polytope takes the largest dot product with its vertices (-Inf without vertices); a
// halfspace polytope solves a linear program (-Inf if empty, Inf if unbounded in the direction).
// R^n has Inf along every direction but zero, where the supremum of 0'x is 0 (CORA gives Inf). The
// direction must be a single column for halfspaces.
//
// Syntax:   rho = P.support_function(d);
// Inputs:   d - direction, a column (..., n, 1)
// Outputs:  rho - max_{x in P} d'x, shape (..., 1, 1)
// See also: Polytope::interval, Polytope::contains
//
// CORA: contSet/@polytope/supportFunc_.m
// CORA: contSet/@polytope/private/priv_supportFunc.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// The supremum over R^n: Inf unless the direction is zero.
Tensor aux_fullspace(const Tensor &d) {
    const Tensor value = d.abs().max(-2, true);
    return Tensor::where(value.greater(value.zeros_like()), value.ones_like() * kInf,
                         value.zeros_like());
}

/// The largest dot product of the direction with the columns of V.
Tensor aux_vertices(const Tensor &V, const Tensor &d) {
    if (V.shape().back() == 0) return priv_full(d, 1, 1, -kInf);
    return d.matrix_transpose().matmul(V).max(-1, true);
}

/// The value of the linear program along a single direction.
Tensor aux_program(const Polytope &P, const Tensor &d) {
    if (d.ndim() != 2)
        CORA_FAIL(Error::not_supported, "support_function: a linear program per direction: it "
                                        "supports a single direction for halfspaces");
    const HostH H = priv_host_h(P.constraints());
    return priv_column_tensor(P.A(), {priv_support(H, d.data(), true).value});
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor Polytope::support_function_impl(const Tensor &d) const {
    // R^n answers without a program, the vertices by their dot products
    if (priv_is_fullspace(*this) && !is_v_rep()) return aux_fullspace(d);
    if (is_v_rep()) return aux_vertices(V_, d);
    return aux_program(*this, d).reshape({1, 1});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
