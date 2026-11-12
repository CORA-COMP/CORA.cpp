// krylov - lean::Krylov, the float Krylov subspaces of CORALean (the `krylov.*` ops of the oracle)
//
// Syntax:     auto r = lean::Krylov::subspace(A, c, k0, step, tf, tol, eps);
// Operations: one function per op: arnoldi (Krylov.arnoldiFloat), error_bound (errorBoundFloat and
//             orderOkFloat), subspace (subspaceFloat) and windows (windowsFloat); p and q are
//             Newton steps per norm and squarings of the exponential, ignored at dtype nearest
// Outputs:    V (n, k + 1) and H (k + 1, k) as tensors of doubles, the scalars as doubles
// See also:   lean/request.h, lean/linear_sys.h, contDynamics/linear_sys/private/priv_krylov.h
//
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/oracle/request.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

/// The Arnoldi data of the passes made: nu = |c|, the vectors V (n, k + 1) and H (k + 1, k).
struct KrylovArnoldi {
    double nu = 0;
    cora::Tensor V, H;
};

/// The bound of the error of `order` passes with its factors and the order test.
struct KrylovBound : KrylovArnoldi {
    std::vector<double> residuals;
    double mu_A = 0, mu_T = 0, bound = 0;
    bool ok = false;
};

/// The subspace the order search returned.
struct KrylovSubspace : KrylovArnoldi {
    int64_t order = 0;
    bool ok = false;
    double bound = 0;
};

/// The subspace and, if it was accepted, the windows of its small system y' = T y from e_1.
struct KrylovWindows : KrylovSubspace {
    std::optional<std::vector<Zonotope>> time_interval, time_point;
};

class Krylov {
  public:
    /// `passes` passes of Arnoldi on A (n, n) from c (n, 1), breakdown threshold eps.
    static KrylovArnoldi arnoldi(const cora::Tensor &A, const cora::Tensor &c, int passes,
                                 double eps);

    /// The error bound and the order test of Arnoldi's `order` passes at the time tf.
    static KrylovBound error_bound(const cora::Tensor &A, const cora::Tensor &c, int order,
                                   double tf, double tol, double eps);

    /// The order search from k0 in steps of `step`, at most `fuel` growths (n + 1 if negative).
    static KrylovSubspace subspace(const cora::Tensor &A, const cora::Tensor &c, int k0, int step,
                                   double tf, double tol, double eps, int fuel = -1);

    /// The subspace as above and its windows over [0, tf] of step time_step.
    static KrylovWindows windows(const cora::Tensor &A, const cora::Tensor &c, int k0, int step,
                                 double tf, double tol, double eps, double time_step,
                                 int taylor_terms, int zonotope_order, int fuel = -1);
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
