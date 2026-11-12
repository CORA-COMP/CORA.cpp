// krylov - lean::Krylov, the float Krylov subspaces of CORALean (the `krylov.*` ops of the oracle)
//
// Syntax:     auto r = lean::Krylov::subspace(A, c, k0, step, tf, tol, eps);
// Operations: one function per op: arnoldi (Krylov.arnoldiFloat), error_bound (errorBoundFloat and
//             orderOkFloat), subspace (subspaceFloat), windows (windowsFloat), lyapunov_certificate
//             (lyapunovCertificateFloat) and restarted_bound (restartedErrorBoundFloat); p and q
//             are Newton steps per norm and squarings of the exponential, ignored at dtype nearest
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

/// A checked Lyapunov weight: accepted, the bounds a' and b of P, the rate of e^{tA} in it.
struct KrylovCertificate {
    bool ok = false;
    double lower = 0, upper = 0, rate = 0;
};

/// The certificate and the 2-norm error bound of restarted windows in its weight.
struct KrylovRestarted : KrylovCertificate {
    bool windows_ok = false;
    double bound = 0;
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

    /// The check of the weight P with the factor L of P - aI for A (all n x n).
    static KrylovCertificate lyapunov_certificate(const cora::Tensor &A, const cora::Tensor &P,
                                                  const cora::Tensor &L, double a);

    /// The error bound after the windows of lengths dt (N, 1) with local errors eps (N, 1).
    static KrylovRestarted restarted_bound(const cora::Tensor &A, const cora::Tensor &P,
                                           const cora::Tensor &L, double a, const cora::Tensor &dt,
                                           const cora::Tensor &eps);
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
