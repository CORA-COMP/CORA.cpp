// nonlinear_sys - a linear system as a nonlinear one, as CORA's linearSys.nonlinearSys
//
// The dynamics f(x) = A x, written on the symbolic states. CORA.cpp's NonlinearSys has no inputs
// and no output map, so only a system without B and with C = I (or none) converts.
//
// Syntax:   NonlinearSys sys2 = sys.nonlinear_sys();
// Inputs:   sys - linear system
// Outputs:  sys2 - the nonlinear system x' = A x
// See also: cont_dynamics/nonlinear_sys/nonlinear_sys.h
//
// CORA: contDynamics/@linearSys/nonlinearSys.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cont_dynamics/linear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

NonlinearSys LinearSys::nonlinear_sys() const {
    detail::Entry entry;
    CORA_CHECK(A_.ndim() == 2, Error::not_supported, [] {
        return std::string("nonlinear_sys: it supports a single system, not a batch of "
                           "matrices A");
    });
    CORA_CHECK(!B_, Error::not_supported, [] {
        return std::string("nonlinear_sys: a system with inputs does not convert; NonlinearSys "
                           "has no inputs");
    });
    CORA_CHECK(!C_ || priv_is_identity(*C_), Error::not_supported, [] {
        return std::string("nonlinear_sys: an output matrix C other than the identity does not "
                           "convert; NonlinearSys has no output map");
    });
    const int64_t n = dim();
    const std::vector<double> a = A_.data();
    // f_i(x) = sum_j A_ij x_j on the symbolic states
    const Dynamics f = [n, a](const std::vector<Expr> &x) {
        std::vector<Expr> out;
        for (int64_t i = 0; i < n; ++i) {
            Expr row(0.0);
            for (int64_t j = 0; j < n; ++j) row = row + Expr(a[i * n + j]) * x[j];
            out.push_back(row);
        }
        return out;
    };
    return NonlinearSys(f, n);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
