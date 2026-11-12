// krylov - lean::Krylov, the float Krylov subspaces computed by CORALean
//
// See also: lean/request.h
//
// CORA.cpp only: the client of the CORALean oracle; CORA has no counterpart.
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/krylov.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The tensor of the reply's member `key` as doubles.
cora::Tensor aux_gather(const Reply &r, const char *key) { return r.get<Tensor>(key).gather(); }

/// The members every op returns: nu, V and H.
void aux_arnoldi(const Reply &r, KrylovArnoldi &out) {
    out.nu = r.get<double>("nu");
    out.V = aux_gather(r, "V");
    out.H = aux_gather(r, "H");
}

void aux_subspace(const Reply &r, KrylovSubspace &out) {
    aux_arnoldi(r, out);
    out.order = r.get<int64_t>("order");
    out.ok = r.get<bool>("ok");
    out.bound = r.get<double>("bound");
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The Arnoldi data and the bound ------------------------------------------------------------------

KrylovArnoldi Krylov::arnoldi(const cora::Tensor &A, const cora::Tensor &c, int passes,
                              double eps) {
    // the matrix, the start vector, the passes and the breakdown threshold
    const Reply r =
        ask("krylov.arnoldi",
            {{"A", Tensor::from(A)}, {"c", Tensor::from(c)}, {"passes", passes}, {"eps", eps}});
    KrylovArnoldi out;
    aux_arnoldi(r, out);
    return out;
}

KrylovBound Krylov::error_bound(const cora::Tensor &A, const cora::Tensor &c, int order, double tf,
                                double tol, double eps) {
    // as arnoldi, with the order, the time and the tolerance of the test
    const Reply r = ask("krylov.errorBound", {{"A", Tensor::from(A)},
                                              {"c", Tensor::from(c)},
                                              {"order", order},
                                              {"tf", tf},
                                              {"tol", tol},
                                              {"eps", eps}});
    // the Arnoldi data, then the factors of the bound and the test
    KrylovBound out;
    aux_arnoldi(r, out);
    out.residuals = r.get<std::vector<double>>("residuals");
    out.mu_A = r.get<double>("muA");
    out.mu_T = r.get<double>("muT");
    out.bound = r.get<double>("bound");
    out.ok = r.get<bool>("ok");
    return out;
}

// The order search and the windows --------------------------------------------------------------

KrylovSubspace Krylov::subspace(const cora::Tensor &A, const cora::Tensor &c, int k0, int step,
                                double tf, double tol, double eps, int fuel) {
    // the order search from k0 in steps of `step`
    std::vector<Operand> request{{"A", Tensor::from(A)},
                                 {"c", Tensor::from(c)},
                                 {"k0", k0},
                                 {"step", step},
                                 {"tf", tf},
                                 {"tol", tol},
                                 {"eps", eps}};
    if (fuel >= 0) request.push_back({"fuel", fuel});
    KrylovSubspace out;
    aux_subspace(ask("krylov.subspace", request), out);
    return out;
}

KrylovWindows Krylov::windows(const cora::Tensor &A, const cora::Tensor &c, int k0, int step,
                              double tf, double tol, double eps, double time_step, int taylor_terms,
                              int zonotope_order, int fuel) {
    // the order search and the reach options of the small system
    std::vector<Operand> request{{"A", Tensor::from(A)},
                                 {"c", Tensor::from(c)},
                                 {"k0", k0},
                                 {"step", step},
                                 {"tFinal", tf},
                                 {"tol", tol},
                                 {"eps", eps},
                                 {"timeStep", time_step},
                                 {"taylorTerms", taylor_terms},
                                 {"zonotopeOrder", zonotope_order}};
    if (fuel >= 0) request.push_back({"fuel", fuel});
    const Reply r = ask("krylov.windows", request);
    KrylovWindows out;
    aux_subspace(r, out);
    if (out.ok) {
        out.time_interval = r.get<std::vector<Zonotope>>("timeInterval");
        out.time_point = r.get<std::vector<Zonotope>>("timePoint");
    }
    return out;
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
