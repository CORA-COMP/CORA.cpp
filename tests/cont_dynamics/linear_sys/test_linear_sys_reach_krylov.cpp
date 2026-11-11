// test_linear_sys_reach_krylov - linear_sys reach with alg::Krylov: a 1-D flow by hand, agreement
//
// with alg::Standard on a heat grid, enclosure of trajectories (constant inputs), a sparse A,
// checks
//
// CORA: contDynamics/@linearSys/private/priv_reach_krylov.m
// PROPERTY: contDynamics/@linearSys/private/priv_reach_krylov.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_testing.h"
#include "cora/global/rng.h"
#include "cora/global/sparse_matrix.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;
using namespace test::lin;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The heat equation on a k x k x k grid with insulated sides (a 7-point stencil), sparse.
SparseMatrix heat_grid(int64_t k) {
    std::vector<int64_t> rows, cols;
    std::vector<double> vals;
    const int64_t n = k * k * k;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t coord[3] = {i % k, (i / k) % k, i / (k * k)};
        const int64_t stride[3] = {1, k, k * k};
        double diag = 0;
        // a neighbor on each side that is inside the grid
        for (int d = 0; d < 3; ++d)
            for (const int64_t s : {-1, 1}) {
                const int64_t q = coord[d] + s;
                if (q < 0 || q >= k) continue;
                rows.push_back(i);
                cols.push_back(i + s * stride[d]);
                vals.push_back(0.5);
                diag -= 0.5;
            }
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(diag - 0.1); // a little loss everywhere, so the flow decays
    }
    return {n, n, rows, cols, vals};
}

/// x' = -x from 1 + [-0.5, 0.5]: the subspaces are one-dimensional and exact, so the time point
/// k is e^{-k dt} times the start, up to rounding.
void flow_in_one_dimension() {
    const LinearSys sys(Tensor({{-1.0}}), Tensor({{0.0}}));
    const Zonotope R0(Tensor({{1.0}}), Tensor({{0.5}}));
    const double dt = 0.1;
    const ReachSet R = sys.reach({.initial_set = R0, .t_final = 1.0}, alg::Krylov{.time_step = dt});
    check(R.time_point.size() == 11 && R.time_interval.size() == 10, "krylov: 10 steps");
    for (size_t k = 0; k < R.time_point.size(); ++k) {
        const double e = std::exp(-double(k) * dt);
        const Interval I = R.time_point[k].interval();
        check(std::abs(I.inf().item() - 0.5 * e) < 1e-12 &&
                  std::abs(I.sup().item() - 1.5 * e) < 1e-12,
              "krylov: time point " + std::to_string(k) + " is e^{-t} R0");
    }
    // the step's enclosure spans both of its time points
    const Interval first = R.time_interval[0].interval();
    check(first.inf().item() <= 0.5 * std::exp(-dt) && first.sup().item() >= 1.5,
          "krylov: the first step's enclosure");
}

/// With a callback, reach passes each step's time-interval set to it and stores nothing: the
/// sets are those of the call without it.
void callback_receives_the_time_intervals() {
    const LinearSys sys(heat_grid(3), Tensor::zeros({27, 1}), Tensor::eye(27));
    const Zonotope R0(Tensor::ones({27, 1}), Tensor::eye(27) * 0.1);
    const ReachProblem problem{.initial_set = R0, .t_final = 0.3};
    const alg::Krylov krylov{.time_step = 0.1, .zonotope_order = 2};
    const ReachSet kept = sys.reach(problem, krylov);
    std::vector<Zonotope> seen;
    const ReachSet none = sys.reach(problem, krylov, [&](const Zonotope &Z) { seen.push_back(Z); });
    check(none.time_interval.empty() && none.time_point.empty(), "krylov callback: nothing stored");
    check(seen.size() == kept.time_interval.size(), "krylov callback: one set per step");
    for (size_t k = 0; k < seen.size(); ++k) {
        const Interval a = seen[k].interval(), b = kept.time_interval[k].interval();
        check(a.inf().data() == b.inf().data() && a.sup().data() == b.sup().data(),
              "krylov callback: step " + std::to_string(k) + " equals the stored set");
    }
}

/// On a 4 x 4 x 4 heat grid (n = 64), with subspaces accurate to eps, the output time
/// points agree with those of alg::Standard; the sparse and the dense A give the same sets.
void agrees_with_standard() {
    const SparseMatrix A = heat_grid(4);
    const int64_t n = A.rows();
    cora::Rng rng(5);
    const Zonotope R0 = Zonotope::generate_random(rng, n, std::nullopt, 3);
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(2, n);
    C(0, 0) = 1.0;
    C(1, 21) = 1.0;
    // the same system with a sparse and a dense A, and alg::Standard's outputs
    const Tensor B = Tensor::zeros({n, 1});
    const LinearSys sparse(A, B, tensor_of(C)), dense(A.dense(), B, tensor_of(C));
    const alg::Krylov krylov{.time_step = 0.05, .zonotope_order = 200};
    const ReachSet Rk = sparse.reach({.initial_set = R0, .t_final = 1.0}, krylov);
    const ReachSet Rd = dense.reach({.initial_set = R0, .t_final = 1.0}, krylov);
    const ReachSet Rs = dense.output_set(
        dense.reach({.initial_set = R0, .t_final = 1.0}, alg::Standard{.time_step = 0.05}));
    double worst = 0, same = 0;
    for (size_t k = 0; k < Rk.time_point.size(); ++k) {
        const Interval Ik = Rk.time_point[k].interval(), Is = Rs.time_point[k].interval();
        worst = std::max({worst, (Ik.inf() - Is.inf()).abs().max().item(),
                          (Ik.sup() - Is.sup()).abs().max().item()});
        same = std::max(
            same,
            (Rk.time_point[k].generators() - Rd.time_point[k].generators()).abs().max().item());
    }
    check(worst < 1e-9, "krylov: the time points of standard, off by " + std::to_string(worst));
    check(same == 0, "krylov: a sparse and a dense A give the same sets");
}

/// Trajectories of x' = A x + B u (constant u, the input set a point) from points of R0 stay in
/// the output time points and, at times inside a step, under the support of its enclosure.
void encloses_trajectories() {
    const SparseMatrix S = heat_grid(2);
    const int64_t n = S.rows();
    const Eigen::MatrixXd A = host_matrix(S.dense().data(), n, n);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n, 1), C = Eigen::MatrixXd::Identity(n, n);
    B(3, 0) = 1.0;
    cora::Rng rng(11);
    const Zonotope R0 = Zonotope::generate_random(rng, n, std::nullopt, 2);
    const Zonotope U(Tensor({{0.7}}), Tensor::zeros({1, 0}));
    const double dt = 0.1;
    const LinearSys sys(S, tensor_of(B), tensor_of(C));
    const ReachSet R = sys.reach({.initial_set = R0, .input_set = U, .t_final = 1.0},
                                 alg::Krylov{.time_step = dt});
    const Eigen::MatrixXd c0 = host_matrix(R0.center().data(), n, 1),
                          g0 = host_matrix(R0.generators().data(), n, 2);
    // the flow of the integrator system [A B u; 0 0] carries the constant input
    Eigen::MatrixXd Aint = Eigen::MatrixXd::Zero(n + 1, n + 1);
    Aint.topLeftCorner(n, n) = A;
    Aint.topRightCorner(n, 1) = B * 0.7;
    double worst = 1e9;
    for (int i = 0; i < 40; ++i) {
        Eigen::VectorXd beta(2), d(n), x0(n + 1);
        rng.uniform(beta.data(), 2, -1.0, 1.0);
        rng.normal(d.data(), n, 1.0);
        x0 << c0 + g0 * beta, 1.0;
        for (size_t k = 0; k < R.time_interval.size(); ++k) {
            const Eigen::VectorXd xk = ((Aint * (double(k + 1) * dt)).exp() * x0).head(n);
            const Eigen::VectorXd xm = ((Aint * ((double(k) + 0.5) * dt)).exp() * x0).head(n);
            worst = std::min({worst, support(R.time_point[k + 1], d) - d.dot(xk),
                              support(R.time_interval[k], d) - d.dot(xm)});
        }
    }
    check(worst >= -1e-9, "krylov: a trajectory left the sets by " + std::to_string(-worst));
}

/// Wrong options and a batch of initial sets are refused; with the checks off the result is the
/// same.
void checks_and_their_switch() {
    const LinearSys sys(heat_grid(2), Tensor::zeros({8, 1}));
    const Zonotope R0(Tensor::ones({8, 1}), Tensor::zeros({8, 0}));
    const ReachProblem problem{.initial_set = R0, .t_final = 0.2};
    // the options: a time step, a positive step of the order, a nonnegative error
    check(test::raises([&] { sys.reach(problem, alg::Krylov{}); }, Error::wrong_value),
          "krylov: no time step");
    check(test::raises([&] { sys.reach(problem, alg::Krylov{.time_step = 0.1, .krylov_step = 0}); },
                       Error::wrong_value),
          "krylov: krylov_step 0");
    check(
        test::raises([&] { sys.reach(problem, alg::Krylov{.time_step = 0.1, .krylov_error = -1}); },
                     Error::wrong_value),
        "krylov: a negative krylov_error");
    const Zonotope batch = Zonotope::stack({R0, R0});
    check(test::raises(
              [&] {
                  sys.reach({.initial_set = batch, .t_final = 0.2}, alg::Krylov{.time_step = 0.1});
              },
              Error::not_supported),
          "krylov: a batch of initial sets");
    // the result with the checks on, then off
    const double on =
        sys.reach(problem, alg::Krylov{.time_step = 0.1}).time_point.back().center().sum().item();
    ChecksScope off(false);
    // the same with the checks off
    check(
        sys.reach(problem, alg::Krylov{.time_step = 0.1}).time_point.back().center().sum().item() ==
            on,
        "krylov: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    flow_in_one_dimension();
    callback_receives_the_time_intervals();
    agrees_with_standard();
    encloses_trajectories();
    checks_and_their_switch();
    return test::finish("linear_sys reach krylov");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
