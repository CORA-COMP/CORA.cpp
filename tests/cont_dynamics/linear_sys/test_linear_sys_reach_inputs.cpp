// test_linear_sys_reach_inputs - linear_sys reach with an input set U, against MATLAB CORA, against
//
// simulations with constant and piecewise-constant inputs, and for the errors it reports
//
// CORA: contDynamics/@linearSys/reach.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_testing.h"
#include "cont_dynamics/linear_sys/matlab_reference_inputs.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;
using test::throws;
using namespace test::lin;
using matlab_reference::InputSystem;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

struct Setup {
    const InputSystem &s;
    Tensor A, B, C;
    Zonotope X0, U;
    explicit Setup(const InputSystem &s)
        : s(s), A(Tensor::from_data(s.A, {3, 3})), B(Tensor::from_data(s.B, {3, 2})),
          C(Tensor::from_data(s.C, {2, 3})),
          X0(Tensor::from_data(s.c0, {3, 1}), Tensor::from_data(s.G0, {3, 3})),
          U(Tensor::from_data(s.c_u, {2, 1}), Tensor::from_data(s.GU, {2, 3})) {}
    double t_final() const { return s.time_step * s.steps; }
};

/// The largest distance between the support values of the sets and the reference.
double distance(const std::vector<Zonotope> &sets, const std::vector<double> &dirs, int cols,
                const std::vector<std::vector<double>> &reference) {
    double worst = 0.0;
    for (std::size_t k = 0; k < reference.size(); ++k)
        for (int i = 0; i < int(dirs.size()) / cols; ++i) {
            const Eigen::VectorXd d = host_matrix(dirs, dirs.size() / cols, cols).row(i);
            worst = std::max(worst, std::abs(support(sets[k], d) - reference[k][i]));
        }
    return worst;
}

void matches_matlab_cora(const std::string &b) {
    for (const std::string &algorithm : kAlgorithms) {
        const InputSystem &s = algorithm == "standard" ? matlab_reference::standard_inputs()
                                                       : matlab_reference::wrapping_free_inputs();
        const Setup in(s);
        const std::string what = b + ": " + name(algorithm);
        const LinearSys sys(in.A, in.B, in.C);
        const ReachSet R =
            sys.reach({.initial_set = in.X0, .input_set = in.U, .t_final = in.t_final()},
                      test::lin::algorithm(algorithm, s.time_step, s.taylor_terms));
        // the numbers of steps
        check(R.time_interval.size() == 5 && R.time_point.size() == 6,
              what + ": the number of steps");

        check(distance(R.time_interval, s.dirs_x, 3, s.state_ti) < 1e-10,
              what + ": state enclosures");
        check(distance(R.time_point, s.dirs_x, 3, s.state_tp) < 1e-10,
              what + ": state time points");
        const ReachSet Y = sys.output_set(R);
        check(distance(Y.time_interval, s.dirs_y, 2, s.output_ti) < 1e-10,
              what + ": output enclosures");
        check(distance(Y.time_point, s.dirs_y, 2, s.output_tp) < 1e-10,
              what + ": output time points");
    }
}

/// Every trajectory of x' = A x + B u(t) with u(t) in U lies in the sets: the points at the time
/// points in the time-point sets, those in between in the enclosures. The input is piecewise
/// constant with extreme values of U, which is the demanding case.
void simulations_stay_in_the_reachable_set(const std::string &b) {
    cora::Rng rng(5);
    const InputSystem &s = matlab_reference::standard_inputs();
    const Setup in(s);
    const LinearSys sys(in.A, in.B);
    const int N = 40, refine = 4;
    Eigen::MatrixXd dirs(3, 30);
    rng.normal(dirs.data(), dirs.size(), 1.0);

    for (const std::string &algorithm : kAlgorithms)
        for (const int zonotope_order : {50, 2}) {
            const ReachSet R =
                sys.reach({.initial_set = in.X0, .input_set = in.U, .t_final = in.t_final()},
                          test::lin::algorithm(algorithm, s.time_step, 6, zonotope_order));
            Tensor x = in.X0.rand_point(N, rng, "extreme");
            double worst = -1e9;
            for (int k = 0; k < s.steps; ++k) {
                const Tensor u = in.U.rand_point(N, rng, "extreme");
                for (int r = 1; r <= refine; ++r) {
                    // a point inside the step: the same input from the step's start
                    const double t = s.time_step * r / refine;
                    const Tensor xr = sys.simulate({.initial_state = x, .input = u, .t_final = t},
                                                   alg::Exact{.time_step = t})
                                          .back();
                    const Eigen::MatrixXd P = host_of(xr);
                    for (int i = 0; i < dirs.cols(); ++i)
                        for (int j = 0; j < N; ++j)
                            worst = std::max(worst, dirs.col(i).dot(P.col(j)) -
                                                        support(R.time_interval[k], dirs.col(i)));
                }
                x = sys.simulate({.initial_state = x, .input = u, .t_final = s.time_step},
                                 alg::Exact{.time_step = s.time_step})
                        .back();
                const Eigen::MatrixXd P = host_of(x);
                for (int i = 0; i < dirs.cols(); ++i)
                    for (int j = 0; j < N; ++j)
                        worst = std::max(worst, dirs.col(i).dot(P.col(j)) -
                                                    support(R.time_point[k + 1], dirs.col(i)));
            }
            // a trajectory stays inside the sets
            check(worst <= 1e-9, b + ": " + name(algorithm) + " order " +
                                     std::to_string(zonotope_order) +
                                     ": a trajectory leaves the sets by " + std::to_string(worst));
        }
}

/// With U = {0} the input adds nothing: wrapping-free gives the sets of the system without input.
void a_zero_input_changes_nothing(const std::string &b) {
    const Setup in(matlab_reference::standard_inputs());
    const Zonotope U0(Tensor::zeros({2, 1}), Tensor::zeros({2, 1}));
    const ReachSet with = LinearSys(in.A, in.B)
                              .reach({.initial_set = in.X0, .input_set = U0, .t_final = 1.0},
                                     alg::WrappingFree{.time_step = 0.2, .taylor_terms = 6});
    const ReachSet without =
        LinearSys(in.A).reach({.initial_set = in.X0, .t_final = 1.0},
                              alg::WrappingFree{.time_step = 0.2, .taylor_terms = 6});
    double worst = 0.0;
    for (std::size_t k = 0; k < with.time_interval.size(); ++k)
        for (const Eigen::VectorXd &d : {Eigen::VectorXd(Eigen::Vector3d(1, 0, 0)),
                                         Eigen::VectorXd(Eigen::Vector3d(1, -1, 2))})
            worst = std::max(worst, std::abs(support(with.time_interval[k], d) -
                                             support(without.time_interval[k], d)));
    check(worst < 1e-12, b + ": U = 0 changed the sets by " + std::to_string(worst));
}

void reports_errors(const std::string &b) {
    const Setup in(matlab_reference::standard_inputs());
    const Zonotope wrong(Tensor::zeros({3, 1}), Tensor::eye(3));
    // an input set without B is refused
    check(throws([&] {
              LinearSys(in.A).reach({.initial_set = in.X0, .input_set = in.U, .t_final = 1.0},
                                    alg::Standard{.time_step = 0.2, .taylor_terms = 6});
          }),
          b + ": an input set without B is refused");
    check(throws([&] {
              LinearSys(in.A, in.B)
                  .reach({.initial_set = in.X0, .input_set = wrong, .t_final = 1.0},
                         alg::Standard{.time_step = 0.2, .taylor_terms = 6});
          }),
          b + ": an input set of the wrong dimension is refused");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        matches_matlab_cora(b);
        simulations_stay_in_the_reachable_set(b);
        a_zero_input_changes_nothing(b);
        reports_errors(b);
    });
    return test::finish("linear_sys reach (inputs)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
