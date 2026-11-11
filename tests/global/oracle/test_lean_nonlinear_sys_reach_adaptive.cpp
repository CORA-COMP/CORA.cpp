// test_lean_nonlinear_sys_reach_adaptive - reach with alg lin-adaptive against the CORALean oracle
//
// The run of van der Pol tunes the same steps while the tensor order is 2: the number of steps and
// the step sizes agree with the oracle's to 1e-6 (relative), and the sets lie in the oracle's,
// tighter by at most 2e-2. The sets are not bit for bit because the linear step is CORA's oneStep
// here and CORALean's interval-matrix exponential bounds its remainders more coarsely, and the
// time is accumulated in doubles here and exactly there. The run is short because the oracle is
// slow (the Robertson reaction is covered by the passes) and stops before the tensor order 3,
// whose quadratic map is CORA's here and the coarser polynomial-zonotope one there (the step
// sizes drift apart after it).
//
// CORA: contDynamics/@nonlinearSys/reach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"
#include "cora/global/oracle/nonlinear_sys.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kAgreement = 1e-6, kSets = 2e-2;

NonlinearSys van_der_pol() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[1], (1 - x[0] * x[0]) * x[1] - x[0] + u[0]};
        },
        2, 1);
}

/// The box of `mine` lies in the oracle's up to 1e-9 (relative) and is tighter by at most kSets:
/// the oracle's linear step bounds the remainders more coarsely.
void aux_inside(const std::string &what, const Zonotope &mine, const lean::Zonotope &theirs) {
    const Interval box = mine.interval();
    const auto [inf, sup] = theirs.interval().gather();
    const std::vector<double> a = box.inf().data(), b = inf.data(), c = box.sup().data(),
                              d = sup.data();
    bool ok = a.size() == b.size();
    for (std::size_t i = 0; ok && i < a.size(); ++i) {
        const double slack = 1e-9 * (1 + std::abs(b[i])), far = kSets * (1 + std::abs(b[i]));
        ok = a[i] >= b[i] - slack && a[i] <= b[i] + far && c[i] <= d[i] + slack &&
             c[i] >= d[i] - far;
    }
    test::check(ok, what + ": inside the oracle's box and close to it");
}

/// The run of CORA.cpp against the oracle's: steps and sets.
void aux_compare(const std::string &name, const NonlinearSys &sys, const Zonotope &R0,
                 const Zonotope &U, double t_final) {
    const alg::LinearizationAdaptive alg;
    const ReachProblem problem{R0, U, t_final};
    const ReachSet mine = sys.reach(problem, alg);
    const lean::AdaptiveRun theirs =
        lean::NonlinearSys(sys).reach_adaptive(lean::Zonotope(R0), lean::Zonotope(U), t_final, alg);
    test::oracle::op_name() = name;
    // the same steps, then their sizes and sets
    test::check(mine.time_interval.size() == theirs.steps.size(), name + ": the same steps");
    if (mine.time_interval.size() != theirs.steps.size()) return;
    bool same = true;
    for (std::size_t k = 0; k < theirs.steps.size(); ++k) {
        const double step = mine.time[k + 1] - mine.time[k];
        same = same && std::abs(step - theirs.steps[k]) <= kAgreement * theirs.steps[k];
    }
    test::check(same, name + ": the step sizes of the oracle");
    test::check(std::abs(mine.time.back() - theirs.t) <= kAgreement, name + ": the end time");
    for (std::size_t k = 0; k < theirs.steps.size(); ++k)
        aux_inside(name, mine.time_interval[k], theirs.time_interval[k]);
    aux_inside(name, mine.time_point.back(), theirs.time_point);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    if (!test::oracle::serves("nonlinearSys.reachAdaptive")) return test::finish("lean reach");
    const lean::DTypeScope dtype("nearest");
    aux_compare("van der Pol", van_der_pol(),
                Zonotope(Tensor({1.4, 2.3}), Tensor({{0.05, 0.0}, {0.0, 0.05}})),
                Zonotope(Tensor({0.0}), Tensor({{0.1}})), 0.015);
    return test::finish("lean nonlinearSys reach adaptive");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
