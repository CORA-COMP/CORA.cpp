// test_lean_nonlinear_sys_lin_reach - one step of alg lin against the CORALean oracle
//
// The step of NonlinearSys::init_reach with linear_step = closed_form (CORA's oneStep in CORALean's
// closed form) against nonlinearSys.linReach with linearStep = closedForm, on van der Pol with an
// input and on the Robertson reaction. The linear step is the oracle's bit for bit
// (test_lean_linear_sys_closed_form_step), but CORALean's linReach is not CORA's loop: it checks
// the region the step stays in and bounds the linearization error over that region at the
// linearization point of the center, in one pass, where CORA guesses the error, solves it over the
// step and enlarges it until it holds. So the sets are compared as boxes, CORA.cpp's inside the
// oracle's (up to 1e-9, relative) and not much tighter (the error over the region is the coarser).
//
// CORA: contDynamics/@contDynamics/linReach.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/nonlinear_sys.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kSlack = 1e-9, kGap = 0.5;

NonlinearSys van_der_pol() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[1], (1 - x[0] * x[0]) * x[1] - x[0] + u[0]};
        },
        2, 1);
}

NonlinearSys robertson() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &) {
            return std::vector<Expr>{-0.4 * x[0] + 100 * x[1] * x[2],
                                     0.4 * x[0] - 100 * x[1] * x[2] - 1000 * pow(x[1], 2),
                                     1000 * pow(x[1], 2)};
        },
        3, 1);
}

/// The box of `mine` lies in the oracle's up to kSlack (relative), at most kGap (relative) inside.
void aux_inside(const std::string &what, const Zonotope &mine, const lean::Zonotope &theirs) {
    const Interval box = mine.interval();
    const auto [inf, sup] = theirs.interval().gather();
    const std::vector<double> a = box.inf().data(), b = inf.data(), c = box.sup().data(),
                              d = sup.data();
    bool ok = a.size() == b.size();
    for (std::size_t i = 0; ok && i < a.size(); ++i) {
        const double slack = kSlack * (1 + std::abs(b[i])), far = kGap * (1 + std::abs(b[i]));
        ok = a[i] >= b[i] - slack && a[i] <= b[i] + far && c[i] <= d[i] + slack &&
             c[i] >= d[i] - far;
    }
    test::check(ok, what + ": inside the oracle's box and close to it");
}

/// One step of both from R under U with the time step dt.
void aux_compare(const std::string &name, const NonlinearSys &sys, const Zonotope &R,
                 const Zonotope &U, double dt) {
    alg::Linearization alg;
    alg.time_step = dt;
    alg.linear_step = alg::LinearStep::closed_form;
    const NonlinearStep mine = sys.init_reach(R, U, alg);
    const std::optional<lean::LinReachResult> theirs =
        lean::NonlinearSys(sys).lin_reach(U, lean::Zonotope(R), alg);
    test::check(theirs.has_value(), name + ": the oracle's step");
    if (!theirs) return;
    aux_inside(name + " time interval", mine.time_interval, theirs->time_interval);
    aux_inside(name + " time point", mine.time_point, theirs->time_point);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    if (!test::oracle::serves("nonlinearSys.linReach")) return test::finish("lean linReach");
    test::oracle::op_name() = "nonlinearSys.linReach";
    const lean::DTypeScope dtype("nearest");
    aux_compare("van der Pol", van_der_pol(),
                Zonotope(Tensor({1.4, 2.3}), Tensor({{0.05, 0.01, 0.002}, {0.0, 0.05, 0.003}})),
                Zonotope(Tensor({0.0}), Tensor({{0.1}})), 0.02);
    aux_compare("Robertson", robertson(),
                Zonotope(Tensor({1.0, 0.0, 0.0}),
                         Tensor({{1e-3, 0.0, 0.0}, {0.0, 1e-4, 0.0}, {0.0, 0.0, 1e-4}})),
                Zonotope(Tensor({0.0}), Tensor::zeros({1, 0})), 0.001);
    return test::finish("lean nonlinearSys lin reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
