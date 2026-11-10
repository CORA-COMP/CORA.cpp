// test_lin_param_sys_reach - LinParamSys reach, init_reach and post on a damped oscillator
//
// The property: for random matrices A of the set, random start points of R0 and random constant
// inputs u of U, the exact solution x(t) = e^{A t} x0 + int_0^t e^{A s} ds (B u + c) lies, at
// random times of every step, in the time-interval set of that step, and at the step times in
// the time-point sets. It is checked for one generator with constant parameters (CORA's
// expmOneParam), two generators (expmMixed), varying parameters (expmIndMixed), and an input set
// without the origin (the curvature of priv_inputTie).
//
// CORA: contDynamics/@contDynamics/reach.m
// CORA: contDynamics/@linParamSys/initReach.m
// CORA: contDynamics/@linParamSys/post.m
// PROPERTY: contDynamics/@contDynamics/reach.m
// PROPERTY: contDynamics/@linParamSys/initReach.m
// PROPERTY: contDynamics/@linParamSys/post.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/lin_param_sys/lin_param_sys.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>
#include <string>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kDt = 0.05, kTFinal = 1.0, kTol = 1e-9;

/// The oscillator [-1 2; -2 -1] with h generators of size 0.1 on the diagonal and the corner.
MatZonotope aux_set(int h) {
    Tensor G = Tensor({{0.1, 0.0}, {0.0, 0.1}}).expand_dims(0);
    if (h == 2) G = Tensor::concat({G, Tensor({{0.0, 0.1}, {0.0, 0.0}}).expand_dims(0)}, 0);
    return MatZonotope(Tensor({{-1.0, 2.0}, {-2.0, -1.0}}), G);
}

/// The exact state at t from x0 for the matrix A and the constant input B u + c (augmented
/// exponential).
Tensor aux_exact(const Tensor &A, const Tensor &v, const Tensor &x0, double t) {
    const Tensor top = Tensor::concat({A, v}, -1);
    const Tensor M = Tensor::concat({top, Tensor::zeros({1, 3})}, -2) * t;
    const Tensor x = M.expm().matmul(Tensor::concat({x0, Tensor::ones({1, 1})}, -2));
    return x.slice(-2, 0, 2);
}

/// Whether reach of sys from R0 under U holds the exact trajectories (the stated property).
void encloses(const std::string &name, const LinParamSys &sys, const Zonotope &U) {
    const Zonotope R0(column({1.0, 0.5}), Tensor({{0.05, 0.0}, {0.0, 0.05}}));
    alg::LinParamStandard a;
    a.time_step = kDt;
    a.taylor_terms = 6;
    a.intermediate_terms = 3;
    a.zonotope_order = 20;
    const ReachSet R = sys.reach({R0, U, kTFinal}, a);
    check(R.time_interval.size() == 20 && R.time_point.size() == 21, name + ": the steps");
    Rng rng(7);
    for (int k = 0; k < 30; ++k) {
        // a member A, a start point, an input and a time
        double b[6];
        rng.uniform(b, 6, -1.0, 1.0);
        const Tensor &G = sys.A().generators();
        Tensor A = sys.A().center();
        for (int64_t g = 0; g < G.size(0); ++g) A = A + G.slice(0, g, g + 1).squeeze(0) * b[g];
        const Tensor x0 = R0.center() + R0.generators().matmul(column({b[2], b[3]}));
        const double u = U.center().item() + U.generators().abs().sum().item() * b[4];
        const Tensor v = sys.B()->matmul(column({u})) + sys.c();
        const auto step = std::size_t(k % 20);
        const double t = (double(step) + 0.5 * (b[5] + 1)) * kDt;
        check(R.time_interval[step].contains(aux_exact(A, v, x0, t), "exact", kTol),
              name + ": an exact state in its time-interval set");
        check(R.time_point[step + 1].contains(aux_exact(A, v, x0, double(step + 1) * kDt), "exact",
                                              kTol),
              name + ": an exact state at a step time");
    }
}

/// Wrong problems and options are refused; with the checks off reach is the same.
void checks_and_their_switch() {
    const LinParamSys sys(aux_set(1), Tensor({{0.0}, {1.0}}));
    const Zonotope R0(column({1.0, 0.5}), Tensor({{0.05}, {0.0}}));
    alg::LinParamStandard a;
    a.time_step = kDt;
    // the problem: an initial set, a horizon of a step, inputs of B
    check(
        test::raises([&] { sys.reach({std::nullopt, std::nullopt, 1.0}, a); }, Error::wrong_value),
        "reach: the problem needs an initial set");
    check(test::raises([&] { sys.reach({R0, std::nullopt, 0.01}, a); }, Error::wrong_value),
          "reach: t_final below one step");
    check(test::raises([&] { sys.reach({R0, R0, 1.0}, a); }, Error::dimension_mismatch),
          "reach: the input set of the columns of B");
    alg::LinParamStandard bad = a;
    bad.intermediate_terms = 5;
    check(test::raises([&] { sys.reach({R0, std::nullopt, 1.0}, bad); }, Error::wrong_value),
          "reach: intermediate_terms above taylor_terms");
    // an input set without the origin at a step where the series of its curvature diverges
    alg::LinParamStandard stiff = a;
    stiff.time_step = 5.0;
    stiff.taylor_terms = 6;
    const Zonotope away(column({0.75}), Tensor({{0.25}}));
    check(test::raises([&] { sys.reach({R0, away, 5.0}, stiff); }, Error::reach_set_explosion),
          "reach: the curvature of a constant input needs norm(A) r < taylor_terms + 2");
    const Tensor on = sys.reach({R0, std::nullopt, 0.2}, a).time_interval.back().center();
    ChecksScope off(false);
    check(close(sys.reach({R0, std::nullopt, 0.2}, a).time_interval.back().center(), on, 0),
          "reach: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const Tensor B = Tensor({{0.0}, {1.0}}), c = column({0.5, 0.0});
    const Zonotope U(column({0.05}), Tensor({{0.25}})), away(column({0.75}), Tensor({{0.25}}));
    // one generator, two, varying parameters, and inputs that miss the origin
    encloses("one generator", LinParamSys(aux_set(1), B, c), U);
    encloses("two generators", LinParamSys(aux_set(2), B, c), U);
    encloses("varying", LinParamSys(aux_set(2), B, c, ParamType::varying), U);
    encloses("input away from 0", LinParamSys(aux_set(1), B, c), away);
    checks_and_their_switch();
    return test::finish("lin_param_sys reach");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
