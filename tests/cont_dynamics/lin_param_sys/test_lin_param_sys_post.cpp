// test_lin_param_sys_post - LinParamSys post: the steps after the first, and its checks
//
// The second step of reach is post of init_reach; without time points post maps the set over the
// step only. A step of another system is refused; the checks switch off.
//
// CORA: contDynamics/@linParamSys/post.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_dynamics/lin_param_sys/lin_param_sys.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// x' = A x + u with A in [-1 2; -2 -1] +- 0.1 I, or with a third, decoupled state.
LinParamSys aux_sys(bool three = false) {
    if (three)
        return LinParamSys(
            MatZonotope(Tensor({{-1.0, 2.0, 0.0}, {-2.0, -1.0, 0.0}, {0.0, 0.0, -1.0}}),
                        Tensor::zeros({1, 3, 3})),
            Tensor::ones({3, 1}));
    const MatZonotope A(Tensor({{-1.0, 2.0}, {-2.0, -1.0}}),
                        Tensor({{0.1, 0.0}, {0.0, 0.1}}).expand_dims(0));
    return LinParamSys(A, Tensor({{0.0}, {1.0}}));
}

/// The second step of reach; no time-point set without them.
void second_step() {
    const LinParamSys sys = aux_sys();
    const Zonotope R0(column({1.0, 0.5}), Tensor({{0.05, 0.0}}));
    const Zonotope U(column({0.1}), Tensor({{0.2}}));
    alg::LinParamStandard a;
    a.time_step = 0.05;
    const ReachSet R = sys.reach({R0, U, 0.15}, a);
    // layout: #163 step 2
    const LinParamStep second =
        sys.post(sys.init_reach(R0, U - detail::cora_center(U), detail::cora_center(U), a), a);
    // layout: #163 step 2
    check(
        close(detail::cora_center(R.time_interval[1]), detail::cora_center(second.time_interval),
              0) &&
            // layout: #163 step 2
            close(detail::cora_generators(R.time_interval[1]),
                  detail::cora_generators(second.time_interval), 0) &&
            // layout: #163 step 2
            close(detail::cora_center(R.time_point[2]), detail::cora_center(*second.time_point), 0),
        "post: the second sets of reach");
    a.comp_time_point = false;
    const ReachSet ti = sys.reach({R0, U, 0.15}, a);
    // layout: #163 step 2
    const LinParamStep next =
        sys.post(sys.init_reach(R0, U - detail::cora_center(U), detail::cora_center(U), a), a);
    check(ti.time_point.empty() && ti.time_interval.size() == 3 && !next.time_point,
          "post: no time-point sets if they are not computed");
}

/// A step of another dimension is refused; with the checks off the step is the same.
void checks_and_their_switch() {
    const Zonotope R0(column({1.0, 0.5}), Tensor({{0.05, 0.0}}));
    const Zonotope U(column({0.0}), Tensor({{0.1}}));
    alg::LinParamStandard a;
    a.time_step = 0.05;
    const LinParamStep first = aux_sys().init_reach(R0, U, column({0.0}), a);
    check(test::raises([&] { aux_sys(true).post(first, a); }, Error::dimension_mismatch),
          "post: a step of the system's dimension");
    // layout: #163 step 2
    const Tensor on = detail::cora_center(aux_sys().post(first, a).time_interval);
    ChecksScope off(false);
    // layout: #163 step 2
    check(close(detail::cora_center(aux_sys().post(first, a).time_interval), on, 0),
          "post: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    second_step();
    checks_and_their_switch();
    return test::finish("lin_param_sys post");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
