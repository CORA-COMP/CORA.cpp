// test_linear_sys_one_step - one step of reach from a set, an input set and a constant input
//
// The sets at the end and over the step are made of the solutions of its parts.
//
// CORA: contDynamics/@linearSys/oneStep.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_steps_testing.h"

using namespace cora;
using namespace test::steps;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The end of the step is e^{A dt} X plus the constant input plus the time-varying input solution.
void the_end_is_the_sum_of_its_parts(const std::string &b) {
    const Tensor A = diagonal(), u = test::column({1.0, -1.0});
    const Zonotope U(test::column({0.0, 0.0}), Tensor({{0.5}, {0.25}}));
    // the parts of the step are those of the single solutions
    const LinearSys sys(A);
    const OneStep s = sys.one_step(initial_set(), U, u, kStep, kTerms);
    const AffineSolution affine = sys.affine_solution(initial_set(), u, kStep, kTerms);
    const Zonotope PU = sys.particular_solution_time_varying(U, kStep, kTerms);
    check(test::close(s.time_point.center(), affine.time_point.center() + PU.center(), 1e-15),
          b + ": the center");
    check(test::close(s.time_varying.generators(), PU.generators(), 0), b + ": PU");
    check(test::close(s.constant.center(), affine.constant.center(), 1e-15), b + ": Pu");
    check(test::close(s.state_curvature.interval().sup(), affine.state_curvature.interval().sup(),
                      1e-15),
          b + ": the state curvature");
    for (const std::vector<double> &d : {std::vector<double>{1, 0}, {0, 1}, {1, -1}}) {
        check(std::abs(support(s.time_point, d) - support(s.homogeneous_time_point.plus(PU), d)) <
                  1e-14,
              b + ": Rtp is Htp + PU");
        check(support(s.time_interval, d) >= support(s.time_point, d) - 1e-14,
              b + ": Rti holds Rtp");
    }
}

/// Every trajectory under an input of U and the constant input stays in the sets of the step.
void trajectories_stay_inside(const std::string &b) {
    const Tensor A = diagonal();
    const Zonotope U(test::column({0.0, 0.0}), Tensor({{0.5}, {0.25}}));
    const Tensor u = test::column({0.5, 0.0});
    const OneStep s = LinearSys(A).one_step(initial_set(), U, u, kStep, kTerms);
    // the end of a trajectory under a held input w = u + factor * generator
    for (const double factor : {-1.0, 0.0, 1.0}) {
        const Tensor w = u + Tensor({{0.5}, {0.25}}) * factor;
        for (const std::vector<double> &x : {std::vector<double>{2, 2.5}, {0, 1.5}}) {
            const Tensor x_end = flow(A, kStep, test::column(x)) +
                                 A.inv().matmul((A * kStep).expm() - Tensor::eye(2)).matmul(w);
            check(s.time_point.contains(x_end), b + ": the end of a trajectory");
            check(s.time_interval.contains(x_end), b + ": inside the step's enclosure");
        }
    }
}

/// The checks of a step: the sets' dimensions and the constant input's shape.
void checks_and_their_switch(const std::string &b) {
    const LinearSys sys(diagonal());
    const Tensor u = test::column({0.0, 0.0});
    // the dimensions of U and u, the order and the checks switch
    check(test::raises([&] { sys.one_step(initial_set(), origin(3), u, kStep, kTerms); },
                       Error::dimension_mismatch),
          b + ": U of three dimensions");
    check(test::raises(
              [&] { sys.one_step(initial_set(), origin(2), test::column({0.0}), kStep, kTerms); },
              Error::dimension_mismatch),
          b + ": u of one row");
    check(test::raises([&] { sys.one_step(initial_set(), origin(2), u, kStep, 0); },
                       Error::wrong_value),
          b + ": no terms");
    const Tensor on =
        sys.one_step(initial_set(), origin(2), u, kStep, kTerms).time_interval.center();
    ChecksScope off(false);
    check(
        test::close(sys.one_step(initial_set(), origin(2), u, kStep, kTerms).time_interval.center(),
                    on, 0),
        b + ": the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_end_is_the_sum_of_its_parts(b);
        trajectories_stay_inside(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys one_step");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
