// test_linear_sys_generate_random - generate_random
//
// The dimensions, the stability of the eigenvalues and the repeatability from a seed.
//
// CORA: contDynamics/@linearSys/generateRandom.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/linear_sys_steps_testing.h"
#include "cora/global/rng.h"

using namespace cora;
using namespace test::steps;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The dimensions asked for are those of the matrices; the defaults stay in CORA's ranges.
void the_dimensions_follow_the_arguments(const std::string &b) {
    cora::Rng rng(3);
    const LinearSys sys = LinearSys::generate_random(rng, 5, 2, 3);
    check(sys.dim() == 5 && sys.B()->shape() == std::vector<int64_t>({5, 2}) &&
              sys.C()->shape() == std::vector<int64_t>({3, 5}),
          b + ": given dimensions");
    for (int i = 0; i < 20; ++i) {
        const LinearSys random = LinearSys::generate_random(rng);
        check(random.dim() >= 4 && random.dim() <= 10 && random.B()->shape()[1] >= 1 &&
                  random.B()->shape()[1] <= 3 && random.C()->shape()[0] >= 1 &&
                  random.C()->shape()[0] <= 2,
              b + ": default dimensions");
    }
}

/// The eigenvalues are in the box: the real parts add up (the trace) inside [n low, n high], and
/// the same seed gives the same system.
void the_eigenvalues_are_in_the_box(const std::string &b) {
    cora::Rng rng(4), same(4);
    const Interval real(test::column({-3.0}), test::column({-1.0})),
        imaginary(test::column({-2.0}), test::column({2.0}));
    for (int i = 0; i < 10; ++i) {
        const LinearSys sys = LinearSys::generate_random(rng, 6, 1, 1, real, imaginary);
        const double trace = sys.A().trace().item();
        check(trace <= -6.0 + 1e-8 && trace >= -18.0 - 1e-8,
              b + ": the real parts add up inside the box");
        check(sys == LinearSys::generate_random(same, 6, 1, 1, real, imaginary),
              b + ": repeatable");
    }
}

/// A dimension of zero and an interval of two dimensions are refused.
void checks_and_their_switch(const std::string &b) {
    cora::Rng rng(5);
    check(test::raises([&] { LinearSys::generate_random(rng, 0); }, Error::wrong_value),
          b + ": n = 0");
    check(test::raises(
              [&] {
                  LinearSys::generate_random(
                      rng, 3, 1, 1,
                      Interval(test::column({-2.0, -1.0}), test::column({-1.0, 0.0})));
              },
              Error::wrong_value),
          b + ": a box of two dimensions");
    ChecksScope off(false);
    check(LinearSys::generate_random(rng, 3).dim() == 3, b + ": with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        the_dimensions_follow_the_arguments(b);
        the_eigenvalues_are_in_the_box(b);
        checks_and_their_switch(b);
    });
    return test::finish("linear_sys generate_random");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
