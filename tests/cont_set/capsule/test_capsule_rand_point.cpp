// test_capsule_rand_point - Capsule rand_point: points of the capsule, or of its boundary
//
// CORA: contSet/@capsule/randPoint_.m
// PROPERTY: contSet/@capsule/randPoint_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: standard points lie in the capsule; extreme points lie on its boundary.
void rand_point_property(const std::string &b) {
    Rng rng(61);
    for (int k = 0; k < 4; ++k) {
        const Capsule C = test::random_capsule(rng, 3);
        check(C.contains(C.rand_point(30, rng)), b + ": standard points are inside");
        const Tensor P = C.rand_point(20, rng, "extreme");
        check(P.shape() == Shape({3, 20}) && C.contains(P), b + ": extreme points are inside");
        bool boundary = true;
        const std::vector<double> x = P.data();
        for (int j = 0; j < 20; ++j)
            boundary = boundary && close(test::axis_distance(C, {x[j], x[20 + j], x[40 + j]}),
                                         C.r().item(), 1e-5);
        check(boundary, b + ": extreme points are at the radius from the axis");
    }
    const Capsule batch(Tensor::zeros({2, 3, 1}), Tensor::ones({2, 3, 1}),
                        Tensor::from_data({0.5, 1.0}, {2, 1, 1}));
    check(batch.rand_point(5, rng).shape() == Shape({2, 3, 5}), b + ": a batch");
}

/// An unknown type, a negative number and the empty capsule are refused; checks off works.
void checks_and_their_switch() {
    Rng rng(3);
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(test::raises([&] { C.rand_point(3, rng, "gaussian"); }, Error::no_specific_alg),
          "rand_point: type");
    check(test::raises([&] { C.rand_point(-1, rng); }, Error::wrong_value), "rand_point: N");
    check(test::raises([&] { Capsule::empty(2).rand_point(1, rng); }, Error::empty_set),
          "rand_point: empty");
    ChecksScope off(false);
    check(C.rand_point(2, rng).shape() == Shape({2, 2}), "rand_point: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { rand_point_property(b); });
    checks_and_their_switch();
    return test::finish("capsule rand_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
