// test_capsule_support_function - Capsule support_function: d'c + |d'g| + r |d|
//
// CORA: contSet/@capsule/supportFunc_.m
// PROPERTY: contSet/@capsule/supportFunc_.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed along the axes; the empty capsule has -Inf.
void support_known(const std::string &b) {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(close(C.support_function(column({1.0, 0.0})), {4.5}, 1e-14), b + ": d = (1, 0)");
    check(close(C.support_function(column({0.0, 2.0})), {13.0}, 1e-14), b + ": d = (0, 2)");
    check(close(C.support_function(column({-1.0, 0.0})), {2.5}, 1e-14), b + ": d = (-1, 0)");
    check(C.support_function(column({1.0, 0.0})).item() == 4.5, b + ": a number");
    check(Capsule::empty(2).support_function(column({1.0, 0.0})).item() ==
              -std::numeric_limits<double>::infinity(),
          b + ": empty");
}

/// Stated property: no sampled point goes beyond it, and the extreme point in the direction reaches
/// it.
void support_property(const std::string &b) {
    Rng rng(59);
    for (int k = 0; k < 6; ++k) {
        const Capsule C = test::random_capsule(rng, 3);
        const std::vector<double> d = test::random_direction(rng, 3);
        const double value = test::support(C, d);
        const Tensor P = test::sample_points(C, rng, 40);
        check(column(d).matrix_transpose().matmul(P).max().item() <= value + 1e-12,
              b + ": the points stay below");
        double norm = 0;
        for (const double x : d) norm += x * x;
        const double along = d[0] * C.generator().data()[0] + d[1] * C.generator().data()[1] +
                             d[2] * C.generator().data()[2];
        const Tensor tip = C.c() + C.generator() * (along >= 0 ? 1.0 : -1.0) +
                           column(d) * (C.r().item() / std::sqrt(norm));
        check(close(column(d).matrix_transpose().matmul(tip).item(), value, 1e-12),
              b + ": the tip attains it");
    }
}

/// A direction of the wrong size is refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(test::raises([&] { C.support_function(column({1.0})); }, Error::dimension_mismatch),
          "support_function: dimension");
    ChecksScope off(false);
    check(close(C.support_function(column({1.0, 0.0})), {4.5}, 1e-14),
          "support_function: same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        support_known(b);
        support_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule support_function");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
