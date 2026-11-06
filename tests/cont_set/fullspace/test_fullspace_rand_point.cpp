// test_fullspace_rand_point - fullspace rand_point: standard normal points
//
// CORA: contSet/@fullspace/randPoint_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void the_points_are_standard_normal(const std::string &b) {
    Rng rng(3);
    const Tensor P = Fullspace(3).rand_point(2000, rng);
    check(P.shape() == std::vector<int64_t>({3, 2000}), b + ": shape (n, N)");
    const std::vector<double> x = P.data();
    double sum = 0.0, square = 0.0;
    for (const double v : x) sum += v, square += v * v;
    check(std::abs(sum / x.size()) < 0.05 && std::abs(square / x.size() - 1.0) < 0.1,
          b + ": mean 0 and variance 1");
    check(Fullspace(2).rand_point(0, rng).shape() == std::vector<int64_t>({2, 0}), b + ": N = 0");
}

/// A negative count and R^0 are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    Rng rng(3);
    check(test::raises([&] { Fullspace(2).rand_point(-1, rng); }, Error::wrong_value), "N = -1");
    check(test::raises([&] { Fullspace(0).rand_point(1, rng); }, Error::not_supported), "R^0");
    ChecksScope off(false);
    check(Fullspace(2).rand_point(3, rng).shape() == std::vector<int64_t>({2, 3}),
          "rand_point: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_points_are_standard_normal);
    checks_and_their_switch();
    return test::finish("fullspace rand_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
