// test_zonotope_corners - zonotope corners: c + G s for every sign vector s
//
// CORA.cpp only: the corners that the containment tests of other sets ask for
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The four corners of a parallelogram, sign j of corner k the bit j of k; a point is its corner.
void corners_of_small_sets(const std::string &b) {
    const Zonotope Z(vec({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.5, 1.0}}));
    check(close(Z.corners(), std::vector<double>{-0.5, 1, 1.5, 1, 0.5, 3, 2.5, 3}, 0),
          b + ": the parallelogram");
    const Zonotope point(vec({3.0, -1.0}), Tensor::from_data({}, {0, 2}));
    check(close(point.corners(), std::vector<double>{3, -1}, 0), b + ": a point");
}

/// More than 30 generators are refused; the checks off give the same corners.
void checks_and_their_switch() {
    const Zonotope wide(vec({0.0}), Tensor::from_data(std::vector<double>(31, 1.0), {31, 1}));
    check(test::raises([&] { wide.corners(); }, Error::not_supported), "corners: 31 generators");
    const Zonotope Z(vec({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.5, 1.0}}));
    const Tensor on = Z.corners();
    ChecksScope off(false);
    check(close(Z.corners(), on, 0), "corners: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { corners_of_small_sets(b); });
    checks_and_their_switch();
    return test::finish("zonotope corners");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
