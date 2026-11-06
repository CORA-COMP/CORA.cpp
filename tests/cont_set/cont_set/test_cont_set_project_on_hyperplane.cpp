// test_cont_set_project_on_hyperplane - ContSet::project_on_hyperplane: the orthogonal projection
//
// onto {x | c'x = d}
//
// CORA: contSet/@contSet/projectOnHyperplane.m
// PROPERTY: contSet/@contSet/projectOnHyperplane.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The square [-1, 1]^2 projects onto x + y = 2 as the segment from (0, 2) to (2, 0); the normal
/// is normalized, so (3, 3) and d = 6 describe the same line; the box hull contains the segment.
void the_projection_of_a_square(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor::from_data({1.0, 0.0, 0.0, 1.0}, {2, 2}));
    const ContSet &S = Z;
    for (const double scale : {1.0, 3.0}) {
        const Set R = S.project_on_hyperplane(column({scale, scale}),
                                              Tensor::from_data({2.0 * scale}, {1, 1}));
        check(R.is<Zonotope>(), b + ": a zonotope stays one");
        check(test::close(test::support(R, {1.0, 1.0}), 2.0, 1e-12) &&
                  test::close(test::support(R, {-1.0, -1.0}), -2.0, 1e-12) &&
                  test::close(test::support(R, {1.0, 0.0}), 2.0, 1e-12) &&
                  test::close(test::support(R, {-1.0, 0.0}), 0.0, 1e-12),
              b + ": the segment between (0, 2) and (2, 0), scale " + std::to_string(scale));
    }
    const Interval I(column({-1.0, -1.0}), column({1.0, 1.0}));
    const Set H = static_cast<const ContSet &>(I).project_on_hyperplane(
        column({1.0, 1.0}), Tensor::from_data({2.0}, {1, 1}));
    check(H.is<Interval>() && test::close(H.as<Interval>().inf(), {0.0, 0.0}, 1e-12) &&
              test::close(H.as<Interval>().sup(), {2.0, 2.0}, 1e-12),
          b + ": the hull of the image of a box");
}

/// Every projected sample lies on the hyperplane and the projection is idempotent, for random
/// zonotopes and hyperplanes in 4 dimensions.
void samples_land_on_the_hyperplane(const std::string &b) {
    Rng rng(11);
    for (int trial = 0; trial < 5; ++trial) {
        const Zonotope Z = Zonotope::generate_random(rng, 4, std::nullopt, 6);
        const Tensor c = column(test::random_direction(rng, 4)),
                     d = Tensor::from_data({test::random_direction(rng, 1)[0]}, {1, 1});
        const Set R = static_cast<const ContSet &>(Z).project_on_hyperplane(c, d);
        const double length = std::sqrt(c.multiply(c).sum().item());
        // the width of the image along the normal is zero, at offset d / |c|
        const Tensor n = c / length;
        check(test::close(R.get().support_function(n).item(), d.item() / length, 1e-9) &&
                  test::close(R.get().support_function(-n).item(), -d.item() / length, 1e-9),
              b + ": the image lies on the hyperplane");
        const Set again = R.get().project_on_hyperplane(c, d);
        for (int k = 0; k < 4; ++k) {
            const std::vector<double> dir = test::random_direction(rng, 4);
            check(test::close(test::support(again, dir), test::support(R, dir), 1e-9),
                  b + ": projecting twice changes nothing");
        }
        // the image of a sampled point is a point of the image
        const Tensor x = Z.rand_point(1, rng);
        const Tensor p = x - n * (n.matrix_transpose().matmul(x) - d / length).item();
        for (int k = 0; k < 4; ++k) {
            const std::vector<double> dir = test::random_direction(rng, 4);
            check(p.matrix_transpose().matmul(column(dir)).item() <= test::support(R, dir) + 1e-9,
                  b + ": the projected sample lies in the image");
        }
    }
}

/// A zero normal, a normal or an offset of the wrong shape are refused; the checks off skip them.
void errors_and_their_switch() {
    const Interval I(column({0.0, 0.0}), column({1.0, 1.0}));
    const ContSet &S = I;
    const Tensor one = Tensor::from_data({1.0}, {1, 1});
    check(
        test::raises([&] { S.project_on_hyperplane(column({0.0, 0.0}), one); }, Error::wrong_value),
        "zero normal");
    check(test::raises([&] { S.project_on_hyperplane(column({1.0}), one); },
                       Error::dimension_mismatch),
          "normal of one entry");
    check(test::raises([&] { S.project_on_hyperplane(column({1.0, 0.0}), column({1.0, 2.0})); },
                       Error::wrong_value),
          "two offsets");
    ChecksScope off(false);
    check(S.project_on_hyperplane(column({1.0, 0.0}), one).is<Interval>(), "the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_projection_of_a_square);
    test::for_each_backend(samples_land_on_the_hyperplane);
    errors_and_their_switch();
    return test::finish("cont_set project_on_hyperplane");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
