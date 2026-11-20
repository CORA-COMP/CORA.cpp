// test_con_zonotope_center - ConZonotope::center: a point of the set near its middle
//
// CORA: contSet/@conZonotope/center.m
// PROPERTY: contSet/@conZonotope/center.m
// CORA authors: Niklas Kochdumper, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: without constraints c; a segment; a square pinned by a trivial factor.
void center_known(const std::string &b) {
    const ConZonotope plain(vec({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 3.0}}));
    check(close(plain.center(), {1, 2}, 0), b + ": without constraints the offset");
    // x = xi on the line xi1 + xi2 = 1: the segment from (0, 1) to (1, 0), centered at (1/2, 1/2)
    const ConZonotope segment(vec({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), vec({1.0}));
    check(close(segment.center(), {0.5, 0.5}, 1e-9), b + ": the midpoint of a segment");
    // the third factor is zero: the square [-1, 1]^2, centered at the origin
    const ConZonotope square(vec({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}}),
                             Tensor({{0.0, 0.0, 1.0}}), vec({0.0}));
    check(close(square.center(), {0, 0}, 1e-9), b + ": the center of a square with a fixed factor");
    // one factor determined by the other: x = xi1 + 2, xi2 = 0.5 xi1 + ... full-dimensional in 1D
    const ConZonotope shifted(vec({2.0}), Tensor({{1.0}, {1.0}}), Tensor({{1.0, -1.0}}),
                              vec({0.0}));
    check(close(shifted.center(), {2}, 1e-9), b + ": xi1 = xi2: the points 2 + 2 xi, center 2");
    check(ConZonotope::empty(2).center().shape() == Shape{2} &&
              std::isnan(ConZonotope::empty(2).center().data()[0]),
          b + ": empty: NaN, as the empty box");
    const ConZonotope none(vec({0.0}), Tensor({{1.0}}), Tensor({{1.0}}), vec({3.0}));
    check(none.center().shape() == Shape{1} && std::isnan(none.center().item()),
          b + ": infeasible constraints: NaN");
}

/// Stated property: the center lies in the set, and the set is not larger than the box around it.
void center_property(const std::string &b) {
    cora::Rng rng(21);
    for (int k = 0; k < 8; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 3, 5, 2);
        const Tensor center = cz.center();
        check(center.shape() == Shape{3}, b + ": a point");
        check(cz.contains(center, "exact", 1e-7), b + ": the center is in the set");
        check(test::inside_box(center.expand_dims(-1), cz.interval()),
              b + ": the center is in the box of the set");
    }
}

/// A batch with constraints is refused; with the checks off a valid call gives the same center.
void checks_and_their_switch() {
    const ConZonotope batch(Tensor::zeros({2, 1}), Tensor::ones({2, 1, 2}).matrix_transpose(),
                            Tensor::ones({2, 1, 2}), Tensor::zeros({2, 1}));
    check(test::raises([&] { batch.center(); }, Error::not_supported),
          "center: a batch with constraints");
    check(
        test::close(
            ConZonotope(Tensor::zeros({2, 1}), Tensor::ones({2, 1, 2}).matrix_transpose()).center(),
            Tensor::zeros({2, 1}), 0),
        "center: a batch without constraints is its offset");
    const ConZonotope segment(vec({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), vec({1.0}));
    const Tensor on = segment.center();
    ChecksScope off(false);
    check(test::close(segment.center(), on, 0), "center: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        center_known(b);
        center_property(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope center");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
