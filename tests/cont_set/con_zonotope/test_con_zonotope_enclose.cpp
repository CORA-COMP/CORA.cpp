// test_con_zonotope_enclose - ConZonotope::enclose: a set that encloses a set and its image
//
// CORA: contSet/@conZonotope/enclose.m
// PROPERTY: contSet/@conZonotope/enclose.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: [-1, 1] and 2 [-1, 1] + 4 = [2, 6]; the generators and the constraints of the
/// result.
void enclose_known(const std::string &b) {
    const ConZonotope A(column({0.0}), Tensor({{1.0}}));
    const ConZonotope image = A.enclose(Tensor({{2.0}}), column({4.0}));
    check(close(image.c(), {2}, 0), b + ": the center is the mean");
    check(close(image.generators(), {1.5, -2, -0.5}, 0), b + ": (G+G2)/2, (c-c2)/2, (G-G2)/2");
    check(close(image.interval().inf(), {-2}, 1e-9) && close(image.interval().sup(), {6}, 1e-9),
          b + ": the box [-2, 6] holds both");
    // the second set with fewer generators: the longer one's extra generators are kept
    const ConZonotope two(column({0.0}), Tensor({{1.0, 0.5}}));
    const ConZonotope shorter(column({4.0}), Tensor({{1.0}}));
    check(close(two.enclose(shorter).generators(), {1, -2, 0, 0.5}, 0),
          b + ": extra generators last");
    // the same constraints on the common factors: they stay, the new factors are free
    const ConZonotope C(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0, 1.0}}),
                        column({1.0}));
    const ConZonotope R = C.enclose(Tensor({{1.0, 0.0}, {0.0, 2.0}}), column({1.0, 1.0}));
    check(R.A().shape() == Shape{1, 5} && close(R.A(), {1, 1, 0, 0, 0}, 0),
          b + ": A padded with zeros");
    check(close(R.b(), {1}, 0), b + ": b is the first set's");
    // with a set to add: M * C + S
    const ConZonotope S(column({1.0, 1.0}), Tensor({{1.0}, {0.0}}));
    check(C.enclose(Tensor({{1.0, 0.0}, {0.0, 1.0}}), S).dim() == 2, b + ": M C + S");
}

/// Stated property: the result holds the points of both sets.
void enclose_property(const std::string &b) {
    cora::Rng rng(28);
    for (int k = 0; k < 4; ++k) {
        const ConZonotope A = test::random_con_zonotope(rng, 2, 4, 1);
        const Tensor M({{1.0, 0.5}, {-0.3, 2.0}}), t = column({1.0, -2.0});
        const ConZonotope R = A.enclose(M, t);
        const Tensor points = test::sample_points(A, rng, 30);
        check(R.contains(points, "exact", 1e-7), b + ": the first set");
        check(R.contains(M.matmul(points) + t, "exact", 1e-7), b + ": the image");
    }
}

/// A set that is no image, other dimensions and a matrix of the wrong size are refused; the same
/// valid call with the checks off.
void checks_and_their_switch() {
    const ConZonotope C(column({0.0}), Tensor({{1.0, 1.0}}), Tensor({{1.0, 1.0}}), column({1.0}));
    const ConZonotope other(column({0.0}), Tensor({{1.0, 1.0}}), Tensor({{1.0, -1.0}}),
                            column({1.0}));
    check(test::raises([&] { C.enclose(other); }, Error::special_error),
          "enclose: the constraints differ");
    check(test::raises([&] { C.enclose(ConZonotope(column({0.0}), Tensor({{1.0, 1.0}}))); },
                       Error::special_error),
          "enclose: a set without the constraints");
    check(test::raises([&] { C.enclose(ConZonotope(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}))); },
                       Error::dimension_mismatch),
          "enclose: different dimensions");
    check(test::raises([&] { C.enclose(Tensor({{1.0, 2.0}}), column({1.0})); },
                       Error::dimension_mismatch),
          "enclose: M must have n columns");
    // the longer set's constraints use a factor that the other lacks: CORA drops them
    const ConZonotope longer(column({0.0}), Tensor({{1.0, 1.0, 1.0}}), Tensor({{1.0, 1.0, 1.0}}),
                             column({1.0}));
    const ConZonotope shorter(column({0.0}), Tensor({{1.0, 1.0}}), Tensor({{1.0, 1.0}}),
                              column({1.0}));
    check(test::raises([&] { longer.enclose(shorter); }, Error::special_error),
          "enclose: constraints on factors only one set has");
    const ConZonotope A(column({0.0}), Tensor({{1.0}}));
    const Tensor on = A.enclose(Tensor({{2.0}}), column({4.0})).generators();
    ChecksScope off(false);
    check(close(A.enclose(Tensor({{2.0}}), column({4.0})).generators(), on, 0),
          "enclose: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        enclose_known(b);
        enclose_property(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope enclose");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
