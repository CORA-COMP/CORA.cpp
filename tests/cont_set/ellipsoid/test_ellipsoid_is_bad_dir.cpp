// test_ellipsoid_is_bad_dir - Ellipsoid::is_bad_dir: the directions of the Minkowski difference
//
// CORA: contSet/@ellipsoid/isBadDir.m
// PROPERTY: contSet/@ellipsoid/isBadDir.m

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;

namespace {

/// Hand-computed: E1 = I and E2 = diag(1/4, 1): the largest generalized eigenvalue is 1, so a
/// direction is bad if sqrt(l' Q1 l) / sqrt(l' Q2 l) exceeds 1.
void is_bad_dir_known(const std::string &b) {
    const Ellipsoid E1(Tensor({{1.0, 0.0}, {0.0, 1.0}})), E2(Tensor({{0.25, 0.0}, {0.0, 1.0}}));
    const std::vector<bool> bad = E1.is_bad_dir(E2, Tensor({{1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}));
    // l = e1: 1 / (1/2) = 2 > 1; l = e2: 1 / 1 is not above 1 + tol; l = (1, 1): sqrt 2 / sqrt 1.25
    check(bad.size() == 3 && bad[0] && !bad[1] && bad[2], b + ": bad, good and a mixed direction");
    check(!E1.is_bad_dir(E1, Tensor({{1.0}, {2.0}}))[0], b + ": equal ellipsoids have none");
    check(E1.is_bad_dir(E2, Tensor::zeros({2, 0})).empty(), b + ": no directions, no answers");
}

/// Stated property for E2 = c E1: the ratio sqrt(l' Q1 l) / sqrt(l' Q2 l) is 1 / sqrt(c) in every
/// direction and the bound 1 / lambda is 1 / c, so all directions are bad iff c > 1.
void is_bad_dir_property(const std::string &b) {
    cora::Rng rng(31);
    for (int k = 0; k < 5; ++k) {
        const Ellipsoid E1 = test::random_ellipsoid(rng, 3);
        const Tensor L = test::normal_matrix(rng, 3, 8);
        for (const bool bad : E1.is_bad_dir(Ellipsoid(E1.Q() * 4.0), L))
            check(bad, b + ": a larger copy is bad in every direction");
        for (const bool bad : E1.is_bad_dir(Ellipsoid(E1.Q() * 0.25), L))
            check(!bad, b + ": a smaller copy is bad in none");
        for (const bool bad : E1.is_bad_dir(Ellipsoid(E1.Q()), L))
            check(!bad, b + ": the same ellipsoid is bad in none");
    }
}

/// Wrong directions, dimensions, empty and batched sets are refused; with the checks off the
/// answer is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { E.is_bad_dir(E, Tensor({{1.0}, {2.0}, {3.0}})); }, Error::wrong_value),
          "is_bad_dir: L has n rows");
    check(test::raises([&] { E.is_bad_dir(Ellipsoid(Tensor({{1.0}})), Tensor({{1.0}, {2.0}})); },
                       Error::dimension_mismatch),
          "is_bad_dir: the dimensions agree");
    check(test::raises([&] { E.is_bad_dir(Ellipsoid::empty(2), Tensor({{1.0}, {2.0}})); },
                       Error::empty_set),
          "is_bad_dir: an empty ellipsoid");
    const Tensor L({{1.0}, {2.0}});
    const std::vector<bool> on = E.is_bad_dir(E, L);
    ChecksScope off(false);
    check(E.is_bad_dir(E, L) == on, "is_bad_dir: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        is_bad_dir_known(b);
        is_bad_dir_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid is_bad_dir");
}
