// test_cont_set_norm - ContSet::norm: the checked entry of every set's norm_
//
// CORA: contSet/@contSet/norm.m
// CORA: contSet/@contSet/norm_.m
// PROPERTY: contSet/@contSet/norm_.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"
#include "unregistered_set.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::column;
using N = Tensor::Norm;

namespace {

/// The box [-3, 1] x [2, 4] has largest magnitudes (3, 4); the zonotope with center (1, -1) and
/// generators (1, 1), (0, 2) has the box [0, 2] x [-4, 2], so (2, 4): the same bound for every
/// type and each of the two.
void norms_of_a_box_and_a_zonotope(const std::string &b) {
    const Interval I(column({-3.0, 2.0}), column({1.0, 4.0}));
    const Zonotope Z(column({1.0, -1.0}), Tensor::from_data({1.0, 0.0, 1.0, 2.0}, {2, 2}));
    const ContSet &S = I, &T = Z;
    check(test::close(S.norm(), {5.0}), b + ": a box, the 2-norm by default");
    check(test::close(S.norm(N::one), {7.0}) && test::close(S.norm(N::inf), {4.0}) &&
              test::close(S.norm(N::fro), {5.0}),
          b + ": a box, 1, inf and fro");
    check(test::close(S.norm(N::two, "exact"), {5.0}), b + ": a box ignores the mode");
    check(test::close(T.norm(), {std::sqrt(20.0)}) && test::close(T.norm(N::one), {6.0}) &&
              test::close(T.norm(N::inf), {4.0}) && test::close(T.norm(N::fro), {std::sqrt(20.0)}),
          b + ": a zonotope, the norm of its box");
    const Zonotope batch = Zonotope::stack({Z, Z + column({10.0, 0.0})});
    const Tensor n = static_cast<const ContSet &>(batch).norm(N::inf);
    check(n.shape() == Shape({2, 1, 1}) && test::close(n, {4.0, 12.0}), b + ": a batch");
}

/// Stated property: no random point of a zonotope has a larger norm than the bound.
void the_bound_holds_for_random_points(const std::string &b) {
    cora::Rng rng(21);
    for (int trial = 0; trial < 10; ++trial) {
        const Zonotope Z = Zonotope::generate_random(3, 4, rng);
        const std::vector<double> x = Z.rand_point(25, rng).data(); // (3, 25), row-major
        for (const N type : {N::one, N::two, N::inf}) {
            const double bound = static_cast<const ContSet &>(Z).norm(type).data()[0];
            for (int k = 0; k < 25; ++k) {
                const double a = std::abs(x[k]), c = std::abs(x[25 + k]), d = std::abs(x[50 + k]);
                const double value = type == N::one ? a + c + d
                                     : type == N::two ? std::sqrt(a * a + c * c + d * d)
                                                      : std::max({a, c, d});
                check(value <= bound * (1.0 + 1e-12), b + ": a point of the zonotope");
            }
        }
    }
}

/// A wrong mode, an unsupported one, a class without norm_, the checks off.
void errors_and_their_switch() {
    const Zonotope Z(column({1.0, -1.0}), Tensor::from_data({1.0, 0.0, 1.0, 2.0}, {2, 2}));
    const ContSet &T = Z;
    check(test::raises([&] { T.norm(N::two, "bogus"); }, Error::wrong_value), "a wrong mode");
    check(test::raises([&] { T.norm(N::two, "exact"); }, Error::not_supported), "exact");
    check(test::raises([&] { T.norm(N::two, "ub_convex"); }, Error::not_supported), "ub_convex");
    check(test::raises([] { static_cast<const ContSet &>(Fullspace(2)).norm(); }, Error::noops),
          "the whole space");
    check(test::raises([] { test::UnregisteredSet().norm(); }, Error::noops), "noops");
    ChecksScope off(false);
    check(test::close(T.norm(N::one), {6.0}), "the checks off");
}

} // namespace

int main() {
    test::for_each_backend(norms_of_a_box_and_a_zonotope);
    test::for_each_backend(the_bound_holds_for_random_points);
    errors_and_their_switch();
    return test::finish("cont_set norm");
}
