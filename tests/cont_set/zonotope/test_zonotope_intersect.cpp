// test_zonotope_intersect - zonotope intersect: boxes exactly, other zonotopes by averaging
//
// CORA: contSet/@zonotope/and_.m
// CORA: contSet/@zonotope/private/priv_andAveraging.m
// PROPERTY: contSet/@zonotope/and_.m

#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Two boxes intersect exactly, with either method; the result is a zonotope of a box.
void boxes_intersect_exactly(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const Zonotope Z2(column({1.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 0.5}}));
    for (const std::string method : {"conZonotope", "averaging"}) {
        const Zonotope R = Z.intersect(Z2, method);
        check(close(R.center(), std::vector<double>{0.5, 0}, 0), b + ": " + method + ": center");
        check(close(R.generators(), std::vector<double>{0.5, 0, 0, 0.5}, 0),
              b + ": " + method + ": the box [0,1]x[-0.5,0.5]");
    }
    const Zonotope touching = Z.intersect(Zonotope(column({2.0, 0.0}), Tensor({{1.0}, {0.0}})));
    check(close(touching.center(), std::vector<double>{1, 0}, 0) &&
              touching.generators().shape() == std::vector<int64_t>({2, 0}),
          b + ": boxes that touch meet in a point");
    const Interval I(column({0.0, -2.0}), column({0.5, 2.0}));
    check(close(Z.intersect(I).center(), std::vector<double>{0.25, 0}, 0), b + ": with a box");
    check(intersect(Z, I).is<Zonotope>(), b + ": through the registry");
}

/// Averaging two zonotopes that are not boxes: equal weights for equal zonotopes, and the
/// 1 / trace(G G') weights otherwise.
void averaging_weights(const std::string &b) {
    const Zonotope Z(column({1.0, 0.0}), Tensor({{1.0, 1.0}, {1.0, -1.0}}));
    const Zonotope same = Z.intersect(Z, "averaging");
    cora::Rng rng(20);
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = test::random_direction(rng, 2);
        check(close(test::support(same, d), test::support(Z, d), 1e-12), b + ": Z and Z average to Z");
    }
    const Zonotope Z2(column({0.0, 1.0}), Tensor({{2.0, 0.0}, {0.0, 2.0}}));
    const Zonotope R = Z.intersect(Z2, "averaging");
    // traces 4 and 8: weights 2/3 and 1/3
    check(close(R.center(), std::vector<double>{2.0 / 3.0, 1.0 / 3.0}, 1e-12), b + ": weighted center");
    check(close(R.generators(),
                std::vector<double>{2.0 / 3, 2.0 / 3, 2.0 / 3, 0, 2.0 / 3, -2.0 / 3, 0, 2.0 / 3}, 1e-12),
          b + ": weighted generators");
}

/// Stated property: a point in both zonotopes is in the average (found by sampling the first one).
void averaging_contains_the_intersection(const std::string &b) {
    cora::Rng rng(21);
    int shared = 0;
    for (int trial = 0; trial < 20; ++trial) {
        const Zonotope Z = Zonotope::generate_random(rng, 2, std::nullopt, 4);
        const Zonotope other = Zonotope::generate_random(rng, 2, std::nullopt, 4);
        const Zonotope Z2(Z.center() + column({0.3, -0.2}),
                          Z.generators() * 0.8 + other.generators() * 0.4);
        const Zonotope R = Z.intersect(Z2, "averaging");
        const Tensor points = Z.rand_point(60, rng);
        for (int k = 0; k < 60; ++k) {
            const Tensor x = points.slice(-1, k, k + 1);
            if (!Z2.contains(x)) continue;
            ++shared;
            check(R.contains(x, "exact", 1e-9), b + ": a common point stays in the average");
        }
    }
    check(shared > 5, b + ": the property was exercised");
}

/// Refusals: a pair of zonotopes that are not boxes needs ConZonotope, disjoint boxes the empty set.
void refusals_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 1.0}, {1.0, -1.0}}));
    const Zonotope box(column({5.0, 5.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { Z.intersect(Z); }, Error::not_supported), "intersect: conZonotope");
    check(test::raises([&] { Z.intersect(Z, "exact"); }, Error::wrong_value), "intersect: unknown method");
    check(test::raises([&] { box.intersect(Zonotope(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}))); },
                       Error::empty_set), "intersect: disjoint boxes");
    check(test::raises([&] { Z.intersect(Zonotope(column({0.0}), Tensor({{1.0}}))); },
                       Error::dimension_mismatch), "intersect: dimensions 2 and 1");
    const Tensor on = Z.intersect(Z, "averaging").generators();
    ChecksScope off(false);
    check(test::close(Z.intersect(Z, "averaging").generators(), on, 0),
          "intersect: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        boxes_intersect_exactly(b);
        averaging_weights(b);
        averaging_contains_the_intersection(b);
    });
    refusals_and_their_switch();
    return test::finish("zonotope intersect");
}
