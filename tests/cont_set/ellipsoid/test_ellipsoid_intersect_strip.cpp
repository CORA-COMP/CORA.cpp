// test_ellipsoid_intersect_strip - Ellipsoid::intersect_strip: an ellipsoid around E and a strip
//
// CORA: contSet/@ellipsoid/intersectStrip.m
// PROPERTY: contSet/@ellipsoid/intersectStrip.m

#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// The examples of CORALean's account of CORA's errors, in one dimension: E = [-1, 1].
void intersect_strip_known(const std::string &b) {
    const Ellipsoid E(Tensor({{1.0}}));
    const Tensor C({{1.0}});
    // Gollamudi: the strip |x - 2| <= 3 holds E; the weight is 1e-4, sigma_sq = (1 + 2 lambda)^2,
    // the shape sigma_sq and the center 2 lambda (CORA divides by sigma_sq and misses x = -1)
    const auto [G, sigma_sq] = E.intersect_strip(C, 3.0, column({2.0}));
    const double lambda = 1e-4;
    check(close(Tensor({{sigma_sq}}), {(1 + 2 * lambda) * (1 + 2 * lambda)}, 1e-12),
          b + ": Gollamudi: the level");
    check(close(G.Q(), {sigma_sq}, 1e-12) && close(G.q(), {2 * lambda}, 1e-12),
          b + ": Gollamudi: shape and center");
    check(test::all_inside(G, column({-1.0})) && test::all_inside(G, column({1.0})),
          b + ": Gollamudi: the points -1 and 1 of E are in the result");
    // a strip that holds E and a center on the strip's axis does not cut: unchanged
    const auto [same, level] = E.intersect_strip(C, 3.0, column({0.0}), 0.25);
    check(close(same.Q(), E.Q(), 0) && close(same.q(), E.q(), 0) && level == 0.25,
          b + ": Gollamudi: a strip that does not cut returns E and sigma_sq_prev");
    // Liu with V = bar_V = 1 and y = 1/2, sigma_sq_prev = 1: lambda = 1/2, sigma_sq = 15/16,
    // center 1/4, shape 15/16 (CORA's [-1/4, 3/4] misses x = 1 of the intersection [-1/2, 1])
    const Ellipsoid::StripMeasurement sys{Tensor({{1.0}}), Tensor({{1.0}})};
    const auto [L, sigma_liu] = E.intersect_strip(C, sys, column({0.5}), 1.0);
    check(close(Tensor({{sigma_liu}}), {15.0 / 16.0}, 1e-12), b + ": Liu: the level");
    check(close(L.Q(), {15.0 / 16.0}, 1e-12) && close(L.q(), {0.25}, 1e-12),
          b + ": Liu: shape and center");
    check(test::all_inside(L, column({1.0})) && test::all_inside(L, column({-0.5})),
          b + ": Liu: the ends of the intersection are in the result");
    // a strip far away: the combination has no point
    const auto [none, negative] = E.intersect_strip(C, 1.0, column({200.0}));
    check(none.Q().numel() == 0 && negative < 0.0, b + ": a strip that misses E: empty");
}

/// Stated property: the points of E (at the level of the previous update) in the strip lie in the
/// result, for random full-rank ellipsoids, strips of two rows and both methods.
void intersect_strip_property(const std::string &b) {
    cora::Rng rng(41);
    int updated = 0;
    for (int k = 0; k < 12; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3);
        const Tensor C = test::normal_matrix(rng, 2, 3);
        const double phi = 0.3;
        // the measurement is off the center of E by a vector of length about 0.6
        const Tensor y = C.matmul(E.q()) + test::normal_matrix(rng, 2, 1, 0.45);
        const Tensor P = test::ellipsoid_points(E, rng, 3000);
        const std::vector<double> misfit = (C.matmul(P) - y).square().sum(0, false).sqrt().data();
        std::vector<bool> in_strip;
        for (const double r : misfit) in_strip.push_back(r <= phi);
        const Tensor S = test::select_columns(P, in_strip);
        const auto [R, sigma_sq] = E.intersect_strip(C, phi, y);
        updated += R.Q().numel() != 0 && sigma_sq != 0.0;
        if (S.shape()[1] > 0)
            check(R.Q().numel() > 0 && test::all_inside(R, S, 1e-9),
                  b + ": Gollamudi: the points of E in the strip are in the result");
    }
    check(updated > 0, b + ": Gollamudi: some strip cut an ellipsoid");
    // Liu: the prior is the level sigma_sq_prev of E, the measurement (y - C x)' V^-1 (y - C x) <=
    // 1
    for (const double prior : {1.0, 0.5}) {
        for (int k = 0; k < 8; ++k) {
            const Ellipsoid E = test::random_ellipsoid(rng, 3);
            const Tensor C = test::normal_matrix(rng, 2, 3);
            const Tensor V = Tensor({{0.3, 0.1}, {0.1, 0.2}}),
                         bar_V = Tensor({{1.0, 0.0}, {0.0, 1.0}});
            const Tensor y = C.matmul(E.q()) + test::normal_matrix(rng, 2, 1, 0.5);
            const Tensor P =
                E.q() + (test::ellipsoid_points(E, rng, 3000) - E.q()) * std::sqrt(prior);
            const Tensor r = y.zeros_like({1, 3000});
            const Tensor d = C.matmul(P) - y;
            const std::vector<double> form = d.multiply(V.inv().matmul(d)).sum(0, false).data();
            std::vector<bool> in_set;
            for (const double f : form) in_set.push_back(f <= 1.0);
            const Tensor S = test::select_columns(P, in_set);
            const auto [R, sigma_sq] = E.intersect_strip(C, {V, bar_V}, y, prior);
            if (S.shape()[1] > 0)
                check(R.Q().numel() > 0 && test::all_inside(R, S, 1e-9),
                      b + ": Liu: the points in both sets are in the result");
            static_cast<void>(r);
        }
    }
}

/// What the strip, the measurement model and the ellipsoid must be; with the checks off the same
/// valid call gives the same result.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const Tensor C({{1.0, 0.0}});
    check(test::raises([&] { E.intersect_strip(Tensor({{1.0, 0.0, 0.0}}), 1.0, column({1.0})); },
                       Error::dimension_mismatch),
          "intersect_strip: C has n columns");
    check(test::raises([&] { E.intersect_strip(C, 1.0, column({1.0, 2.0})); },
                       Error::dimension_mismatch),
          "intersect_strip: y has a row per row of C");
    check(test::raises([&] { E.intersect_strip(C, std::nan(""), column({1.0})); },
                       Error::wrong_value),
          "intersect_strip: phi is finite");
    check(test::raises(
              [&] { E.intersect_strip(C, {Tensor({{1.0, 0.0}}), Tensor({{1.0}})}, column({1.0})); },
              Error::wrong_value),
          "intersect_strip: V and bar_V are m x m");
    check(test::raises(
              [] {
                  Ellipsoid(Tensor::zeros({2, 2}))
                      .intersect_strip(Tensor({{1.0, 0.0}}), 1.0, column({1.0}));
              },
              Error::degenerate_set),
          "intersect_strip: E must be full-dimensional");
    check(test::raises([&] { Ellipsoid::empty(2).intersect_strip(C, 1.0, column({1.0})); },
                       Error::empty_set),
          "intersect_strip: the empty ellipsoid");
    check(test::raises(
              [&] { Ellipsoid(Tensor::zeros({2, 2, 2})).intersect_strip(C, 1.0, column({1.0})); },
              Error::not_supported),
          "intersect_strip: a batch is not supported");
    const auto on = E.intersect_strip(C, 0.3, column({1.0}));
    ChecksScope off(false);
    const auto same = E.intersect_strip(C, 0.3, column({1.0}));
    check(close(same.first.Q(), on.first.Q(), 0) && same.second == on.second,
          "intersect_strip: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        intersect_strip_known(b);
        intersect_strip_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid intersect_strip");
}
