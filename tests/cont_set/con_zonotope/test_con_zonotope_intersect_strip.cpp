// test_con_zonotope_intersect_strip - ConZonotope::intersect_strip: the intersection with strips
//
// CORA: contSet/@conZonotope/intersectStrip.m
// PROPERTY: contSet/@conZonotope/intersectStrip.m

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: the strip |x1 - 0.5| <= 0.1 through [-1, 1]^2 with the gain (1, 0)': the new
/// center, generators and the strip row C G x - phi x_new = y - C c.
void intersect_strip_known(const std::string &b) {
    const ConZonotope cz(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const Tensor C({{1.0, 0.0}}), phi = column({0.1}), y = column({0.5}),
                                  lambda = column({1.0, 0.0});
    const ConZonotope R = cz.intersect_strip(C, phi, y, lambda);
    check(close(R.c(), {0.5, 0}, 1e-15), b + ": center c + L (y - C c)");
    check(close(R.generators(), {0, 0, 0.1, 0, 1, 0}, 1e-15),
          b + ": generators [(I - L C) G, L diag(phi)]");
    check(close(R.A(), {1, 0, -0.1}, 1e-15) && close(R.b(), {0.5}, 1e-15), b + ": the strip row");
    check(close(R.interval().inf(), {0.4, -1}, 1e-9) && close(R.interval().sup(), {0.6, 1}, 1e-9),
          b + ": the strip [0.4, 0.6] in the box");
    // a constrained set keeps its rows, the strip rows come last and the new factors are zero in
    // them
    const ConZonotope constrained(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                                  Tensor({{0.0, 1.0}}), column({0.5}));
    const ConZonotope S = constrained.intersect_strip(C, phi, y, lambda);
    check(close(S.A(), {0, 1, 0, 1, 0, -0.1}, 1e-15) && close(S.b(), {0.5, 0.5}, 1e-15),
          b + ": the rows of the set, then the strip");
    // the gains of the methods give the strip rows and the center of the zonotope's method
    const Zonotope Z(cz.c(), cz.generators());
    for (const char *method : {"normGen", "alamo-FRad"}) {
        const ConZonotope M = cz.intersect_strip(C, phi, y, method);
        const Zonotope expected = Z.intersect_strip(C, phi, y, method);
        check(close(M.c(), expected.center(), 0) && close(M.generators(), expected.generators(), 0),
              b + ": the gain of " + method);
        check(M.A().shape() == Shape{1, 3}, b + ": one strip row");
    }
    const Zonotope::StripObserver aux{Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0}}),
                                      Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0, 0.0}})};
    check(cz.intersect_strip(C, phi, y, aux).dim() == 2,
          b + ": the gain of wang-FRad from an observer");
    // bravo: the zonotope around the strip, with no constraints
    const ConZonotope V = constrained.intersect_strip(C, phi, y, "bravo");
    check(V.A().shape()[0] == 0, b + ": bravo drops the constraints, as CORA");
}

/// Stated property: the points of the set inside the strip are in the result, whatever the gain.
void intersect_strip_property(const std::string &b) {
    cora::Rng rng(31);
    for (int k = 0; k < 4; ++k) {
        const ConZonotope A = test::random_con_zonotope(rng, 3, 5, 2);
        const Tensor C({{1.0, 0.5, -0.5}, {0.0, 1.0, 1.0}}), phi = column({1.0, 1.5});
        const Tensor y = C.matmul(A.c()) + column({0.2, -0.1});
        const Tensor points = test::sample_points(A, rng, 120);
        for (const char *method : {"normGen"}) {
            const ConZonotope R = A.intersect_strip(C, phi, y, method);
            int inside = 0;
            for (int64_t j = 0; j < 120; ++j) {
                const Tensor x = points.slice(1, j, j + 1);
                const Tensor d = (C.matmul(x) - y).abs() - phi;
                if (d.max().item() > 0) continue;
                ++inside;
                check(R.contains(x, "exact", 1e-6), b + ": a point in the strip is in the result");
            }
            check(inside > 0, b + ": the strips meet the set");
        }
        const ConZonotope G =
            A.intersect_strip(C, phi, y, test::random_con_zonotope(rng, 3, 2, 0).generators());
        check(G.dim() == 3, b + ": a given gain");
    }
}

/// Wrong strips, gains and methods are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const ConZonotope cz(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const Tensor C({{1.0, 0.0}}), phi = column({0.1}), y = column({0.5});
    check(test::raises([&] { cz.intersect_strip(Tensor({{1.0}}), phi, y); },
                       Error::dimension_mismatch),
          "intersect_strip: C has n columns");
    check(test::raises([&] { cz.intersect_strip(C, column({0.1, 0.2}), y); },
                       Error::dimension_mismatch),
          "intersect_strip: phi has a row per strip");
    check(test::raises([&] { cz.intersect_strip(C, phi, y, Tensor({{1.0}})); },
                       Error::dimension_mismatch),
          "intersect_strip: the gain is (n, k)");
    check(test::raises([&] { cz.intersect_strip(C, phi, y, "nothing"); }, Error::wrong_value),
          "intersect_strip: an unknown method");
    check(test::raises([&] { cz.intersect_strip(C, phi, y, "svd"); }, Error::not_supported),
          "intersect_strip: svd needs an optimizer");
    check(test::raises([&] { cz.intersect_strip(C, phi, y, "wang-FRad"); }, Error::wrong_value),
          "intersect_strip: wang-FRad needs an observer");
    check(test::raises(
              [&] {
                  cz.intersect_strip(Tensor({{1.0, 0.0}, {0.0, 1.0}}), column({0.1, 0.1}),
                                     column({0.0, 0.0}), "alamo-FRad");
              },
              Error::special_error),
          "intersect_strip: alamo-FRad is for one strip");
    const Tensor on = cz.intersect_strip(C, phi, y).A();
    ChecksScope off(false);
    check(close(cz.intersect_strip(C, phi, y).A(), on, 0),
          "intersect_strip: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        intersect_strip_known(b);
        intersect_strip_property(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope intersect_strip");
}
