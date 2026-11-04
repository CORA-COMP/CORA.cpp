// test_lean_polytope - the operations of Polytope that the CORALean oracle serves
//
// Bit for bit where CORALean builds the same set: Inf, empty, and and cartProd of halfspace
// polytopes (their rows stack), interval, box and center of vertices (the extremes, and the mean
// as a box that is a point at nearest), contains (a Boolean), the polytope of a box, and the
// vertices that convHull and enclosePoints keep (CORALean lists every point, CORA.cpp the extreme
// ones: compared after compact 'V'). Checked as enclosures, without an ORACLE tag: and of vertex
// polytopes (CORALean the box of the intersection of the boxes) and conZonotope (another
// construction).
//
// ORACLE: contSet/@polytope/Inf.m
// ORACLE: contSet/@polytope/empty.m
// ORACLE: contSet/@polytope/and_.m
// ORACLE: contSet/@polytope/cartProd_.m
// ORACLE: contSet/@polytope/box.m
// ORACLE: contSet/@polytope/center.m
// ORACLE: contSet/@polytope/interval.m
// ORACLE: contSet/@polytope/contains_.m
// ORACLE: contSet/@polytope/convHull_.m
// ORACLE: contSet/@polytope/enclosePoints.m
// ORACLE: contSet/@interval/polytope.m

#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/oracle/polytope.h"
#include "cora/global/tensor/tolerance.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;
namespace oracle = test::oracle;

namespace {

/// k points of n dimensions on a sphere of the given radius about a center: all extreme.
Tensor aux_sphere(oracle::Random &rnd, int64_t n, int64_t k, double radius = 1.0,
                  double shift = 0.0) {
    std::vector<double> x = rnd.matrix(n, k).data();
    for (int64_t c = 0; c < k; ++c) {
        double norm = 0.0;
        for (int64_t i = 0; i < n; ++i) norm += x[i * k + c] * x[i * k + c];
        for (int64_t i = 0; i < n; ++i)
            x[i * k + c] = shift + radius * x[i * k + c] / std::sqrt(norm);
    }
    return Tensor::from_data(x, {n, k});
}

/// The box |x| <= 2 and m random rows with offset 3: nonempty and bounded.
Polytope aux_halfspaces(oracle::Random &rnd, int64_t n, int64_t m) {
    // the negative identity with +0 off the diagonal, as the oracle prints it
    const Tensor eye = Tensor::eye(n), negative = eye * 0.0 - eye;
    return {Tensor::concat({eye, negative, rnd.matrix(m, n)}, 0),
            Tensor::concat({Tensor::ones({2 * n, 1}) * 2.0, Tensor::ones({m, 1}) * 3.0}, 0)};
}

/// The vertices of a result, reduced to the extreme ones as CORA.cpp keeps them.
Polytope aux_extreme(const Polytope &P) { return P.compact("V"); }

} // namespace

int main() {
    if (!oracle::available()) return 0;
    oracle::run("polytope.Inf",
                [](oracle::Random &) { oracle::check(Polytope::inf(3), lean::Polytope::inf(3)); });
    oracle::run("polytope.empty", [](oracle::Random &) {
        const Polytope mine = Polytope::empty(3);
        // CORALean's empty set is the halfspace 0 <= -1, as CORA.cpp's rows
        oracle::check(Polytope(mine.A(), mine.b()), lean::Polytope::empty(3));
    });
    oracle::run("polytope.and", [](oracle::Random &rnd) {
        const Polytope a = aux_halfspaces(rnd, 3, 2), b = aux_halfspaces(rnd, 3, 3);
        oracle::check(a.intersect(b), lean::Polytope(a).intersect(b));
    });
    oracle::run("polytope.and (vertices)", [](oracle::Random &rnd) {
        const Polytope a(aux_sphere(rnd, 3, 6)), b(aux_sphere(rnd, 3, 6, 1.0, 0.2));
        // CORALean: the box of the intersection of the boxes, which encloses the intersection
        const Polytope mine = a.intersect(b);
        const lean::Interval theirs = lean::Polytope(a).intersect(b).interval();
        if (!mine.interval().is_empty()) oracle::check_enclosed("and", mine.interval(), theirs);
    });
    oracle::run("polytope.cartProd", [](oracle::Random &rnd) {
        const Polytope a = aux_halfspaces(rnd, 3, 2), b = aux_halfspaces(rnd, 2, 1);
        oracle::check(a.cart_prod(b), lean::Polytope(a).cart_prod(b));
    });
    oracle::run("polytope.interval", [](oracle::Random &rnd) {
        const Polytope P(aux_sphere(rnd, 3, 7));
        oracle::check(P.interval(), lean::Polytope(P).interval());
    });
    oracle::run("polytope.box", [](oracle::Random &rnd) {
        const Polytope P(aux_sphere(rnd, 3, 7));
        const Interval box = P.box().interval();
        oracle::check(box, lean::Polytope(P).interval());
    });
    oracle::run("polytope.center", [](oracle::Random &rnd) {
        const Polytope P(aux_sphere(rnd, 3, 7));
        // the mean of the vertices is the oracle's box at nearest; a float mean on another backend
        // sums in another order, so sound mode allows a few ulp around the oracle's box
        const Tensor mean = P.center("avg");
        const lean::Interval theirs = lean::Polytope(P).center();
        if (oracle::mode() == oracle::Mode::exact)
            return oracle::check_exact("center", mean, theirs);
        const auto [lo, hi] = theirs.gather();
        const std::vector<double> m = mean.data(), a = lo.data(), b = hi.data();
        bool ok = true;
        for (std::size_t i = 0; i < m.size(); ++i)
            ok = ok && a[i] - 16 * machine_eps() * std::abs(a[i]) <= m[i] &&
                 m[i] <= b[i] + 16 * machine_eps() * std::abs(b[i]);
        test::check(ok, "center: the mean within a few ulp of the oracle's box");
    });
    oracle::run("polytope.contains", [](oracle::Random &rnd) {
        const Polytope P = aux_halfspaces(rnd, 3, 2);
        const Tensor c = rnd.vector(3) * 0.0;
        const Interval inside(c - 0.1, c + 0.1), outside(c - 5.0, c + 5.0);
        oracle::check(P.contains(inside), lean::Polytope(P).contains(inside));
        oracle::check(P.contains(outside), lean::Polytope(P).contains(outside));
        const Zonotope near(c, rnd.matrix(3, 4) * 0.05), far(c, rnd.matrix(3, 4) * 5.0);
        oracle::check(P.contains(near), lean::Polytope(P).contains(near));
        oracle::check(P.contains(far), lean::Polytope(P).contains(far));
    });
    oracle::run("polytope.convHull", [](oracle::Random &rnd) {
        const Polytope a(aux_sphere(rnd, 3, 5)), b(aux_sphere(rnd, 3, 5, 0.5, 1.0));
        const Polytope theirs = lean::Polytope(a).convex_hull(b).gather();
        test::check(
            oracle::same_bits("convHull", a.convex_hull(b), lean::Polytope(aux_extreme(theirs))),
            "convHull: the extreme vertices of the oracle's list");
    });
    oracle::run("polytope.enclosePoints", [](oracle::Random &rnd) {
        const Tensor points = aux_sphere(rnd, 3, 9);
        const Polytope theirs = lean::Polytope::enclose_points(points).gather();
        test::check(oracle::same_bits("enclosePoints", Polytope::enclose_points(points),
                                      lean::Polytope(aux_extreme(theirs))),
                    "enclosePoints: the extreme points of the oracle's list");
    });
    oracle::run("interval.polytope", [](oracle::Random &rnd) {
        const Interval I = rnd.box(3);
        oracle::check(Polytope(I), lean::Polytope::from_interval(I));
    });
    oracle::run("polytope.conZonotope", [](oracle::Random &rnd) {
        const Polytope P(aux_sphere(rnd, 3, 7));
        // the oracle's set contains the polytope: CORA.cpp's set lies inside its hull
        oracle::check_enclosed("conZonotope", P.con_zonotope("exact:vertices"),
                               lean::Polytope(P).con_zonotope());
    });
    return test::finish("lean polytope");
}
