// test_lean_capsule - the operations of Capsule that the CORALean oracle serves
//
// Bit for bit where CORALean builds the same capsule at nearest: center, convHull (and enclose of
// two capsules, which is the hull), empty, origin, enclosePoints, interval, isBounded, isFullDim,
// isequal, lift, mtimes and contains for a point, plus (the longer generator stays), project,
// reduce, radius, representsa (point, origin) and zonotope (the segment and the box of the ball).
// polytope is bit for bit too.
// isIntersecting is a certificate of disjointness (the bounding radii separate the centers in one
// coordinate) where CORA.cpp computes the least distance of the axes, so the oracle's answer may
// not contradict CORA.cpp's (no bit-exact test exists). k is the number of Newton steps of
// CORALean's roots.
//
// ORACLE: contSet/@capsule/center.m
// ORACLE: contSet/@capsule/contains_.m
// ORACLE: contSet/@capsule/convHull_.m
// ORACLE: contSet/@capsule/empty.m
// ORACLE: contSet/@capsule/enclose.m
// ORACLE: contSet/@capsule/enclosePoints.m
// ORACLE: contSet/@capsule/interval.m
// ORACLE: contSet/@capsule/isBounded.m
// ORACLE: contSet/@capsule/isFullDim.m
// ORACLE: contSet/@capsule/isIntersecting_.m
// ORACLE: contSet/@capsule/isequal.m
// ORACLE: contSet/@capsule/lift_.m
// ORACLE: contSet/@capsule/mtimes.m
// ORACLE: contSet/@capsule/origin.m
// ORACLE: contSet/@capsule/plus.m
// ORACLE: contSet/@capsule/polytope.m
// ORACLE: contSet/@capsule/radius.m
// ORACLE: contSet/@capsule/project.m
// ORACLE: contSet/@capsule/reduce.m
// ORACLE: contSet/@capsule/representsa_.m
// ORACLE: contSet/@capsule/zonotope.m
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cora/global/oracle/capsule.h"
#include "cora/global/oracle/polytope.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A point near the axis of C, at the parameter t (beyond the ends for |t| > 1) and the offset
/// `offset` times r along a random direction: sometimes inside, sometimes outside.
Tensor aux_point(oracle::Random &rnd, const Capsule &C, double offset) {
    const std::vector<double> c = C.c().data(), g = C.generator().data(),
                              d = rnd.vector(c.size()).data();
    const double t = rnd.uniform(-1.5, 1.5), r = C.r().data()[0];
    double length = 0;
    for (const double x : d) length += x * x;
    std::vector<double> p(c.size());
    for (std::size_t i = 0; i < p.size(); ++i)
        p[i] = c[i] + t * g[i] + offset * r * d[i] / std::sqrt(length);
    return Tensor::from_data(p, {int64_t(p.size()), 1});
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("capsule.center", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        oracle::check(C.center(), lean::Capsule(C).center());
    });
    oracle::run("capsule.contains", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        // points at 0.5 r (inside unless beyond an end) and at 1.5 r from the axis
        for (const double offset : {0.5, 1.5}) {
            const Tensor p = aux_point(rnd, C, offset);
            oracle::check(C.contains(p), lean::Capsule(C).contains(p));
        }
    });
    oracle::run("capsule.convHull", [](oracle::Random &rnd) {
        const Capsule a = rnd.capsule(3), b = rnd.capsule(3);
        oracle::check(a.convex_hull(b), lean::Capsule(a).convex_hull(b));
    });
    // CORA's enclose of two capsules is their hull; with a matrix, the same hull, unsupported
    oracle::run("capsule.enclose", [](oracle::Random &rnd) {
        const Capsule a = rnd.capsule(3), b = rnd.capsule(3);
        oracle::check(a.enclose(b), lean::Capsule(a).convex_hull(b));
    });
    oracle::run("capsule.empty", [](oracle::Random &) {
        oracle::check(Capsule::empty(3), lean::Capsule::empty(3));
    });
    oracle::run("capsule.origin", [](oracle::Random &) {
        oracle::check(Capsule::origin(3), lean::Capsule::origin(3));
    });
    oracle::run("capsule.enclosePoints", [](oracle::Random &rnd) {
        const Tensor points = rnd.matrix(3, 12);
        oracle::check(Capsule::enclose_points(points), lean::Capsule::enclose_points(points));
    });
    // the box, and the predicates on it
    oracle::run("capsule.interval", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        oracle::check(C.interval(), lean::Capsule(C).interval());
    });
    oracle::run("capsule.isBounded", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        // is_bounded and is_full_dim against the oracle
        oracle::check(C.is_bounded(), lean::Capsule(C).is_bounded());
    });
    oracle::run("capsule.isFullDim", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        oracle::check(C.is_full_dim(), lean::Capsule(C).is_full_dim());
        // a segment is not full-dimensional in R^3, and in R^1 it is
        const Capsule segment(C.c(), C.generator(), 0.0);
        oracle::check(segment.is_full_dim(), lean::Capsule(segment).is_full_dim());
        const Capsule line(rnd.vector(1), rnd.vector(1), 0.0);
        oracle::check(line.is_full_dim(), lean::Capsule(line).is_full_dim());
    });
    // equality with another capsule and with itself
    oracle::run("capsule.isequal", [](oracle::Random &rnd) {
        const Capsule a = rnd.capsule(3), b = rnd.capsule(3);
        oracle::check(a.isequal(b, 0.0), lean::Capsule(a).isequal(b));
        oracle::check(a.isequal(a, 0.0), lean::Capsule(a).isequal(a));
    });
    oracle::run("capsule.isIntersecting", [](oracle::Random &rnd) {
        for (const double spread : {1.0, 6.0}) {
            const Capsule a = rnd.capsule(3);
            const Capsule b(a.c() + rnd.vector(3) * spread, rnd.vector(3), rnd.uniform(0.1, 1.0));
            oracle::check(a.is_intersecting(b), lean::Capsule(a).is_intersecting(b));
        }
    });
    // the maps: lift into R^3 and a linear map to R^2
    oracle::run("capsule.lift", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        oracle::check(C.lift(3, {1, 2, 0}), lean::Capsule(C).lift(3, {1, 2, 0}));
    });
    oracle::run("capsule.mtimes", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        const Tensor M = rnd.matrix(2, 3);
        oracle::check(C.linear_map(M), lean::Capsule(C).linear_map(M));
    });
    // the sum (the longer generator stays) and the radius |g| + r, through a square root
    oracle::run("capsule.plus", [](oracle::Random &rnd) {
        const Capsule a = rnd.capsule(3), b = rnd.capsule(3);
        oracle::check(a.plus(b), lean::Capsule(a).plus(b));
        oracle::check(b.plus(a), lean::Capsule(b).plus(a));
    });
    oracle::run("capsule.radius", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        oracle::check(C.radius().item(), lean::Capsule(C).radius());
    });
    // the halfspaces in the Gram-Schmidt basis of g, then the projection and the reduction
    oracle::run("capsule.polytope", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3), line = rnd.capsule(1);
        // a ball, a generator along an axis, a general one and a line
        const Tensor zero = C.c().zeros_like();
        const Capsule ball(C.c(), zero, C.r().item()),
            axis(C.c(), Tensor::from_data({0.0, 2.5, 0.0}, {3, 1}), C.r().item());
        for (const Capsule *S : {&ball, &axis, &C, &line})
            oracle::check(S->polytope("outer"), lean::Capsule(*S).polytope());
    });
    oracle::run("capsule.project", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        oracle::check(C.project({2, 0}).as<Capsule>(), lean::Capsule(C).project({2, 0}));
    });
    oracle::run("capsule.reduce", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3);
        oracle::check(C.reduce(2).as<Capsule>(), lean::Capsule(C).reduce(2));
    });
    // the predicates on points
    oracle::run("capsule.representsa", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3), point(C.c()), origin = Capsule::origin(3);
        for (const char *type : {"point", "origin"})
            for (const Capsule *S : {&C, &point, &origin})
                oracle::check(S->representsa(type, 0.0), lean::Capsule(*S).representsa(type));
    });
    oracle::run("capsule.zonotope", [](oracle::Random &rnd) {
        const Capsule C = rnd.capsule(3), segment(C.c(), C.generator(), 0.0);
        oracle::check(C.zonotope(), lean::Capsule(C).zonotope());
        oracle::check(segment.zonotope(), lean::Capsule(segment).zonotope());
    });
    return test::finish("lean capsule");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
