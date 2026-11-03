// test_interval_representsa - interval representsa: whether a box is also a set of another type
//
// CORA: contSet/@interval/representsa_.m
// CORA: contSet/@contSet/representsa_emptyObject.m
// CORA: contSet/@contSet/representsa.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::column;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Interval box(const std::vector<double> &lo, const std::vector<double> &hi) {
    return {column(lo), column(hi)};
}

void points_and_the_origin(const std::string &b) {
    const Interval p = box({1, 2}, {1, 2}), O = box({0, 0}, {0, 0}), I = box({0, 0}, {1, 2});
    check(p.representsa("point", 0) && O.representsa("point", 0) && !I.representsa("point", 0),
          b + ": point");
    check(O.representsa("origin", 0) && !p.representsa("origin", 0) && !I.representsa("origin", 0),
          b + ": origin");
    const Interval near = box({-1e-10, 0}, {1e-10, 0});
    check(near.representsa("origin", 1e-9) && !near.representsa("origin", 0),
          b + ": the tolerance of a width and of the origin");
    check(O.representsa("origin") && !I.representsa("origin"), b + ": the repository tolerance");
    const Interval S = Interval::stack({O, p});
    check(!S.representsa("origin", 0) && Interval::stack({O, O}).representsa("origin", 0),
          b + ": a batch must hold for every member");
}

void lines_planes_and_halfspaces(const std::string &b) {
    const Interval full = box({0, 0}, {1, 2}), seg = box({0, 2}, {1, 2}), pt = box({1, 2}, {1, 2});
    const Interval line1 = box({0}, {5});
    for (const char *type : {"capsule", "ellipsoid"})
        check(seg.representsa(type, 0) && pt.representsa(type, 0) && line1.representsa(type, 0) &&
                  !full.representsa(type, 0),
              b + ": " + type + ": at most one coordinate with a width (CORA accepts the box)");
    check(seg.representsa("conHyperplane", 0) && pt.representsa("conHyperplane", 0) &&
              line1.representsa("conHyperplane", 0) && !full.representsa("conHyperplane", 0),
          b + ": conHyperplane: a coordinate of width 0");
    check(seg.representsa("hyperplane", 0) && !pt.representsa("hyperplane", 0) &&
              !full.representsa("hyperplane", 0),
          b + ": hyperplane: exactly one coordinate of width 0");
    const Interval half = box({0, -kInf}, {kInf, kInf});
    check(half.representsa("halfspace") && !full.representsa("halfspace") &&
              !box({-kInf, -kInf}, {kInf, kInf}).representsa("halfspace"),
          b + ": halfspace: all bounds infinite but one");
}

void sets_that_every_box_is(const std::string &b) {
    const Interval I = box({0, 0}, {1, 2});
    for (const char *type : {"interval", "zonotope", "conZonotope", "polytope", "polyZonotope",
                             "zonoBundle", "conPolyZono", "levelSet", "parallelotope", "convexSet"})
        check(I.representsa(type), b + ": " + type);
    check(!I.representsa("probZonotope") && !I.representsa("emptySet"),
          b + ": probZonotope, emptySet");
    check(box({-kInf, -kInf}, {kInf, kInf}).representsa("fullspace") &&
              !I.representsa("fullspace") &&
              !box({0, -kInf}, {kInf, kInf}).representsa("fullspace"),
          b + ": fullspace");
}

void the_empty_interval(const std::string &b) {
    const Interval E = Interval::empty(2);
    for (const char *type : {"emptySet", "interval", "polytope", "hyperplane", "capsule",
                             "zonotope", "conZonotope", "halfspace", "probZonotope"})
        check(E.representsa(type), b + ": the empty interval is " + type);
    for (const char *type : {"origin", "point", "parallelotope", "fullspace"})
        check(!E.representsa(type), b + ": the empty interval is not " + type);
}

/// An unknown type, polygon, a matrix of intervals and a negative tolerance are refused; with the
/// checks off the result is the same.
void checks_and_their_switch() {
    const Interval I = box({0, 0}, {1, 2});
    const Interval M(Tensor::from_data({0, 0, 0, 0}, {2, 2}),
                     Tensor::from_data({1, 1, 1, 1}, {2, 2}));
    check(test::raises([&] { I.representsa("circle"); }, Error::wrong_value),
          "representsa: a type");
    check(test::raises([&] { I.representsa("polygon"); }, Error::not_supported),
          "representsa: polygon");
    check(test::raises([&] { M.representsa("point"); }, Error::not_supported),
          "representsa: a matrix of intervals");
    check(test::raises([&] { I.representsa("point", -1.0); }, Error::wrong_value),
          "representsa: a negative tolerance");
    check(!M.representsa("origin") && !M.representsa("emptySet"),
          "representsa: matrices, origin and emptySet");
    ChecksScope off(false);
    check(!I.representsa("point") && I.representsa("zonotope"),
          "representsa: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        points_and_the_origin(b);
        lines_planes_and_halfspaces(b);
        sets_that_every_box_is(b);
        the_empty_interval(b);
    });
    checks_and_their_switch();
    return test::finish("interval representsa");
}
