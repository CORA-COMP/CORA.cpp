// test_con_zonotope_representsa - ConZonotope::representsa: whether it is a set of another type
//
// CORA: contSet/@conZonotope/representsa_.m

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// Hand-computed answers for a box, a segment, a point and a set without points.
void representsa_known(const std::string &b) {
    const ConZonotope box(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}}));
    const ConZonotope segment(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                              Tensor({{1.0, 1.0}}), column({1.0}));
    const ConZonotope point(column({1.0, 2.0}), Tensor({{0.0}, {0.0}}));
    const ConZonotope none(column({0.0}), Tensor({{1.0}}), Tensor({{1.0}}), column({3.0}));
    check(box.representsa("zonotope") && box.representsa("interval"), b + ": a box");
    check(!segment.representsa("zonotope") && !segment.representsa("interval"),
          b + ": a segment with a constraint is neither");
    check(!box.representsa("emptySet") && !segment.representsa("emptySet") &&
              none.representsa("emptySet"),
          b + ": emptySet");
    check(point.representsa("point") && !box.representsa("point") && !point.representsa("origin") &&
              ConZonotope::origin(2).representsa("origin"),
          b + ": point and origin");
    check(box.representsa("conZonotope") && box.representsa("polytope") && box.representsa("convexSet") &&
              !box.representsa("halfspace") && !box.representsa("fullspace"),
          b + ": the constant types");
    const ConZonotope line(column({0.0}), Tensor({{1.0}}));
    const ConZonotope dot(column({1.0}), Tensor({{0.0}}));
    check(!line.representsa("hyperplane") && dot.representsa("hyperplane") &&
              !box.representsa("hyperplane"),
          b + ": a hyperplane is a point in one dimension");
}

} // namespace

int main() {
    test::for_each_backend(representsa_known);
    const ConZonotope box(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { box.representsa("cube"); }, Error::wrong_value), "representsa: a type");
    for (const char *type : {"conHyperplane", "capsule", "ellipsoid", "levelSet", "parallelotope"})
        check(test::raises([&] { box.representsa(type); }, Error::not_supported),
              std::string("representsa: ") + type);
    check(test::raises([&] { box.representsa("point", -1.0); }, Error::wrong_value), "representsa: tol");
    ChecksScope off(false);
    check(box.representsa("zonotope"), "representsa: the same with the checks off");
    return test::finish("con_zonotope representsa");
}
