// test_zonotope_representsa - zonotope representsa: whether a zonotope is also a set of another type
//
// CORA: contSet/@zonotope/representsa_.m
// CORA: contSet/@contSet/representsa.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

namespace {

Zonotope zonotope(const std::vector<double> &c, const Tensor &G) { return {column(c), G}; }

void points_and_the_origin(const std::string &b) {
    const Zonotope origin = Zonotope::origin(2), point = zonotope({1, 2}, Tensor::zeros({2, 0}));
    const Zonotope zeros = zonotope({0, 0}, Tensor::zeros({2, 2}));
    const Zonotope seg = zonotope({0, 0}, Tensor({{1.0}, {0.0}}));
    check(origin.representsa("origin", 0) && zeros.representsa("origin", 0) &&
              !point.representsa("origin", 0) && !seg.representsa("origin", 0),
          b + ": origin (generators of zeros are dropped)");
    check(origin.representsa("point", 0) && point.representsa("point", 0) &&
              zeros.representsa("point", 0) && !seg.representsa("point", 0),
          b + ": point");
    const Zonotope near = zonotope({1e-10, 0}, Tensor({{1e-10}, {0.0}}));
    check(near.representsa("origin", 1e-9) && !near.representsa("origin", 0) &&
              near.representsa("point", 1e-9) && !near.representsa("point", 0),
          b + ": the tolerance of a width and of the origin");
    check(origin.representsa("origin") && !seg.representsa("origin"), b + ": the repository tolerance");
}

void intervals_parallelotopes_and_capsules(const std::string &b) {
    const Zonotope box = zonotope({0, 0}, Tensor({{1.0, 0.0}, {0.0, 2.0}}));
    const Zonotope tilted = zonotope({0, 0}, Tensor({{1.0, 0.0}, {1.0, 2.0}}));
    const Zonotope hexagon = zonotope({0, 0}, Tensor({{1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}));
    const Zonotope seg = zonotope({0, 0, 0}, Tensor({{1.0}, {1.0}, {0.0}}));
    check(box.representsa("interval", 0) && !tilted.representsa("interval", 0) &&
              !hexagon.representsa("interval", 0) && seg.representsa("interval", 0) == false,
          b + ": interval: at most one entry per generator");
    check(zonotope({0}, Tensor({{1.0, 2.0}})).representsa("interval", 0), b + ": interval in 1-D");
    check(box.representsa("parallelotope", 0) && tilted.representsa("parallelotope", 0) &&
              !hexagon.representsa("parallelotope", 0) && !seg.representsa("parallelotope", 0),
          b + ": parallelotope: n independent generators");
    check(zonotope({0}, Tensor({{1.0, 2.0}})).representsa("parallelotope", 0),
          b + ": parallelotope in 1-D");
    check(seg.representsa("capsule", 0) && !hexagon.representsa("capsule", 0) &&
              zonotope({0}, Tensor({{1.0, 2.0}})).representsa("capsule", 0),
          b + ": capsule: at most one generator, or one dimension");
}

void sets_that_every_zonotope_is(const std::string &b) {
    const Zonotope Z = zonotope({0, 0}, Tensor({{1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}));
    for (const char *type : {"zonotope", "conZonotope", "polytope", "polyZonotope", "zonoBundle",
                             "conPolyZono", "levelSet", "convexSet"})
        check(Z.representsa(type), b + ": " + type);
    for (const char *type : {"halfspace", "hyperplane", "probZonotope", "emptySet", "fullspace"})
        check(!Z.representsa(type), b + ": not " + type);
}

/// An unknown type, the unsupported ones, a batch and a negative tolerance are refused.
void checks_and_their_switch() {
    const Zonotope Z = zonotope({0, 0}, Tensor({{1.0}, {0.0}}));
    check(test::raises([&] { Z.representsa("circle"); }, Error::wrong_value), "representsa: a type");
    for (const char *type : {"conHyperplane", "ellipsoid", "polygon"})
        check(test::raises([&] { Z.representsa(type); }, Error::not_supported),
              std::string("representsa: ") + type);
    check(test::raises([&] { Zonotope::stack({Z, Z}).representsa("point"); }, Error::not_supported),
          "representsa: a batch");
    check(test::raises([&] { Z.representsa("point", -1.0); }, Error::wrong_value),
          "representsa: a negative tol");
    ChecksScope off(false);
    check(Z.representsa("capsule") && !Z.representsa("point"), "representsa: checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        points_and_the_origin(b);
        intervals_parallelotopes_and_capsules(b);
        sets_that_every_zonotope_is(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope representsa");
}
