// test_zono_bundle_project - ZonoBundle::project: the projection of a bundle onto some dimensions
//
// CORA: contSet/@zonoBundle/project.m
// PROPERTY: contSet/@zonoBundle/project.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the selected rows of every member, in the given order.
void project_known(const std::string &b) {
    const ZonoBundle zono_bundle = test::two_zonotopes();
    const ZonoBundle one = Set(zono_bundle)->project({1}).as<ZonoBundle>();
    check(one.dim() == 1 && one.parallel_sets() == 2 &&
              close(one.zonotopes()[0].generators(), {0, 1}, 0) &&
              close(one.zonotopes()[1].center(), {0}, 0) &&
              close(one.zonotopes()[1].generators(), {0, 1}, 0),
          b + ": the second row of every member");
    const ZonoBundle swapped = Set(zono_bundle)->project({1, 0}).as<ZonoBundle>();
    check(close(swapped.zonotopes()[1].center(), {0, 1}, 0) &&
              close(swapped.zonotopes()[1].generators(), {0, 1, 1, 1}, 0),
          b + ": the rows in the given order");
}

/// Stated property: the projection of every point of the bundle is in the projection.
void project_property(const std::string &b) {
    Rng rng(63);
    Tensor common;
    const ZonoBundle zono_bundle = test::random_zono_bundle(rng, 4, 2, 5, &common);
    const Tensor points = test::bundle_points(zono_bundle, rng, 12, common);
    check(Set(zono_bundle)->project({3, 1}).as<ZonoBundle>().contains(points.take({3, 1}, 0)),
          b + ": the projected points are inside");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        project_known(b);
        project_property(b);
    });
    const ZonoBundle zono_bundle = test::two_zonotopes();
    check(test::raises([&] { zono_bundle.project({2}); }, Error::wrong_value),
          "project: a dimension too big");
    ChecksScope off(false);
    check(zono_bundle.project({0}).as<ZonoBundle>().dim() == 1,
          "project: the same with the checks off");
    return test::finish("zono_bundle project");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
