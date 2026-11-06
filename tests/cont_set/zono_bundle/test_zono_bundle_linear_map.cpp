// test_zono_bundle_linear_map - ZonoBundle::linear_map: the image of a bundle under a matrix
//
// CORA: contSet/@zonoBundle/mtimes.m
// PROPERTY: contSet/@zonoBundle/mtimes.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: every member is mapped on its own; a row map drops the dimension.
void linear_map_known(const std::string &b) {
    const ZonoBundle zono_bundle = test::two_zonotopes();
    const Tensor M({{1.0, 0.0}, {0.0, 2.0}});
    const ZonoBundle R = zono_bundle.linear_map(M);
    check(R.parallel_sets() == 2 && close(R.zonotopes()[0].generators(), {2, 0, 0, 2}, 0) &&
              close(R.zonotopes()[1].center(), {1, 0}, 0) &&
              close(R.zonotopes()[1].generators(), {1, 1, 0, 2}, 0),
          b + ": the members are M c and M G");
    check(zono_bundle.linear_map(Tensor({{1.0, 1.0}})).dim() == 1,
          b + ": a row map gives one dimension");
    const Set via_set = Set(zono_bundle)->linear_map(M);
    check(via_set.is<ZonoBundle>() &&
              close(via_set.as<ZonoBundle>().zonotopes()[1].generators(), {1, 1, 0, 2}, 0),
          b + ": through the interface");
}

/// Stated property: the image of every point of the bundle is in the image.
void linear_map_property(const std::string &b) {
    Rng rng(61);
    for (int k = 0; k < 3; ++k) {
        Tensor common;
        const ZonoBundle zono_bundle = test::random_zono_bundle(rng, 3, 2, 4, &common);
        const Tensor M = Tensor::from_data({1.0, -2.0, 0.5, 0.0, 1.0, 3.0}, {2, 3});
        const Tensor points = test::bundle_points(zono_bundle, rng, 15, common);
        check(zono_bundle.linear_map(M).contains(M.matmul(points)), b + ": the images are inside");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        linear_map_known(b);
        linear_map_property(b);
    });
    const ZonoBundle zono_bundle = test::two_zonotopes();
    check(test::raises([&] { zono_bundle.linear_map(Tensor({{1.0, 0.0, 0.0}})); },
                       Error::dimension_mismatch),
          "linear_map: a matrix with too many columns");
    ChecksScope off(false);
    check(zono_bundle.linear_map(Tensor({{1.0, 0.0}})).dim() == 1,
          "linear_map: the same with the checks off");
    return test::finish("zono_bundle linear_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
