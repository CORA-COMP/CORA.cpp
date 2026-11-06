// test_con_zonotope_zono_bundle - ConZonotope::zono_bundle: the bundle of a set
//
// CORA: contSet/@conZonotope/zonoBundle.m
// PROPERTY: contSet/@conZonotope/zonoBundle.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Without constraints one member; a fixed point is one point; the empty set is the empty bundle.
void zono_bundle_known(const std::string &b) {
    const ConZonotope plain(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}}));
    check(plain.zono_bundle().parallel_sets() == 1 &&
              close(plain.zono_bundle().zonotopes()[0].generators(), {1, 0, 0, 2}, 0),
          b + ": without constraints");
    const ConZonotope point(column({1.0, 1.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                            Tensor({{1.0, 0.0}, {0.0, 1.0}}), column({0.5, 0.25}));
    check(point.zono_bundle().parallel_sets() == 1 &&
              close(point.zono_bundle().zonotopes()[0].center(), {1.5, 1.25}, 1e-9),
          b + ": the constraints fix the factors");
    check(ConZonotope::empty(2).zono_bundle().representsa("emptySet"), b + ": the empty set");
}

/// Stated property: the bundle contains the sampled points and lies in the box of the set (up to
/// the slack of the slabs: the bundle is the same set, so its points are in the set).
void zono_bundle_property(const std::string &b) {
    Rng rng(78);
    for (int k = 0; k < 3; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 3, 5, 2);
        const ZonoBundle bundle = cz.zono_bundle();
        check(bundle.contains(test::sample_points(cz, rng, 15)), b + ": the points are inside");
        check(cz.contains(bundle.rand_point(8, rng, "standard"), "exact", 1e-6),
              b + ": the points of the bundle are in the set");
    }
    const ConZonotope hex = test::hexagon();
    check(hex.zono_bundle().contains(
              Tensor::from_data({1, 1, 0, -1, -1, 0, 0, -1, -1, 0, 1, 1}, {2, 6})),
          b + ": the corners of the hexagon");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        zono_bundle_known(b);
        zono_bundle_property(b);
    });
    const Tensor c = Tensor::from_data({0.0, 0.0, 1.0, 1.0}, {2, 2, 1});
    check(test::raises(
              [&] {
                  ConZonotope(c, Tensor::zeros({2, 2, 1}), Tensor::zeros({2, 1, 1}),
                              Tensor::zeros({2, 1, 1}))
                      .zono_bundle();
              },
              Error::not_supported),
          "zono_bundle: a batch");
    ChecksScope off(false);
    check(test::hexagon().zono_bundle().dim() == 2, "zono_bundle: the same with the checks off");
    return test::finish("con_zonotope zono_bundle");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
