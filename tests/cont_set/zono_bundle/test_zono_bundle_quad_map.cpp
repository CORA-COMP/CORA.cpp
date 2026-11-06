// test_zono_bundle_quad_map - ZonoBundle::quad_map: the quadratic map of a bundle
//
// CORA: contSet/@zonoBundle/quadMap.m
// PROPERTY: contSet/@zonoBundle/quadMap.m
// CORA authors: Niklas Kochdumper
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

/// The image x' Q x of every column of the points, as a row.
Tensor aux_image(const Tensor &Q, const Tensor &x) { return x.multiply(Q.matmul(x)).sum(0, true); }

/// Every member goes through Zonotope::quad_map.
void quad_map_known(const std::string &b) {
    const ZonoBundle zono_bundle = test::two_zonotopes();
    const std::vector<Tensor> Q{Tensor({{1.0, 0.0}, {0.0, 1.0}})};
    const ZonoBundle R = zono_bundle.quad_map(Q);
    const Zonotope first = zono_bundle.zonotopes()[0].quad_map(Q);
    check(R.dim() == 1 && R.parallel_sets() == 2 &&
              close(R.zonotopes()[0].center(), first.center(), 0) &&
              close(R.zonotopes()[0].generators(), first.generators(), 0),
          b + ": the members are the zonotopes' maps");
}

/// Stated property: x' Q x of every point of the bundle is in the image.
void quad_map_property(const std::string &b) {
    Rng rng(64);
    Tensor common;
    const ZonoBundle zono_bundle = test::random_zono_bundle(rng, 3, 2, 3, &common);
    const std::vector<Tensor> Q{Tensor({{1.0, 0.5, 0.0}, {-1.0, 2.0, 0.0}, {0.0, 0.0, 1.0}}),
                                Tensor({{0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 0.0, 0.0}})};
    const Tensor x = test::bundle_points(zono_bundle, rng, 12, common);
    check(zono_bundle.quad_map(Q).contains(
              Tensor::concat({aux_image(Q[0], x), aux_image(Q[1], x)}, 0)),
          b + ": the images are inside");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        quad_map_known(b);
        quad_map_property(b);
    });
    const ZonoBundle zono_bundle = test::two_zonotopes();
    check(test::raises([&] { zono_bundle.quad_map({Tensor::zeros({3, 3})}); },
                       Error::dimension_mismatch),
          "quad_map: a matrix of the wrong size");
    ChecksScope off(false);
    check(zono_bundle.quad_map({Tensor({{1.0, 0.0}, {0.0, 1.0}})}).dim() == 1,
          "quad_map: the same with the checks off");
    return test::finish("zono_bundle quad_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
