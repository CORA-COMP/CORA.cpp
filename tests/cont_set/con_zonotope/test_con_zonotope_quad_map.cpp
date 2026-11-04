// test_con_zonotope_quad_map - ConZonotope::quad_map: an enclosure of a quadratic map
//
// CORA: contSet/@conZonotope/quadMap.m
// PROPERTY: contSet/@conZonotope/quadMap.m
//
// CORA authors:  Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The image x' Q x of every column of the points, as a row.
Tensor aux_image(const Tensor &Q, const Tensor &x) { return x.multiply(Q.matmul(x)).sum(0, true); }

/// Hand-computed: x^2 for x in [-1, 1] has the center 1/2 and two generators 1/2 (the square and
/// the linear term, which is 0); a constraint x1 + x2 = 0 stays on the linear block.
void quad_map_known(const std::string &b) {
    const ConZonotope unit(column({0.0}), Tensor({{1.0}}));
    const ConZonotope r = unit.quad_map({Tensor({{1.0}})});
    check(close(r.c(), {0.5}, 0) && close(r.generators(), {0.5, 0}, 0), b + ": x^2");
    const ConZonotope a(column({0.0}), Tensor({{1.0, 1.0}}), Tensor({{1.0, 1.0}}), column({0.0}));
    const ConZonotope q = a.quad_map({Tensor({{1.0}})});
    check(close(q.A(), {0, 0, 1, 1, 0}, 0) && close(q.b(), {0}, 0),
          b + ": the constraint on the linear block");
    const ConZonotope plain(column({1.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const ConZonotope m = plain.quad_map(plain, {Tensor({{1.0, 0.0}, {0.0, 1.0}})});
    // [c G]' [c G] has the center 1 and the entries of the products in column order
    check(close(m.c(), {1}, 0) && m.generators().shape() == Shape{1, 8}, b + ": the mixed map");
}

/// Stated property: the image of every sampled point (pair) is in the enclosure.
void quad_map_property(const std::string &b) {
    cora::Rng rng(45);
    for (int k = 0; k < 3; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 2, 3, 1);
        const ConZonotope other = test::random_con_zonotope(rng, 2, 3, 1);
        const std::vector<Tensor> Q{Tensor({{1.0, 0.5}, {-1.0, 2.0}}),
                                    Tensor({{0.0, 1.0}, {1.0, 0.0}})};
        const Tensor x = test::sample_points(cz, rng, 10), y = test::sample_points(other, rng, 10);
        const ConZonotope r = cz.quad_map(Q);
        const ConZonotope mixed = cz.quad_map(other, Q);
        const Tensor images = Tensor::concat({aux_image(Q[0], x), aux_image(Q[1], x)}, 0);
        check(r.contains(images), b + ": the single map encloses");
        const Tensor cross = Tensor::concat(
            {x.multiply(Q[0].matmul(y)).sum(0, true), x.multiply(Q[1].matmul(y)).sum(0, true)}, 0);
        check(mixed.contains(cross), b + ": the mixed map encloses");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        quad_map_known(b);
        quad_map_property(b);
    });
    const ConZonotope a(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    check(test::raises([&] { a.quad_map({Tensor({{1.0}})}); }, Error::dimension_mismatch),
          "quad_map: a matrix of the wrong size");
    check(test::raises([&] { a.quad_map({}); }, Error::dimension_mismatch), "quad_map: no matrix");
    ChecksScope off(false);
    check(a.quad_map({Tensor({{1.0, 0.0}, {0.0, 1.0}})}).dim() == 1, "quad_map: checks off");
    return test::finish("con_zonotope quad_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
