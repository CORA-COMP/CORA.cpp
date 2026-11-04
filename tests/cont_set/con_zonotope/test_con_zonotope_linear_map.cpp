// test_con_zonotope_linear_map - ConZonotope::linear_map: the image under a matrix
//
// CORA: contSet/@conZonotope/mtimes.m
// PROPERTY: contSet/@conZonotope/mtimes.m

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: center and generators map, the constraints stay; M may change the dimension.
void linear_map_known(const std::string &b) {
    const ConZonotope a(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0, 1.0}}),
                        column({0.5}));
    const ConZonotope r = a.linear_map(Tensor({{2.0, 0.0}, {0.0, 3.0}, {1.0, 1.0}}));
    check(r.dim() == 3 && close(r.c(), {2, 6, 3}, 0), b + ": the center maps");
    check(close(r.generators(), {2, 0, 0, 3, 1, 1}, 0), b + ": the generators map");
    check(close(r.A(), a.A(), 0) && close(r.b(), a.b(), 0), b + ": the constraints stay");
    check(close((Tensor({{1.0, 0.0}}) * a).c(), {1}, 0), b + ": the operator");
    check(close(Set(a)->linear_map(Tensor({{0.0, 1.0}})).as<ConZonotope>().c(), {2}, 0),
          b + ": through the set interface");
}

/// Stated property: the image of every sampled point is a point of the image.
void linear_map_property(const std::string &b) {
    cora::Rng rng(42);
    for (int k = 0; k < 4; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 3, 4, 2);
        const Tensor M = Tensor::like(cz.c(), {1, 2, 0, -1, 0.5, 3}, {2, 3});
        const ConZonotope r = cz.linear_map(M);
        check(r.contains(M.matmul(test::sample_points(cz, rng, 20))), b + ": images are inside");
    }
}

/// A matrix with the wrong number of columns is refused.
void checks_and_batches() {
    const ConZonotope a(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { a.linear_map(Tensor({{1.0, 2.0, 3.0}})); }, Error::dimension_mismatch),
          "linear_map: columns");
    const Tensor M = Tensor::concat(
        {Tensor({{1.0, 0.0}}).expand_dims(0), Tensor({{0.0, 2.0}}).expand_dims(0)}, 0);
    check(a.linear_map(M).c().shape() == Shape{2, 1, 1}, "linear_map: a batch of matrices");
    ChecksScope off(false);
    check(a.linear_map(Tensor({{1.0, 1.0}})).dim() == 1, "linear_map: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        linear_map_known(b);
        linear_map_property(b);
    });
    checks_and_batches();
    return test::finish("con_zonotope linear_map");
}
