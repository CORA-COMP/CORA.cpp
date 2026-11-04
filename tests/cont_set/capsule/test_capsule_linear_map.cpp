// test_capsule_linear_map - Capsule linear_map: M * C, enclosed by scaling the radius
//
// CORA: contSet/@capsule/mtimes.m
// PROPERTY: contSet/@capsule/mtimes.m

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: the center and generator map by M, the radius scales by the Frobenius norm.
void linear_map_known(const std::string &b) {
    const Capsule C(column({1.0, 2.0}), column({1.0, -1.0}), 0.5);
    const Tensor M({{2.0, 0.0}, {0.0, 3.0}});
    const Capsule R = C.linear_map(M);
    check(close(R.c(), {2, 6}, 0) && close(R.generator(), {2, -3}, 0), b + ": c and g map by M");
    check(close(R.r(), {0.5 * std::sqrt(13.0)}, 1e-14), b + ": r scales by |M|_F");
    // a map to another dimension
    const Capsule line = C.linear_map(Tensor({{1.0, 1.0}}));
    check(line.dim() == 1 && close(line.c(), {3}, 0) && close(line.generator(), {0}, 0) &&
              close(line.r(), {0.5 * std::sqrt(2.0)}, 1e-14),
          b + ": M (1, 2)");
    check(close((M * C).c(), R.c(), 0), b + ": M * C");
    check(Capsule::empty(2).linear_map(M).is_empty(), b + ": the empty capsule stays empty");
    check(close(C.linear_map(M.zeros_like()).r(), {0}, 0), b + ": the zero map gives a point");
    // a batch of maps and a batch of capsules
    const Tensor Ms = Tensor::from_data({1, 0, 0, 1, 2, 0, 0, 2}, {2, 2, 2});
    check(close(C.linear_map(Ms).c(), {1, 2, 2, 4}, 0), b + ": a batch of maps");
}

/// Stated property: the image of every sampled point lies in the image capsule, also for a map
/// that shrinks (CORA scales by max(eig(M M')) = |M|^2 < |M| there and misses points).
void linear_map_property(const std::string &b) {
    Rng rng(19);
    for (int k = 0; k < 8; ++k) {
        const Capsule C = test::random_capsule(rng, 3);
        std::vector<double> m(6);
        rng.normal(m.data(), m.size(), k % 2 ? 0.3 : 2.0);
        const Tensor M = Tensor::from_data(m, {2, 3});
        const Tensor points = test::sample_points(C, rng, 30);
        check(C.linear_map(M).contains(M.matmul(points)), b + ": the image points are inside");
    }
}

/// A matrix with the wrong number of columns is refused; with the checks off the same call works.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0}), column({1.0, -1.0}), 0.5);
    check(test::raises([&] { C.linear_map(Tensor({{1.0, 2.0, 3.0}})); }, Error::dimension_mismatch),
          "linear_map: M needs dim(C) columns");
    const Tensor M({{2.0, 0.0}, {0.0, 3.0}});
    const Capsule on = C.linear_map(M);
    ChecksScope off(false);
    check(close(C.linear_map(M).r(), on.r(), 0), "linear_map: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        linear_map_known(b);
        linear_map_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule linear_map");
}
