// test_zonotope_radius - zonotope radius: a bound on the distance from the center
//
// CORA: contSet/@zonotope/radius.m
// PROPERTY: contSet/@zonotope/radius.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// CORA's unit test: G = [1 3; 2 1] has the box radius (4, 3), whose norm is 5 (the sum of the
/// generator lengths, 5.4, is larger); a point has radius 0; a batch gives one per member.
void radius_known(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 3.0}, {2.0, 1.0}}));
    check(close(Z.radius(), std::vector<double>{5}, 1e-14),
          b + ": the box radius (4, 3) has norm 5");
    check(Z.radius().shape() == std::vector<int64_t>({1, 1}), b + ": a (1, 1) result");
    check(close(Zonotope::origin(3).radius(), std::vector<double>{0}, 0), b + ": a point");
    const Zonotope axis(column({1.0, 1.0}), Tensor({{3.0}, {4.0}}));
    check(close(axis.radius(), std::vector<double>{5}, 1e-14), b + ": one generator (3, 4)");
    const Tensor batch = Zonotope::stack({Z, axis}).radius();
    check(batch.shape() == std::vector<int64_t>({2, 1, 1}) &&
              close(batch, std::vector<double>{5, 5}, 1e-14),
          b + ": a batch");
}

/// Stated property: every point of the set is within radius of the center, and the radius never
/// exceeds the sum of the generator lengths.
void radius_bounds_the_set(const std::string &b) {
    cora::Rng rng(31);
    for (const auto &[n, m] : {std::pair{2, 3}, {4, 6}, {3, 1}}) {
        const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, m);
        const double r = Z.radius().item();
        check(r <= Z.generator_length().sum().item() + 1e-12,
              b + ": at most the sum of the lengths");
        const Tensor points = Z.rand_point(200, rng, "extreme") - Z.center();
        const Tensor distance = points.vector_norm(Tensor::Norm::two, -2);
        check(distance.max().item() <= r + 1e-12, b + ": a point beyond the radius");
    }
}

/// With the checks off the result is the same.
void checks_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 3.0}, {2.0, 1.0}}));
    ChecksScope off(false);
    check(close(Z.radius(), std::vector<double>{5}, 1e-14), "radius: checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        radius_known(b);
        radius_bounds_the_set(b);
    });
    checks_switch();
    return test::finish("zonotope radius");
}
