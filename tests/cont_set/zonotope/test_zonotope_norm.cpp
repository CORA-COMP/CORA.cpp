// test_zonotope_norm - zonotope norm: an upper bound on the norm of the points
//
// CORA: contSet/@zonotope/norm_.m
// CORA: contSet/@interval/norm_.m
// PROPERTY: contSet/@zonotope/norm_.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Hand-computed: the box around the zonotope is [0, 2] x [-3, 1], whose largest magnitudes are (2,
/// 3).
void norm_known(const std::string &b) {
    const Zonotope Z(column({1.0, -1.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}}));
    check(close(Z.norm(), std::vector<double>{std::sqrt(13.0)}, 1e-14),
          b + ": the 2-norm, by default");
    check(close(Z.norm(2, "ub"), Z.norm(), 0), b + ": 'ub' is the default mode");
    check(close(Z.norm(1), std::vector<double>{5}, 0), b + ": the 1-norm");
    check(close(Z.norm(kInf), std::vector<double>{3}, 0), b + ": the inf-norm");
    const Zonotope batch = Zonotope::stack({Z, Z + column({10.0, 0.0})});
    check(batch.norm(1).shape() == std::vector<int64_t>({2, 1, 1}),
          b + ": a batch gives one per member");
    check(close(batch.norm(1).slice(0, 1, 2).squeeze(0), std::vector<double>{15}, 0),
          b + ": member 2: the box [10, 12] x [-3, 1]");
}

/// Stated property: no point of the zonotope has a larger norm, and the bound is attained by the
/// box.
void norm_bounds_the_points(const std::string &b) {
    cora::Rng rng(15);
    const Zonotope Z = Zonotope::generate_random(3, 4, rng);
    const Tensor points = Z.rand_point(200, rng, "extreme");
    const std::vector<double> x = points.data();
    for (const double type : {1.0, 2.0, kInf}) {
        const double bound = Z.norm(type).item();
        for (int k = 0; k < 200; ++k) {
            double one = 0, two = 0, inf = 0;
            for (int i = 0; i < 3; ++i) {
                const double v = std::abs(x[i * 200 + k]);
                one += v, two += v * v, inf = std::max(inf, v);
            }
            const double value = type == 1.0 ? one : type == 2.0 ? std::sqrt(two) : inf;
            check(value <= bound + 1e-9, b + ": a point exceeds the bound");
        }
    }
}

/// The modes that need a solver and an unknown type or mode are refused; with the checks off the
/// result is the same.
void norm_refuses() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    check(test::raises([&] { Z.norm(2, "exact"); }, Error::not_supported), "norm: mode exact");
    check(test::raises([&] { Z.norm(2, "ub_convex"); }, Error::not_supported),
          "norm: mode ub_convex");
    check(test::raises([&] { Z.norm(2, "other"); }, Error::wrong_value), "norm: an unknown mode");
    check(test::raises([&] { Z.norm(3); }, Error::wrong_value), "norm: type 3");
    const Tensor on = Z.norm();
    ChecksScope off(false);
    check(close(Z.norm(), on, 0), "norm: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        norm_known(b);
        norm_bounds_the_points(b);
    });
    norm_refuses();
    return test::finish("zonotope norm");
}
