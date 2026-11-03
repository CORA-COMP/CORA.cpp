// test_zonotope_generator_length - zonotope generator_length: the 2-norm of every generator
//
// CORA: contSet/@zonotope/generatorLength.m
// PROPERTY: contSet/@zonotope/generatorLength.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed lengths, no generators, and a batch.
void lengths_known(const std::string &b) {
    const Zonotope Z(column({7.0, 7.0}), Tensor({{3.0, 0.0, -1.0}, {4.0, 2.0, 0.0}}));
    const Tensor len = Z.generator_length();
    check(len.shape() == std::vector<int64_t>({1, 3}), b + ": a row, one length per generator");
    check(close(len, std::vector<double>{5, 2, 1}, 1e-14), b + ": (3,4), (0,2), (-1,0)");
    const Zonotope point(column({1.0, 2.0}), Tensor::zeros({2, 0}));
    check(point.generator_length().shape() == std::vector<int64_t>({1, 0}), b + ": no generators");
    const Zonotope batch = Zonotope::stack({Z, Z + column({1.0, 1.0})});
    check(batch.generator_length().shape() == std::vector<int64_t>({2, 1, 3}), b + ": a batch");
}

/// Stated property: the lengths bound every point's distance from the center,
/// |x - c|_2 <= sum of the lengths, and the support function along a unit d is at most their sum.
void lengths_bound_the_set(const std::string &b) {
    cora::Rng rng(5);
    const Zonotope Z = Zonotope::generate_random(rng, 4, std::nullopt, 6);
    const double total = Z.generator_length().sum().item();
    for (int k = 0; k < 40; ++k) {
        std::vector<double> d = test::random_direction(rng, 4);
        double norm = 0;
        for (const double x : d) norm += x * x;
        for (double &x : d) x /= std::sqrt(norm);
        const double reach = test::support(Z, d) - Z.center().data()[0] * d[0] -
                             Z.center().data()[1] * d[1] - Z.center().data()[2] * d[2] -
                             Z.center().data()[3] * d[3];
        check(reach <= total + 1e-9, b + ": the support function exceeds the sum of the lengths");
    }
}

/// With the checks off the result is the same.
void checks_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{3.0}, {4.0}}));
    ChecksScope off(false);
    check(close(Z.generator_length(), std::vector<double>{5}, 1e-14),
          "generator_length: checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        lengths_known(b);
        lengths_bound_the_set(b);
    });
    checks_switch();
    return test::finish("zonotope generator_length");
}
