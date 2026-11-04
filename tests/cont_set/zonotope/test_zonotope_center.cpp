// test_zonotope_center - zonotope center and dim
//
// CORA: contSet/@zonotope/zonotope.m
// CORA: contSet/@zonotope/center.m
// PROPERTY: contSet/@zonotope/center.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

void dim_and_center(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0, 0.5}, {0.0, 2.0, 0.5}}));
    check(Z.dim() == 2, b + ": dim");
    check(close(Z.center(), std::vector<double>{1, 2}), b + ": center");
    const ContSet &S = Z;
    check(close(S.center(), std::vector<double>{1, 2}, 0), b + ": the center of the abstract set");
}

/// Stated property: the center is the midpoint of the support function along d and along -d.
void center_is_the_midpoint(const std::string &b) {
    cora::Rng rng(13);
    const Zonotope Z = Zonotope::generate_random(rng, 3, std::nullopt, 5);
    for (int k = 0; k < 10; ++k) {
        const std::vector<double> d = test::random_direction(rng, 3);
        std::vector<double> minus(d);
        for (double &x : minus) x = -x;
        const double mid = (test::support(Z, d) - test::support(Z, minus)) / 2;
        const std::vector<double> c = Z.center().data();
        double dot = 0;
        for (std::size_t i = 0; i < d.size(); ++i) dot += c[i] * d[i];
        check(close(mid, dot, 1e-9), b + ": midpoint");
    }
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        dim_and_center(b);
        center_is_the_midpoint(b);
    });
    ChecksScope off(false);
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0}, {0.5}}));
    check(close(Z.center(), std::vector<double>{1, 2}, 0), "center: the same with the checks off");
    return test::finish("zonotope center");
}
