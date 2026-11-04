// test_con_zonotope_con_intersect - ConZonotope::con_intersect: the points whose image is in a set
//
// CORA: contSet/@conZonotope/conIntersect.m
// PROPERTY: contSet/@conZonotope/conIntersect.m

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: the points x of [0, 2] with 2 x in [1, 3] are [0.5, 1.5]; the rows of the result.
void con_intersect_known(const std::string &b) {
    const ConZonotope A(column({1.0}), Tensor({{1.0}}));
    const ConZonotope B(column({2.0}), Tensor({{1.0}}));
    const ConZonotope R = A.con_intersect(B, Tensor({{2.0}}));
    check(close(R.c(), {1}, 0) && close(R.generators(), {1, 0}, 0),
          b + ": the first set's c and G");
    check(close(R.A(), {2, -1}, 0) && close(R.b(), {0}, 0),
          b + ": the row M G x - G2 x2 = c2 - M c");
    check(close(R.interval().inf(), {0.5}, 1e-9) && close(R.interval().sup(), {1.5}, 1e-9),
          b + ": the points [0.5, 1.5]");
    // the constraints of both sets stay, first set's rows first
    const ConZonotope C(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0, 1.0}}),
                        column({1.0}));
    const ConZonotope D(column({0.0}), Tensor({{1.0, 1.0}}), Tensor({{1.0, -1.0}}), column({0.0}));
    const ConZonotope R2 = C.con_intersect(D, Tensor({{1.0, 0.0}}));
    check(close(R2.A(), {1, 1, 0, 0, 0, 0, 1, -1, 1, 0, -1, -1}, 0),
          b + ": the rows: C, D, agreement");
    check(close(R2.b(), {1, 0, 0}, 0), b + ": the offsets");
}

/// Stated property: a point x of the first set is in the result exactly if M x is in the second.
void con_intersect_property(const std::string &b) {
    cora::Rng rng(27);
    for (int k = 0; k < 4; ++k) {
        const ConZonotope A = test::random_con_zonotope(rng, 3, 4, 1);
        const ConZonotope B = test::random_con_zonotope(rng, 2, 3, 1);
        std::vector<double> m(6);
        rng.normal(m.data(), 6, 1.0);
        const Tensor M = Tensor::from_data(m, {2, 3});
        const ConZonotope R = A.con_intersect(B, M);
        const Tensor points = test::sample_points(A, rng, 60);
        for (int64_t j = 0; j < 60; ++j) {
            const Tensor x = points.slice(1, j, j + 1);
            check(R.contains(x, "exact", 1e-7) == B.contains(M.matmul(x), "exact", 1e-7),
                  b + ": x is in the result iff M x is in the second set");
        }
    }
}

/// A matrix of the wrong size is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const ConZonotope A(column({1.0}), Tensor({{1.0}}));
    check(
        test::raises([&] { A.con_intersect(A, Tensor({{1.0, 2.0}})); }, Error::dimension_mismatch),
        "con_intersect: M must map to the dimension of the second set");
    const Tensor on = A.con_intersect(A, Tensor({{1.0}})).A();
    ChecksScope off(false);
    check(close(A.con_intersect(A, Tensor({{1.0}})).A(), on, 0),
          "con_intersect: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        con_intersect_known(b);
        con_intersect_property(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope con_intersect");
}
