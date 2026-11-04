// test_zonotope_cubic_map - zonotope cubic_map: an enclosure of {(x' T x) x | x in Z}
//
// CORA: contSet/@zonotope/cubMap.m
// PROPERTY: contSet/@zonotope/cubMap.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

using Grid = std::vector<std::vector<Tensor>>;

/// n = 1, z = t x^3 for x in [0.5, 1.5]: x^3 = 1 + 1.5 b + 0.75 b^2 + 0.125 b^3, b^2 in [0, 1].
void single_cubic_map_by_hand(const std::string &b) {
    const Zonotope Z(column({1.0}), Tensor({{0.5}}));
    const Zonotope R = Z.cubic_map({{Tensor({{2.0}})}});
    check(close(R.center(), std::vector<double>{2.75}, 1e-14), b + ": center 2 (1 + 0.375)");
    check(close(R.generators(), std::vector<double>{3.0, 0.75, 0.25}, 1e-14),
          b + ": generators 2 (1.5, 0.375, 0.125)");
    // a mixed map (x1 t x2) x3 has the products of all factors as generators
    const Zonotope Z2(column({2.0}), Tensor({{1.0}})), Z3(column({3.0}), Tensor({{0.5}}));
    const Zonotope M = Z.cubic_map(Z2, Z3, {{Tensor({{1.0}})}});
    check(close(M.center(), std::vector<double>{6.0}, 0), b + ": mixed: center 1 * 2 * 3");
    check(close(M.generators(), std::vector<double>{3, 3, 1.5, 1, 0.5, 0.5, 0.25}, 1e-14),
          b + ": mixed: the products of the factors");
}

/// The generator count N(N+1)(N+2)/6 - 1 of a single map, and the columns of T left out by ind.
void count_and_ind(const std::string &b) {
    cora::Rng rng(22);
    const int n = 3, m = 3;
    const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, m);
    Grid T(2, std::vector<Tensor>());
    for (auto &row : T)
        for (int k = 0; k < n; ++k) {
            std::vector<double> data(n * n);
            rng.normal(data.data(), data.size(), 1.0);
            row.push_back(Tensor::from_data(data, {n, n}));
        }
    const Zonotope R = Z.cubic_map(T);
    check(R.dim() == 2 && R.generators().shape().back() == 19, b + ": 2 outputs, 19 generators");
    // ind {0, 2} equals T with the matrix of column 1 set to zero
    Grid T0 = T;
    for (auto &row : T0) row[1] = row[1].zeros_like();
    const Zonotope via_ind = Z.cubic_map(T, {{0, 2}, {0, 2}});
    check(close(via_ind.generators(), Z.cubic_map(T0).generators(), 1e-12), b + ": ind");
    check(close(via_ind.center(), Z.cubic_map(T0).center(), 1e-12), b + ": ind: center");
}

/// Stated property: z_i = sum_k x_k (x' T_ik x) of every sampled point lies in the result.
void the_map_of_the_set_is_enclosed(const std::string &b) {
    cora::Rng rng(23);
    const int n = 3, m = 4;
    const Zonotope Z = Zonotope::generate_random(rng, n, std::nullopt, m);
    Grid T(2, std::vector<Tensor>());
    for (auto &row : T)
        for (int k = 0; k < n; ++k) {
            std::vector<double> data(n * n);
            rng.normal(data.data(), data.size(), 1.0);
            row.push_back(Tensor::from_data(data, {n, n}));
        }
    const Zonotope R = Z.cubic_map(T);
    const Tensor points = Z.rand_point(100, rng);
    for (int s = 0; s < 100; ++s) {
        const Tensor x = points.slice(-1, s, s + 1);
        std::vector<double> z(2, 0.0);
        for (int i = 0; i < 2; ++i)
            for (int k = 0; k < n; ++k)
                z[i] += x.data()[k] * x.matrix_transpose().matmul(T[i][k]).matmul(x).item();
        for (int t = 0; t < 10; ++t) {
            const std::vector<double> d = test::random_direction(rng, 2);
            check(d[0] * z[0] + d[1] * z[1] <= test::support(R, d) + 1e-9,
                  b + ": a mapped point leaves the enclosure");
        }
    }
}

/// Wrong grids and lists are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z(column({1.0, 0.0}), Tensor({{0.5}, {0.2}}));
    const Tensor M({{1.0, 0.0}, {0.0, 1.0}});
    check(test::raises([&] { Z.cubic_map({}); }, Error::wrong_value), "cubic_map: no matrices");
    check(test::raises([&] { Z.cubic_map({{Tensor({{1.0}})}}); }, Error::wrong_value),
          "cubic_map: a 1x1 matrix for dimension 2");
    check(test::raises([&] { Z.cubic_map({{M, M}}, {{0, 5}}); }, Error::wrong_value),
          "cubic_map: ind out of range");
    check(test::raises([&] { Z.cubic_map(Zonotope(column({1.0}), Tensor({{1.0}})), Z, {{M, M}}); },
                       Error::dimension_mismatch), "cubic_map: mixed, dimension 1");
    const Tensor on = Z.cubic_map({{M, M}}).generators();
    ChecksScope off(false);
    check(test::close(Z.cubic_map({{M, M}}).generators(), on, 0), "cubic_map: same with checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        single_cubic_map_by_hand(b);
        count_and_ind(b);
        the_map_of_the_set_is_enclosed(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope cubic_map");
}
