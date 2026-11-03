// test_con_zonotope_cubic_map - ConZonotope::cubic_map: an enclosure of a cubic map
//
// CORA: contSet/@conZonotope/cubMap.m
// PROPERTY: contSet/@conZonotope/cubMap.m

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// z = x1 (x1 x2) in the single output: T[0][0] picks x1 x2, T[0][1] is zero.
std::vector<std::vector<Tensor>> aux_map() {
    return {{Tensor({{0.0, 1.0}, {0.0, 0.0}}), Tensor({{0.0, 0.0}, {0.0, 0.0}})}};
}

/// The image of the point x under z_i = sum_k x_k (x' T[i][k] x), as a column.
Tensor aux_image(const std::vector<std::vector<Tensor>> &T, const Tensor &x) {
    std::vector<double> z;
    for (const auto &row : T) {
        double value = 0;
        for (std::size_t k = 0; k < row.size(); ++k)
            value += x.data()[k] * x.matrix_transpose().matmul(row[k]).matmul(x).item();
        z.push_back(value);
    }
    return Tensor::like(x, z, {int64_t(z.size()), 1});
}

/// Hand-computed: without constraints the zonotope's map; with them the same generators and the
/// constraints on the first columns, the linear monomials.
void cubic_map_known(const std::string &b) {
    const auto T = aux_map();
    const ConZonotope plain(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const ConZonotope R = plain.cubic_map(T);
    const Zonotope expected = Zonotope(plain.c(), plain.generators()).cubic_map(T);
    check(close(R.c(), expected.center(), 0) && close(R.generators(), expected.generators(), 0),
          b + ": without constraints: the zonotope's map");
    check(R.A().shape() == Shape{0, expected.generators().shape().back()},
          b + ": and no constraints");
    const ConZonotope constrained(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                                  Tensor({{1.0, 1.0}}), column({0.5}));
    const ConZonotope C = constrained.cubic_map(T);
    check(close(C.generators(), expected.generators(), 0), b + ": the same generators");
    check(C.A().shape() == Shape{1, expected.generators().shape().back()} &&
              close(C.A().slice(1, 0, 2), {1, 1}, 0) &&
              close(C.A().slice(1, 2, C.A().shape()[1]).abs().sum(), {0}, 0),
          b + ": the constraints on the linear monomials, zero elsewhere");
    check(close(C.b(), {0.5}, 0), b + ": b stays");
}

/// Stated property: the image of every point of the set is in the enclosure, for the single and
/// the mixed form, with an index list.
void cubic_map_property(const std::string &b) {
    cora::Rng rng(30);
    const std::vector<std::vector<Tensor>> T{
        {Tensor({{1.0, 0.5}, {0.0, -1.0}}), Tensor({{0.0, 1.0}, {1.0, 0.0}})},
        {Tensor({{0.5, 0.0}, {0.0, 0.5}}), Tensor({{1.0, 1.0}, {0.0, 0.0}})}};
    for (int k = 0; k < 3; ++k) {
        const ConZonotope A = test::random_con_zonotope(rng, 2, 3, 1);
        const ConZonotope R = A.cubic_map(T);
        const Tensor points = test::sample_points(A, rng, 12);
        for (int64_t j = 0; j < 12; ++j)
            check(R.contains(aux_image(T, points.slice(1, j, j + 1)), "exact", 1e-6),
                  b + ": the image of a point is in the enclosure");
        const std::vector<std::vector<int64_t>> ind{{0}, {1}};
        const ConZonotope S = A.cubic_map(T, ind);
        check(S.dim() == 2, b + ": with an index list");
    }
    // the mixed form (x1' T x2) x3 with three sets: one point of each
    const ConZonotope A = test::random_con_zonotope(rng, 2, 2, 1),
                      B = test::random_con_zonotope(rng, 2, 3, 1),
                      C = test::random_con_zonotope(rng, 2, 2, 0);
    const std::vector<std::vector<Tensor>> M{
        {Tensor({{1.0, 0.5}, {0.0, -1.0}}), Tensor({{0.0, 1.0}, {1.0, 0.0}})}};
    const ConZonotope R = A.cubic_map(B, C, M);
    const Tensor xa = test::sample_points(A, rng, 8), xb = test::sample_points(B, rng, 8),
                 xc = test::sample_points(C, rng, 8);
    for (int64_t j = 0; j < 8; ++j) {
        const Tensor x1 = xa.slice(1, j, j + 1), x2 = xb.slice(1, j, j + 1),
                     x3 = xc.slice(1, j, j + 1);
        double z = 0;
        for (int64_t kk = 0; kk < 2; ++kk)
            z += x3.data()[kk] * x1.matrix_transpose().matmul(M[0][kk]).matmul(x2).item();
        check(R.contains(Tensor::like(x1, {z}, {1, 1}), "exact", 1e-6),
              b + ": the mixed image is in");
    }
    check(R.A().shape()[0] == 2 && R.b().shape()[0] == 2,
          b + ": the rows of both constrained sets");
}

/// A map of the wrong size and other dimensions are refused; with the checks off the result is the
/// same.
void checks_and_their_switch() {
    const ConZonotope A(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(test::raises([&] { A.cubic_map({{Tensor({{1.0}})}}); }, Error::wrong_value),
          "cubic_map: the matrices are (n, n)");
    check(test::raises([&] { A.cubic_map(aux_map(), {{5}}); }, Error::wrong_value),
          "cubic_map: an index out of range");
    check(test::raises(
              [&] { A.cubic_map(ConZonotope(column({1.0}), Tensor({{1.0}})), A, aux_map()); },
              Error::dimension_mismatch),
          "cubic_map: the sets differ in dimension");
    const Tensor on = A.cubic_map(aux_map()).generators();
    ChecksScope off(false);
    check(close(A.cubic_map(aux_map()).generators(), on, 0),
          "cubic_map: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        cubic_map_known(b);
        cubic_map_property(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope cubic_map");
}
