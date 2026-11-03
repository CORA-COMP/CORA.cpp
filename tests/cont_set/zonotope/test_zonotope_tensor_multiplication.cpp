// test_zonotope_tensor_multiplication - zonotope tensor_multiplication: the third-order image
//
// CORA: contSet/@zonotope/tensorMultiplication.m
// PROPERTY: contSet/@zonotope/tensorMultiplication.m

#include "tensor_lists.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

using Lists = Zonotope::TensorLists;

/// n = 1, M = 2, g = (1, 2): 2 (b1 + 2 b2)^3 has the generators 2 (1, 3 g1^2 g2, 3 g1 g2^2, g2^3).
void one_dimension_by_hand(const std::string &b) {
    const Zonotope Z(column({0.0}), Tensor({{1.0, 2.0}}));
    const Tensor M = Tensor::from_data({2.0}, {1, 1, 1, 1});
    const Zonotope R = Z.tensor_multiplication(M, test::triples(2));
    check(close(R.center(), std::vector<double>{0}, 0), b + ": the center is the origin");
    check(close(R.generators(), std::vector<double>{2, 12, 24, 16}, 1e-14),
          b + ": generators 2 (1, 6, 12, 8)");
}

/// Stated property: for x = G b the image y_i = sum M[i, j, k, l] x_j x_k x_l is H beta, with
/// beta the monomials b_a b_b b_c of the columns, so the image of every point lies in the result.
void the_image_of_the_set_is_enclosed(const std::string &b) {
    cora::Rng rng(31);
    const int n = 3, m = 3;
    const Zonotope Z(column({0.0, 0.0, 0.0}), Zonotope::generate_random(rng, n, std::nullopt, m).generators());
    const Tensor M = test::random_tensor(rng, n);
    const Lists list = test::triples(m);
    const Zonotope R = Z.tensor_multiplication(M, list);
    const int64_t columns = list.comb.size();
    check(R.generators().shape().back() == columns, b + ": one column per sorted triple");
    const Tensor points = Z.rand_point(30, rng);
    const std::vector<double> tensor = M.data(), H = R.generators().data();
    for (int s = 0; s < 30; ++s) {
        const Tensor point = points.slice(-1, s, s + 1);
        const std::vector<double> x = point.data();
        // the coefficients b of x: G is square
        const std::vector<double> bv = Z.generators().pinv().matmul(point).data();
        for (int i = 0; i < n; ++i) {
            double image = 0, enclosed = 0;
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k)
                    for (int l = 0; l < n; ++l)
                        image += tensor[((i * n + j) * n + k) * n + l] * x[j] * x[k] * x[l];
            for (int64_t q = 0; q < columns; ++q)
                enclosed += H[i * columns + q] * bv[list.comb[q][0]] * bv[list.comb[q][1]] *
                            bv[list.comb[q][2]];
            check(close(image, enclosed, 1e-9), b + ": H beta is the image of the point");
        }
    }
}

/// A second-order tensor, wrong shapes, bad lists and a center off the origin are refused; with
/// the checks off the same call gives the same result.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0}), Tensor({{1.0, 2.0}}));
    const Tensor M = Tensor::from_data({2.0}, {1, 1, 1, 1});
    const Lists list = test::triples(2);
    const auto raises = [&](const Zonotope &S, const Tensor &T, const Lists &l, Error::Id id) {
        return test::raises([&] { S.tensor_multiplication(T, l); }, id);
    };
    check(raises(Z, Tensor::from_data({2.0}, {1, 1, 1}), list, Error::not_supported),
          "tensor_multiplication: second order");
    check(raises(Z, Tensor::from_data({2.0, 1.0}, {1, 1, 1, 2}), list, Error::dimension_mismatch),
          "tensor_multiplication: shape");
    Lists bad = list;
    bad.perm[0][0] = 5;
    check(raises(Z, M, bad, Error::wrong_value), "tensor_multiplication: generator out of range");
    bad = list;
    bad.map[0] = 9;
    check(raises(Z, M, bad, Error::wrong_value), "tensor_multiplication: column out of range");
    bad = list;
    bad.perm[0].pop_back();
    check(raises(Z, M, bad, Error::wrong_value), "tensor_multiplication: a pair, not a triple");
    const Zonotope shifted(column({1.0}), Tensor({{1.0, 2.0}}));
    check(raises(shifted, M, list, Error::wrong_value), "tensor_multiplication: center not 0");
    const Tensor on = Z.tensor_multiplication(M, list).generators();
    ChecksScope off(false);
    check(close(Z.tensor_multiplication(M, list).generators(), on, 0),
          "tensor_multiplication: same with checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        one_dimension_by_hand(b);
        the_image_of_the_set_is_enclosed(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope tensor_multiplication");
}
