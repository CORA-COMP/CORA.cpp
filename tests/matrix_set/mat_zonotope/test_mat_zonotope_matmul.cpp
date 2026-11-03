// test_mat_zonotope_matmul - MatZonotope matmul: the product with sets, matrices and numbers
//
// CORA: matrixSet/@matZonotope/mtimes.m
// PROPERTY: matrixSet/@matZonotope/mtimes.m

#include "mat_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// A = I + b [[0, 1], [0, 0]] times B = 2 I + b1 [[1, 0], [0, 0]] + b2 [[0, 0], [0, 1]]: the matrices
/// are [I, GA] and [2I, GB1, GB2], the product's center is 2 I and its generators are, in
/// CORALean's order, GA 2I, I GB1, I GB2, GA GB1, GA GB2.
void product_of_two_sets(const std::string &b) {
    const MatZonotope A = test::set_of({1, 0, 0, 1}, {0, 1, 0, 0}, 2, 2);
    const MatZonotope B = test::set_of({2, 0, 0, 2}, {1, 0, 0, 0, 0, 0, 0, 1}, 2, 2);
    const MatZonotope P = A.matmul(B);
    check(close(P.center(), {2, 0, 0, 2}), b + ": the center");
    check(P.num_generators() == 5, b + ": (1 + 1)(1 + 2) - 1 generators");
    check(close(P.generators(), {0, 2, 0, 0,     // GA 2I
                                 1, 0, 0, 0,     // I GB1
                                 0, 0, 0, 1,     // I GB2
                                 0, 0, 0, 0,     // GA GB1
                                 0, 1, 0, 0}),   // GA GB2
          b + ": the generators, in CORALean's order");
    check(test::same_set(A * B, P, 0), b + ": the operator");
}

/// The product of members is the member of the product with the coefficients b_i, b_j and b_i b_j,
/// exactly; a batch multiplies per set.
void members_multiply(const std::string &b) {
    Rng rng(21);
    const MatZonotope A = test::random_set(rng, 3, 2, 0.5), B = test::random_set(rng, 3, 3, 0.5);
    const std::vector<double> ca = test::coefficients(rng, 2), cb = test::coefficients(rng, 3);
    // the coefficients of the center, of G_A(i) C_B, of C_A G_B(j) and of G_A(i) G_B(j), j fastest
    std::vector<double> both{1.0};
    both.insert(both.end(), ca.begin(), ca.end());
    both.insert(both.end(), cb.begin(), cb.end());
    for (const double ai : ca)
        for (const double bj : cb) both.push_back(ai * bj);
    const Tensor expected = test::member(A, ca).matmul(test::member(B, cb));
    const Tensor center = A.matmul(B).center();
    const MatZonotope P = A.matmul(B);
    Tensor sum = center;
    for (std::size_t k = 1; k < both.size(); ++k)
        sum = sum + P.generators().slice(0, k - 1, k).squeeze(0) * both[k];
    check(close(sum, expected, 1e-12), b + ": the product of members");
    const MatZonotope batch = test::batch_of(A, A).matmul(test::batch_of(B, B));
    check(test::same_set(test::batch_member(batch, 1), P, 1e-12), b + ": a batch");
    const MatZonotope point = A.matmul(MatZonotope(B.center()));
    check(point.num_generators() == 2 &&
              close(point.center(), A.center().matmul(B.center()), 1e-12),
          b + ": a point on the right keeps A's generators");
}

/// A matrix on either side and a number scale exactly: the center and every generator.
void matrices_and_numbers(const std::string &b) {
    const MatZonotope A = test::example();
    const Tensor M = test::matrix({1, 1, 0, 1}, 2, 2);
    const MatZonotope R = A.matmul(M), L = MatZonotope::matmul(M, A), S = A.matmul(2.0);
    check(close(R.center(), {1, 3, 3, 7}) && close(L.center(), {4, 6, 3, 4}), b + ": the centers");
    check(close(R.generators(), {1, 1, 0, 1, 0, 1, 1, 1}), b + ": generators times M");
    check(close(L.generators(), {1, 1, 0, 1, 1, 1, 1, 0}), b + ": M times generators");
    check(close(S.center(), {2, 4, 6, 8}) && close(S.generators(), {2, 0, 0, 2, 0, 2, 2, 0}),
          b + ": the number 2");
    check(test::same_set(2.0 * A, S, 0) && test::same_set(A * 2.0, S, 0) &&
              test::same_set(A * M, R, 0) && test::same_set(M * A, L, 0),
          b + ": the operators");
    Rng rng(22);
    const MatZonotope Q = test::random_set(rng, 3, 2, 0.5);
    const Tensor N = test::member(test::random_set(rng, 3, 1, 0.5), {0.3});
    const std::vector<double> coeff = test::coefficients(rng, 2);
    check(close(test::member(Q.matmul(N), coeff), test::member(Q, coeff).matmul(N), 1e-12) &&
              close(test::member(MatZonotope::matmul(N, Q), coeff),
                    N.matmul(test::member(Q, coeff)), 1e-12),
          b + ": members times a matrix");
}

/// A (C, G1) times Z (c, g): the center C c, the generators C g, G1 c and G1 g.
void product_with_a_zonotope(const std::string &b) {
    const MatZonotope A = test::set_of({1, 2, 3, 4}, {1, 0, 0, 1}, 2, 2);
    const Zonotope Z(test::column({1, 1}), test::matrix({1, 0}, 2, 1));
    const Zonotope P = A.matmul(Z);
    check(close(P.center(), {3, 7}), b + ": the center C c");
    check(P.generators().shape() == std::vector<int64_t>({2, 3}), b + ": 1 + 1 + 1 generators");
    check(close(P.generators(), {1, 1, 1,   // C g, G1 c, G1 g: first row
                                 3, 1, 0}), // second row
          b + ": the generators C g, G1 c, G1 g");
    Rng rng(23);
    const MatZonotope Q = test::random_set(rng, 3, 2, 0.5);
    const Zonotope W(test::column({1, -1, 2}), test::matrix({1, 0.5, 0, 1, 2, -1}, 3, 2));
    const std::vector<double> cq = test::coefficients(rng, 2), cw = test::coefficients(rng, 2);
    // the point of the product for the coefficients cq of Q, cw of W: the products Gi g_j weigh
    // cq_i cw_j
    const Zonotope R = Q.matmul(W);
    std::vector<double> weights;
    weights.insert(weights.end(), cw.begin(), cw.end());
    weights.insert(weights.end(), cq.begin(), cq.end());
    for (const double cj : cw)
        for (const double ci : cq) weights.push_back(ci * cj);
    const int64_t count = int64_t(weights.size());
    const Tensor point =
        R.center() + R.generators().matmul(Tensor::like(R.center(), weights, {count, 1}));
    const Tensor w = W.center() + W.generators().matmul(Tensor::like(W.center(), cw, {2, 1}));
    const Tensor expected = test::member(Q, cq).matmul(w);
    check(close(point, expected, 1e-12), b + ": the product of a member and a point");
    check(close((Q * W).center(), R.center(), 0), b + ": the operator");
}

/// Inner dimensions that differ, a batch mismatch and a 1 x 1 scalar factor are refused; with the
/// checks off a valid product is the same.
void checks_and_their_switch() {
    const MatZonotope A = test::example(), scalar(test::matrix({2}, 1, 1));
    const MatZonotope wide(test::matrix({1, 2, 3}, 1, 3));
    check(test::raises([&] { A.matmul(wide); }, Error::dimension_mismatch),
          "matmul: inner dimensions of two sets");
    check(test::raises([&] { A.matmul(test::matrix({1, 2, 3}, 1, 3)); }, Error::dimension_mismatch),
          "matmul: a matrix with another number of rows");
    check(test::raises([&] { MatZonotope::matmul(test::matrix({1, 2, 3}, 1, 3), A); },
                       Error::dimension_mismatch),
          "matmul: a matrix with another number of columns");
    check(test::raises(
              [&] { A.matmul(Zonotope(test::column({1, 2, 3}), test::matrix({1, 1, 1}, 3, 1))); },
              Error::dimension_mismatch),
          "matmul: a zonotope of another dimension");
    check(test::raises([&] { A.matmul(test::batch_of(A, A)); }, Error::dimension_mismatch),
          "matmul: a batch against one set");
    check(test::raises([&] { scalar.matmul(A); }, Error::not_supported),
          "matmul: a 1 x 1 set as a factor");
    check(test::raises([&] { A.matmul(scalar); }, Error::not_supported),
          "matmul: a 1 x 1 set on the right");
    check(test::raises(
              [&] { scalar.matmul(Zonotope(test::column({1, 2}), test::matrix({1, 1}, 2, 1))); },
              Error::not_supported),
          "matmul: a 1 x 1 set times a zonotope");
    const Tensor on = A.matmul(A).center();
    ChecksScope off(false);
    check(close(A.matmul(A).center(), on, 0), "matmul: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        product_of_two_sets(b);
        members_multiply(b);
        matrices_and_numbers(b);
        product_with_a_zonotope(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope matmul");
}
