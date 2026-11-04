// test_zonotope_tensor_multiplication_zono - zonotope tensor_multiplication_zono: a tensor zonotope
//
// CORA: contSet/@zonotope/tensorMultiplication_zono.m
// PROPERTY: contSet/@zonotope/tensorMultiplication_zono.m

#include "tensor_lists.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// n = 1, the tensor 1 + 0.5 a, g = (1, 2): the results for M = 1 and for M = 0.5 are joined.
void one_dimension_by_hand(const std::string &b) {
    const Zonotope Z(column({0.0}), Tensor({{1.0, 2.0}}));
    const Tensor one = Tensor::from_data({1.0}, {1, 1, 1, 1});
    const Zonotope S = Z.tensor_multiplication_zono(one, {one * 0.5}, test::triples(2));
    check(close(S.generators(), std::vector<double>{1, 6, 12, 8, 0.5, 3, 6, 4}, 1e-14),
          b + ": the generators for the center and for the generator");
}

/// Stated property: the result for every member M_c + sum a_i M_i of the tensor zonotope lies in
/// the sum, which frees the factors a_i to [-1, 1].
void every_member_of_the_tensor_zonotope_is_enclosed(const std::string &b) {
    cora::Rng rng(32);
    const int n = 2, m = 2;
    const Zonotope Z(column({0.0, 0.0}),
                     Zonotope::generate_random(rng, n, std::nullopt, m).generators());
    const Tensor Mc = test::random_tensor(rng, n), M1 = test::random_tensor(rng, n),
                 M2 = test::random_tensor(rng, n);
    const Zonotope::TensorLists list = test::triples(m);
    const Zonotope R = Z.tensor_multiplication_zono(Mc, {M1, M2}, list);
    check(R.generators().shape().back() == 3 * int64_t(list.comb.size()), b + ": three parts");
    for (const auto &[a1, a2] : {std::pair{0.7, -0.4}, {1.0, 1.0}, {-1.0, 0.2}}) {
        const Zonotope member = Z.tensor_multiplication(Mc + M1 * a1 + M2 * a2, list);
        for (int t = 0; t < 10; ++t) {
            const std::vector<double> d = test::random_direction(rng, n);
            check(test::support(member, d) <= test::support(R, d) + 1e-9,
                  b + ": the result of a member lies in the sum");
        }
    }
}

/// A generator of the wrong shape is refused; with the checks off the call gives the same result.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0}), Tensor({{1.0, 2.0}}));
    const Tensor M = Tensor::from_data({2.0}, {1, 1, 1, 1});
    const Zonotope::TensorLists list = test::triples(2);
    check(test::raises(
              [&] {
                  Z.tensor_multiplication_zono(M, {Tensor::from_data({1.0, 1.0}, {1, 1, 1, 2})},
                                               list);
              },
              Error::dimension_mismatch),
          "tensor_multiplication_zono: a generator's shape");
    check(test::raises([&] { Z.tensor_multiplication_zono(M, {M}, {list.perm, list.comb, {}}); },
                       Error::wrong_value),
          "tensor_multiplication_zono: map shorter than perm");
    const Tensor on = Z.tensor_multiplication_zono(M, {M}, list).generators();
    ChecksScope off(false);
    check(close(Z.tensor_multiplication_zono(M, {M}, list).generators(), on, 0),
          "tensor_multiplication_zono: same with checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        one_dimension_by_hand(b);
        every_member_of_the_tensor_zonotope_is_enclosed(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope tensor_multiplication_zono");
}
