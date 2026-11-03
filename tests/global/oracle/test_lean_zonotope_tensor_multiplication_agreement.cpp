// test_lean_zonotope_tensor_multiplication_agreement - tensor_multiplication and its tensor-zonotope form
// of Zonotope against the CORALean oracle, by the interval hull
//
// The zonotope is centered at the origin, as CORA requires; the lists are CORA's options.list for
// m generators (the ordered triples, the sorted ones and the column of each). The oracle's M is the
// third-order tensor as n blocks of n matrices (n, n): M[i][j] is M(i, j, :, :) of CORA.cpp.
// CORA.cpp sums the products with a matrix product, CORALean in the order of its permutations, so
// the generators differ by a few ulp; the box they span agrees to a relative 1e-10 (as the cubic
// map). No bit-exact test exists, so there is no ORACLE tag.

#include "../../cont_set/zonotope/tensor_lists.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

namespace {

/// A tensor (n, n, n, n) of random numbers, and the same as the oracle's n blocks of n matrices.
struct Tensor4 {
    Tensor tensor;
    lean::Zonotope::Tensor3 blocks;
};

Tensor4 aux_tensor(oracle::Random &rnd, int n) {
    std::vector<double> data(n * n * n * n);
    for (double &x : data) x = rnd.scalar();
    Tensor4 result{Tensor::from_data(data, {n, n, n, n}), lean::Zonotope::Tensor3(n)};
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const auto first = data.begin() + (i * n + j) * n * n;
            result.blocks[i].push_back(
                lean::Tensor::from(Tensor::from_data({first, first + n * n}, {n, n})));
        }
    return result;
}

/// The oracle's lists for CORA's: the ordered triples, the column of each, and the column count.
lean::Zonotope::TensorLists aux_lists(const Zonotope::TensorLists &list) {
    lean::Zonotope::TensorLists result;
    for (const auto &triple : list.perm) result.perm.push_back(triple);
    result.map = list.map;
    result.C = static_cast<int64_t>(list.comb.size());
    return result;
}

/// A zonotope of n dimensions and m generators centered at the origin.
Zonotope aux_origin_centered(oracle::Random &rnd, int n, int m) {
    return {Tensor::zeros({n, 1}), rnd.matrix(n, m)};
}

} // namespace

int main() {
    if (!oracle::available()) return 0;
    oracle::run("zonotope.tensorMultiplication", [](oracle::Random &rnd) {
        for (const auto [n, m] : {std::pair{1, 2}, {2, 2}, {3, 3}}) {
            const Zonotope Z = aux_origin_centered(rnd, n, m);
            const Tensor4 M = aux_tensor(rnd, n);
            const Zonotope::TensorLists list = test::triples(m);
            oracle::check_hull(Z.tensor_multiplication(M.tensor, list),
                          lean::Zonotope(Z).tensor_multiplication(M.blocks, aux_lists(list)));
        }
    });
    oracle::run("zonotope.tensorMultiplication_zono", [](oracle::Random &rnd) {
        for (const auto [n, m] : {std::pair{1, 2}, {2, 2}, {3, 3}}) {
            const Zonotope Z = aux_origin_centered(rnd, n, m);
            const Tensor4 center = aux_tensor(rnd, n), first = aux_tensor(rnd, n),
                          second = aux_tensor(rnd, n);
            const Zonotope::TensorLists list = test::triples(m);
            oracle::check_hull(Z.tensor_multiplication_zono(center.tensor,
                                                            {first.tensor, second.tensor}, list),
                          lean::Zonotope(Z).tensor_multiplication_zono(
                              center.blocks, {first.blocks, second.blocks}, aux_lists(list)));
        }
    });
    return test::finish("lean zonotope tensor_multiplication");
}
