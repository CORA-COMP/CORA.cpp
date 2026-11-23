// cubic_map - the image of a polynomial zonotope under a cubic map, as CORA's cubMap
//
// As CORALean's cubMap: the independent generators become dependent first (each on a factor of its
// own, with one zero column per coordinate for the error box). Then every ordered triple (a, b, c)
// of the columns of [c, G] is a generator of the image, the first column the slowest index, with
// the sum of the three exponent columns: its entry i is sum_k a_k (b' T[i][k] c), the quadratic
// form summed first, the sum over k from the first term. The center is 0, a constant generator
// stays one. CORA instead keeps the independent part independent and encloses its cubic terms, and
// only for the symmetrized triples (docs/cora-differences.md).
//
// The map is z_i = sum_k x_k (x' T[i][k] x).
//
// Syntax:   S = pZ.cubic_map(T);   S = pZ.cubic_map(T, ind);
// Inputs:   T - T[i][k] is the (n, n) matrix of output i and column k
//           ind - for every output i the k that enter it (default: all of them)
// Outputs:  S - the image, of the dimension of the rows of T, without independent generators
// See also: quad_map, zonotope cubic_map
//
// CORA: contSet/@polyZonotope/cubMap.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The generators of output i: row a, column (b, c) of Zext T[i][k] summed over k, flattened.
Tensor aux_output(const Tensor &Zext, const std::vector<Tensor> &Ti,
                  const std::vector<int64_t> &ind) {
    const int64_t N = Zext.shape()[0], n = Zext.shape()[1];
    const Tensor Zt = Zext.matrix_transpose();
    // the forms b' T c of the columns, one row per k; a missing k is a zero row
    std::vector<Tensor> rows(std::size_t(n), Zext.zeros_like({1, N * N}));
    for (const int64_t k : ind)
        rows[std::size_t(k)] = Zext.matmul(Ti[std::size_t(k)].matmul(Zt)).reshape({1, N * N});
    const Tensor forms = Tensor::concat(rows, -2);
    return Zext.matmul(forms).reshape({1, N * N * N});
}

} // namespace


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::cubic_map(const std::vector<std::vector<Tensor>> &T,
                                     const std::vector<std::vector<int64_t>> &ind) const {
    detail::Entry entry;
    const int64_t n = dim();
    CORA_CHECK(!T.empty(), Error::wrong_value,
               [] { return std::string("cubic_map: T needs at least one output"); });
    // the matrices: one per output and coordinate, each (n, n)
    for (const std::vector<Tensor> &Ti : T) {
        CORA_CHECK(int64_t(Ti.size()) == n, Error::dimension_mismatch, [&] {
            return "cubic_map: every output needs " + std::to_string(n) + " matrices, got " +
                   std::to_string(Ti.size());
        });
        for (const Tensor &M : Ti)
            CORA_CHECK(M.ndim() == 2 && M.shape()[0] == n && M.shape()[1] == n,
                       Error::dimension_mismatch, [&] {
                           return "cubic_map: the matrices must be (" + std::to_string(n) + ", " +
                                  std::to_string(n) + "), got " + shape_text(M.shape());
                       });
    }
    CORA_CHECK(ind.empty() || ind.size() == T.size(), Error::dimension_mismatch,
               [] { return std::string("cubic_map: ind needs one list per output"); });
    // ind: the default is every k, a k outside the matrices is refused
    std::vector<int64_t> all;
    for (int64_t k = 0; k < n; ++k) all.push_back(k);
    for (const std::vector<int64_t> &list : ind)
        for (const int64_t k : list)
            CORA_CHECK(k >= 0 && k < n, Error::wrong_value,
                       [] { return std::string("cubic_map: ind outside the matrices"); });

    // the exact image of the dependent set, one output after the other
    const PolyZonotope D = priv_absorb_rounding(*this);
    const Tensor Zext = Tensor::concat({D.center().expand_dims(0), D.generators()}, -2);
    const Exponents Ee = priv_extended_exponents(D.exponents(), D.ids().size());
    const auto N = int64_t(Ee.size());
    std::vector<Tensor> outputs;
    for (std::size_t i = 0; i < T.size(); ++i)
        outputs.push_back(aux_output(Zext, T[i], ind.empty() ? all : ind[i]));
    // the exponents of the triples, the first column the slowest
    Exponents E;
    E.reserve(std::size_t(N * N * N));
    for (const std::vector<int> &a : Ee)
        for (const std::vector<int> &b : Ee)
            for (const std::vector<int> &c : Ee) {
                std::vector<int> e = a;
                for (std::size_t r = 0; r < e.size(); ++r) e[r] += b[r] + c[r];
                E.push_back(std::move(e));
            }
    return {Zext.zeros_like({int64_t(T.size())}),
            Tensor::concat(outputs, -2).matrix_transpose(),
            Zext.zeros_like({0, int64_t(T.size())}),
            std::move(E),
            D.ids(),
            detail::Verbatim{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
