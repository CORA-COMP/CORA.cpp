// priv_block - sets of a partitioned state, as CORA's block_mtimes, block_operation and recompose
//
// See also: priv_block.h
//
// CORA.cpp only: CORA spreads these over block_mtimes, block_operation, block_zeros and recompose.
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_block.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The entries of M in the rows and columns of two blocks.
Tensor aux_sub(const Tensor &M, const std::array<int64_t, 2> &rows,
               const std::array<int64_t, 2> &cols) {
    return M.slice(-2, rows[0], rows[1] + 1).slice(-1, cols[0], cols[1] + 1);
}

Interval aux_sub(const Interval &M, const std::array<int64_t, 2> &rows,
                 const std::array<int64_t, 2> &cols) {
    return {aux_sub(M.inf(), rows, cols), aux_sub(M.sup(), rows, cols)};
}

/// The block i of M Xb: the sum over j of M(i, j) Xb_j.
template <class Matrix>
std::vector<Zonotope> aux_mtimes(const Matrix &M, const Blocks &blocks,
                                 const std::vector<Zonotope> &Xb) {
    std::vector<Zonotope> out;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        // the own block comes first, so its generators stay aligned with those of Xb_i (enclose)
        std::optional<Zonotope> sum;
        for (std::size_t k = 0; k < blocks.size(); ++k) {
            const std::size_t j = k == 0 ? i : (k <= i ? k - 1 : k);
            Zonotope term = Xb[j].linear_map(aux_sub(M, blocks[i], blocks[j]));
            sum = sum ? sum->plus(term) : std::move(term);
        }
        out.push_back(std::move(*sum));
    }
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Blocks priv_one_block(int64_t n) { return {{0, n - 1}}; }

std::vector<Zonotope> priv_decompose(const Zonotope &X, const Blocks &blocks) {
    // one block that spans the state is the set itself
    if (blocks.size() == 1) return {X};
    std::vector<Zonotope> parts;
    for (const Set &S : X.decompose(blocks)) parts.push_back(S.as<Zonotope>());
    return parts;
}

Zonotope priv_recompose(const std::vector<Zonotope> &Xb) {
    Zonotope X = Xb.front();
    for (std::size_t i = 1; i < Xb.size(); ++i) X = X.cart_prod(Xb[i]);
    return X;
}

// the same product for a matrix and for an interval matrix
std::vector<Zonotope> priv_block_mtimes(const Tensor &M, const Blocks &blocks,
                                        const std::vector<Zonotope> &Xb) {
    return aux_mtimes(M, blocks, Xb);
}

std::vector<Zonotope> priv_block_mtimes(const Interval &M, const Blocks &blocks,
                                        const std::vector<Zonotope> &Xb) {
    return aux_mtimes(M, blocks, Xb);
}

// the Minkowski sums and the enclosures act on the blocks one by one
std::vector<Zonotope> priv_block_plus(const std::vector<Zonotope> &Xb,
                                      const std::vector<Zonotope> &Yb) {
    std::vector<Zonotope> out;
    for (std::size_t i = 0; i < Xb.size(); ++i) out.push_back(Xb[i].plus(Yb[i]));
    return out;
}

std::vector<Zonotope> priv_block_enclose(const std::vector<Zonotope> &Xb,
                                         const std::vector<Zonotope> &Yb) {
    std::vector<Zonotope> out;
    for (std::size_t i = 0; i < Xb.size(); ++i) out.push_back(Xb[i].enclose(Yb[i]));
    return out;
}

std::vector<Zonotope> priv_block_origin(const Tensor &like, const Blocks &blocks) {
    std::vector<Zonotope> out;
    for (const std::array<int64_t, 2> &block : blocks) {
        const int64_t n = block[1] - block[0] + 1;
        out.emplace_back(like.zeros_like({n, 1}), like.zeros_like({n, 0}));
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
