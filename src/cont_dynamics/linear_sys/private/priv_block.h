// priv_block - sets of a partitioned state, as CORA's block_mtimes, block_operation and recompose
//
// A state split into blocks is a list of zonotopes, one per block. A matrix maps the list by its
// blocks: the block i of M S is the sum over j of M(i, j) S_j, which drops the dependence of the
// blocks on each other and costs less than the product of the whole set.
//
// Syntax:   std::vector<Zonotope> Xb = priv_decompose(X, blocks);   priv_recompose(Xb);
//           priv_block_mtimes(M, blocks, Xb);   priv_block_plus(Xb, Yb);
// Inputs:   blocks - first and last coordinate of each block (0-based, inclusive)
//           M - matrix (..., n, n) or interval matrix
//           Xb - one zonotope per block
// Outputs:  the blocks of the result, or the whole set for priv_recompose
// See also: priv_steps.h, priv_reach_decomp
//
// CORA.cpp only: CORA spreads these over block_mtimes, block_operation, block_zeros and recompose.
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/linear_sys/linear_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

using Blocks = std::vector<std::array<int64_t, 2>>;

/// The one block that holds a state of dimension n.
Blocks priv_one_block(int64_t n);

/// The projections of X onto the blocks.
std::vector<Zonotope> priv_decompose(const Zonotope &X, const Blocks &blocks);

/// The Cartesian product of the blocks, in order.
Zonotope priv_recompose(const std::vector<Zonotope> &Xb);

/// M Xb for a matrix or an interval matrix, block by block.
std::vector<Zonotope> priv_block_mtimes(const Tensor &M, const Blocks &blocks,
                                        const std::vector<Zonotope> &Xb);
std::vector<Zonotope> priv_block_mtimes(const Interval &M, const Blocks &blocks,
                                        const std::vector<Zonotope> &Xb);

/// The Minkowski sum of two lists, block by block.
std::vector<Zonotope> priv_block_plus(const std::vector<Zonotope> &Xb,
                                      const std::vector<Zonotope> &Yb);

/// enclose(Xb[i], Yb[i]) for every block.
std::vector<Zonotope> priv_block_enclose(const std::vector<Zonotope> &Xb,
                                         const std::vector<Zonotope> &Yb);

/// The origin in every block, made like `like` (a tensor of the sets it is added to).
std::vector<Zonotope> priv_block_origin(const Tensor &like, const Blocks &blocks);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
