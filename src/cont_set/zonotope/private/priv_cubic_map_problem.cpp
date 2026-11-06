// priv_cubic_map_problem - what is wrong with the matrices and the indices of a cubic map
//
// Syntax:   std::string problem = priv_cubic_map_problem(T, ind, n);
// Inputs:   T - the (n, n) matrices of the outputs, T[i][k]
//           ind - the columns k of each output (empty: all)
//           n - dimension of the sets
// Outputs:  problem - empty if T and ind are consistent, else what is wrong
// See also: Zonotope::cubic_map, ConZonotope::cubic_map
//
// CORA: contSet/@zonotope/cubMap.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string priv_cubic_map_problem(const std::vector<std::vector<Tensor>> &T,
                                   const std::vector<std::vector<int64_t>> &ind, int64_t n) {
    if (T.empty() || T.front().empty())
        return "T needs at least one row and one column of matrices";
    for (const auto &row : T) {
        if (row.size() != T.front().size()) return "the rows of T must have the same length";
        for (const Tensor &M : row)
            if (M.shape() != Shape{n, n})
                return "every matrix of T must be (" + std::to_string(n) + ", " +
                       std::to_string(n) + "), got the shape " + shape_text(M.shape());
    }
    if (ind.empty()) return "";
    if (ind.size() != T.size()) return "ind needs one list per row of T";
    const int64_t columns = std::min<int64_t>(int64_t(T.front().size()), n);
    for (const auto &list : ind)
        for (const int64_t k : list)
            if (k < 0 || k >= columns)
                return "ind has the entry " + std::to_string(k) + ", outside [0, " +
                       std::to_string(columns) + ")";
    // no offending index
    return "";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
