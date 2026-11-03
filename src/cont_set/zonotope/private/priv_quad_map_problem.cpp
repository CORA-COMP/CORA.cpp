// priv_quad_map_problem - what is wrong with the matrices of a quadratic map
//
// Syntax:   std::string problem = priv_quad_map_problem(Q, n);
// Inputs:   Q - the matrices of the outputs;  n - dimension of the sets
// Outputs:  problem - empty if Q is one or more (n, n) matrices, else what is wrong
// See also: Zonotope::quad_map, Interval::quad_map
//
// CORA: contSet/@zonotope/quadMap.m

#include "cont_set/zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::string priv_quad_map_problem(const std::vector<Tensor> &Q, int64_t n) {
    if (Q.empty()) return "Q needs at least one matrix";
    for (const Tensor &M : Q)
        if (M.shape() != Shape{n, n})
            return "every matrix of Q must be (" + std::to_string(n) + ", " + std::to_string(n) +
                   "), got the shape " + shape_text(M.shape());
    return "";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
