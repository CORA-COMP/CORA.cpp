// stack - a batch of zonotopes as one zonotope
//
// Syntax:   Z = Zonotope::stack({Z1, Z2, ...});
// Inputs:   Zs - zonotopes of one dimension and batch shape; the generator counts may differ
// Outputs:  Z - one zonotope whose leading batch dimension indexes Zs, in order
// See also: plus, contSet/interval/stack

#include "contSet/zonotope/zonotope.h"

#include <algorithm>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// G with zero columns appended up to m generators.
Tensor aux_padGenerators(const Tensor &G, int64_t m) {
    std::vector<int64_t> shape = G.shape();
    const int64_t missing = m - shape.back();
    if (missing == 0) return G;
    shape.back() = missing;
    int64_t count = 1;
    for (int64_t d : shape) count *= d;
    return Tensor::catLast({G, Tensor::like(G, std::vector<double>(count, 0.0), shape)});
}

} // namespace

// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::stack(const std::vector<Zonotope> &Zs) {
    if (Zs.empty()) throw std::invalid_argument("Zonotope::stack: no zonotopes given");
    int64_t m = 0;
    for (const Zonotope &Z : Zs) m = std::max(m, Z.G.shape().back());
    std::vector<Tensor> cs, Gs;
    for (const Zonotope &Z : Zs) {
        cs.push_back(Z.c);
        Gs.push_back(aux_padGenerators(Z.G, m));
    }
    return {Tensor::stack(cs), Tensor::stack(Gs)};
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
