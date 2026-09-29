#include "contSet/zonotope.h"

namespace cora::ct {

Zonotope Zonotope::mtimes(const Tensor &M) const { return {M.matmul(c), M.matmul(G)}; }

Zonotope Zonotope::mtimes(const IntervalMatrix &M) const {
    const Tensor reach = c.abs() + G.abs().sum_last();
    const Tensor extra = M.radius.matmul(reach);
    return {M.center.matmul(c), Tensor::cat_last({M.center.matmul(G), extra.diag()})};
}

Zonotope Zonotope::plus(const Zonotope &Z2) const {
    return {c + Z2.c, Tensor::cat_last({G, Z2.G})};
}

Zonotope Zonotope::lin_comb(const Zonotope &Z2) const {
    return {(c + Z2.c) * 0.5,
            Tensor::cat_last({(G + Z2.G) * 0.5, (G - Z2.G) * 0.5, (c - Z2.c) * 0.5})};
}

Tensor Zonotope::support_func(const Tensor &d) const {
    const Tensor dt = d.transpose();
    return dt.matmul(c) + dt.matmul(G).abs().sum_last();
}

} // namespace cora::ct
