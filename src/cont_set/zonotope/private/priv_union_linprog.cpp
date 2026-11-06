// priv_union_linprog - the zonotope of CORA's 'linprog' union, from an enclosing zonotope
//
// A linear program scales and moves them to the least that still covers every set along the facet
// normals of the enclosure.
//
// Syntax:   Zonotope Zu = priv_union_linprog(enclosure, support);
// Inputs:   enclosure - a zonotope that contains every set of the union (the iterative union);
//           support - the largest support value over the sets along each row of A (k, n): (k, 1)
// Outputs:  Zu - the generators of the enclosure, unit length, scaled, with the center moved
// See also: Zonotope::unite, ConZonotope::unite
//
// CORA: contSet/@zonotope/or.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/global/linprog.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope priv_union_linprog(const Zonotope &iterative,
                            const std::function<Tensor(const Tensor &)> &support) {
    const DType dtype = iterative.center().dtype();
    const Zonotope enclosure = iterative.compact("zeros", machine_eps(dtype));
    const Tensor &G0 = enclosure.generators();
    const Tensor length = G0.square().sum(-2, true).sqrt();
    const Tensor G = G0.multiply(length.ones_like().divide(length)); // unit generators
    const int64_t n = enclosure.dim(), ng = G.shape().back();
    const auto [A, unused] = priv_zonotope_halfspaces({enclosure.center().zeros_like(), G0});
    const Tensor d = support(A);

    // min sum(scal) s.t. A c + |A G| scal >= d, scal >= 0
    const Tensor AG = A.matmul(G).abs();
    const int64_t k = A.shape()[0];
    LinProg lp;
    lp.n = int(n + ng);
    lp.f.assign(lp.n, 0.0);
    std::fill(lp.f.begin() + n, lp.f.end(), 1.0);
    const std::vector<double> a = A.data(), ag = AG.data(), dv = d.data();
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < n; ++j) lp.Aineq.push_back(-a[i * n + j]);
        for (int64_t j = 0; j < ng; ++j) lp.Aineq.push_back(-ag[i * ng + j]);
        lp.bineq.push_back(-dv[i]);
    }
    for (int64_t j = 0; j < ng; ++j) {
        for (int64_t l = 0; l < n + ng; ++l) lp.Aineq.push_back(l == n + j ? -1.0 : 0.0);
        lp.bineq.push_back(0.0);
    }
    const LinProgResult r = linprog(lp);
    if (r.status != LinProgResult::Status::optimal)
        CORA_FAIL(Error::solver_issue, "unite: the linear program of 'linprog' has no solution");
    const Tensor c = Tensor::like(G, std::vector<double>(r.x.begin(), r.x.begin() + n), {n, 1});
    // the scaling of the generators is the second part of the solution
    const Tensor scal = Tensor::like(G, std::vector<double>(r.x.begin() + n, r.x.end()), {1, ng});
    return {c, G.multiply(scal)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
