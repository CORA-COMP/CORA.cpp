// unite - a zonotope that encloses the union of zonotopes, as CORA's zonotope.or
//
// Syntax:   Zu = Z.unite(Z2);   Zu = Z.unite({Z2, Z3}, "iterative");   Zu = Z.unite(I);
// Inputs:   Zs - the other sets: zonotopes, a box (as its zonotope) or a point;  method -
//           "linprog" (default): the enclosure of pairwise enclose, its generators rescaled and
//           its center moved, by a linear program, to the least that still covers every set
//           along its facet normals; "iterative": the pairwise enclose alone; "parallelotope":
//           the box around all of them in the frame of the principal axes of their generators.
//           "tedrake" and "althoff", and an order reduction (order), are Error::not_supported
// Outputs:  Zu - a zonotope that contains the union. Not a batch
// See also: enclose, convex_hull, reduce
//
// CORA: contSet/@zonotope/or.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/linprog.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <algorithm>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The three methods -------------------------------------------------------------------------------

/// The sets enclosed pairwise, level by level, until one is left (CORA's aux_unionIterative).
Zonotope aux_iterative(std::vector<Zonotope> Zs) {
    while (Zs.size() > 1) {
        std::vector<Zonotope> next;
        for (std::size_t i = 0; i + 1 < Zs.size(); i += 2) next.push_back(Zs[i].enclose(Zs[i + 1]));
        if (Zs.size() % 2 == 1) next.push_back(Zs.back());
        Zs = std::move(next);
    }
    return Zs.front();
}

/// The enclosure of the iterative union with the least generator scalings and the center that
/// cover every set along the facet normals of that enclosure (CORA's aux_unionLinprog).
Zonotope aux_linprog(const std::vector<Zonotope> &Zs) {
    const DType dtype = Zs.front().center().dtype();
    const Zonotope enclosure = aux_iterative(Zs).compact("zeros", machine_eps(dtype));
    const Tensor &G0 = enclosure.generators();
    const Tensor length = G0.square().sum(-2, true).sqrt();
    const Tensor G = G0.multiply(length.ones_like().divide(length)); // unit generators
    const int64_t n = enclosure.dim(), ng = G.shape().back();
    const auto [A, unused] = priv_zonotope_halfspaces({enclosure.center().zeros_like(), G0});

    // the support value of every set along every normal, and the largest of them
    const auto support = [&](const Zonotope &Z) {
        return A.matmul(Z.center()) + A.matmul(Z.generators()).abs().sum(-1, true);
    };
    Tensor d = support(Zs.front());
    for (std::size_t j = 1; j < Zs.size(); ++j) d = d.maximum(support(Zs[j]));

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
    const Tensor scal = Tensor::like(G, std::vector<double>(r.x.begin() + n, r.x.end()), {1, ng});
    return {c, G.multiply(scal)};
}

/// The box around all sets in the frame of the principal axes of their generators (CORA's
/// aux_unionParallelotope).
Zonotope aux_parallelotope(const std::vector<Zonotope> &Zs) {
    std::vector<Tensor> columns;
    for (const Zonotope &Z : Zs) {
        columns.push_back(Z.generators());
        columns.push_back(Z.generators().negative());
    }
    const Tensor V = Tensor::concat(columns, -1);
    const Tensor covariance = V.matmul(V.matrix_transpose()) * (1.0 / double(V.shape().back() - 1));
    const Tensor U = covariance.eigh().second.flip(-1); // the principal axes, the largest first
    const Tensor Ut = U.matrix_transpose();
    Interval hull = Zs.front().linear_map(Ut).interval();
    for (std::size_t i = 1; i < Zs.size(); ++i) hull = hull.unite(Zs[i].linear_map(Ut).interval());
    return hull.zonotope().linear_map(U);
}

/// The union of Z, the sets Zs and the box (if given) by the method; every overload ends here, so
/// the checks run once and under the one entry.
Zonotope aux_unite(const Zonotope &Z, std::vector<Zonotope> Zs, const Interval *box,
                   const std::string &method, std::optional<int> order) {
    detail::Entry entry;
    // the method and its options first, then the operands
    CORA_CHECK(method == "linprog" || method == "tedrake" || method == "iterative" ||
                   method == "althoff" || method == "parallelotope",
               Error::wrong_value, [&] {
                   return "unite: the method must be linprog, tedrake, iterative, althoff or "
                          "parallelotope, got '" + method + "'";
               });
    CORA_CHECK(method != "tedrake" && method != "althoff", Error::not_supported, [&] {
        return "unite: the method '" + method + "' is not supported";
    });
    CORA_CHECK(!order, Error::not_supported, [] {
        return std::string("unite: reducing the order needs the reduction methods 'methC' and "
                           "'pca', which are not supported");
    });
    CORA_CHECK(Z.center().ndim() == 2, Error::not_supported, [] {
        return std::string("unite: it supports a single zonotope, not a batch");
    });
    if (box) {
        CORA_CHECK(box->is_bounded() && box->inf().shape().back() == 1, Error::wrong_value, [] {
            return std::string("unite: the box must be bounded and a vector");
        });
        Zs.push_back(box->zonotope());
    }
    // this set first, as CORA's cell array
    std::vector<Zonotope> all{Z};
    for (const Zonotope &Z2 : Zs) {
        CORA_CHECK(Z2.dim() == Z.dim() && Z2.center().ndim() == 2, Error::dimension_mismatch, [&] {
            return "unite: the sets must be single and of dimension " + std::to_string(Z.dim()) +
                   ", got " + std::to_string(Z2.dim());
        });
        all.push_back(Z2);
    }
    if (all.size() == 1) return Z;
    if (method == "iterative") return aux_iterative(all);
    if (method == "parallelotope") return aux_parallelotope(all);
    return aux_linprog(all);
}

/// The entry points below forward here, so the checks run once; the point and the box are
/// converted by aux_unite after the options are checked.
Zonotope aux_sets(const Zonotope &Z, const std::vector<Zonotope> &Zs, const std::string &method,
                  std::optional<int> order) {
    return aux_unite(Z, Zs, nullptr, method, order);
}

Zonotope aux_box(const Zonotope &Z, const Interval &I, const std::string &method,
                 std::optional<int> order) {
    return aux_unite(Z, {}, &I, method, order);
}

Zonotope aux_point(const Zonotope &Z, const Tensor &p, const std::string &method,
                   std::optional<int> order) {
    return aux_unite(Z, {priv_zonotope_of_point(p)}, nullptr, method, order);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The union with sets -----------------------------------------------------------------------------

Zonotope Zonotope::unite(const std::vector<Zonotope> &Zs, const std::string &method,
    std::optional<int> order) const { return aux_sets(*this, Zs, method, order); }

// The union with one set --------------------------------------------------------------------------

Zonotope Zonotope::unite(const Zonotope &Z2, const std::string &method,
    std::optional<int> order) const { return aux_sets(*this, {Z2}, method, order); }

Zonotope Zonotope::unite(const Interval &I, const std::string &method,
    std::optional<int> order) const { return aux_box(*this, I, method, order); }

Zonotope Zonotope::unite(const Tensor &p, const std::string &method,
    std::optional<int> order) const { return aux_point(*this, p, method, order); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
