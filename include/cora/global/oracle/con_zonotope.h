// con_zonotope - lean::ConZonotope, a constrained zonotope of exact values, computed by CORALean
//
// Syntax:     lean::ConZonotope cz(cora_cz);   lean::ConZonotope r = cz.intersect(cz2); r.gather();
// Operations: one method per `conZonotope.*` op of the oracle, named as CORA.cpp names CORA's
// (`and`
//             is intersect, `cartProd` cart_prod, `conIntersect` con_intersect, `cubMap` cub_map)
// Parts:      the oracle's set is {c + G x + E | |x + E_x|_inf <= 1, Ae x = be + be_error} with
//             the error boxes of the sound float operations; at dtype nearest they are empty
//             (nominal()), else interval() is the box the oracle proves to enclose the set
// Crossing:   gather() returns the nominal part as a cora::ConZonotope, exactly or throws
// See also:   lean/zonotope.h, lean/request.h, cora/cont_set/con_zonotope/con_zonotope.h
//
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/tensor.h"
#include "cora/global/oracle/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class ConZonotope {
  public:
    /// An exact constrained zonotope (empty error boxes) in the current dtype.
    ConZonotope(const cora::ConZonotope &cz);

    /// An exact set from a cut: the rows C x <= d of the cut are the inequalities of the oracle's
    /// polytope on the factors, the equalities A x = b its equalities.
    ConZonotope(const cora::ConZonotope::Cut &cut);

    /// The dtype of the values.
    const std::string &dtype() const noexcept { return dtype_; }

    // Crossing ------------------------------------------------------------------------------------

    /// The nominal part c, G and the equalities (Ae, be) as the oracle returns them; throws if a
    /// value is not a double. The inequality rows and the error boxes are not looked at.
    cora::ConZonotope nominal() const;

    /// The inequality rows A x <= b of the oracle's set, as a pair (A, b): none for an input.
    std::pair<cora::Tensor, cora::Tensor> inequalities() const;

    /// nominal() when there are no inequality rows and every error box is empty, else it throws.
    cora::ConZonotope gather() const;

    /// Whether every error box is empty and there are no inequality rows.
    bool exact() const;

    /// The box the oracle proves to enclose the set (`conZonotope.interval`).
    Interval interval() const;

    /// The box of c + G J with J the factor bounds after k Fourier-Motzkin passes on the
    /// constraint (`conZonotope.intervalTight`).
    Interval interval_tight(int64_t k) const;

    /// A box from dual certificates and the multipliers it was made from
    /// (`conZonotope.intervalCert`).
    struct Cert {
        Interval I;
        cora::ConZonotope::Certificate weights;
    };

    /// The box at the multipliers w, bit for bit; the multipliers are the ones sent.
    Cert interval_cert(const cora::ConZonotope::Certificate &w) const;

    /// The box at the multipliers of the oracle's own exact simplex, which it returns.
    Cert interval_cert() const;

    // Set operations ---------------------------------------------------------------------------

    ConZonotope intersect(const ConZonotope &cz2) const;
    ConZonotope convex_hull(const ConZonotope &cz2) const;
    ConZonotope cart_prod(const ConZonotope &cz2) const;
    ConZonotope con_intersect(const ConZonotope &cz2, const Tensor &R) const;

    /// The enclosure of this set and M * this + cz2 (the oracle's `enclose`).
    ConZonotope enclose(const Tensor &M, const ConZonotope &cz2) const;

    /// The zonotope-like strips |C x - y| <= phi at the gain Lambda.
    ConZonotope intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                const Tensor &Lambda) const;

    /// The cubic map T_ij x x x; method "zeros" of compact.
    ConZonotope cub_map(const std::vector<std::vector<Tensor>> &T) const;
    ConZonotope compact(const std::string &method = "zeros", double tol = 0) const;

    static ConZonotope empty(int64_t n);
    static ConZonotope origin(int64_t n);

    /// The sum with a constrained zonotope and the image M * this.
    ConZonotope plus(const ConZonotope &cz2) const;
    ConZonotope mtimes(const Tensor &M) const;

    /// The projection onto dims and the lift to N dimensions with state i at position dims[i].
    ConZonotope project(const std::vector<int64_t> &dims) const;
    ConZonotope lift(int64_t N, const std::vector<int64_t> &dims) const;

    /// Girard's reduction to the order, and the quadratic map with the matrices Q.
    ConZonotope reduce(int order) const;
    ConZonotope quad_map(const std::vector<Tensor> &Q) const;

    /// The halves of the set cut at the center of the box in coordinate dim.
    std::pair<ConZonotope, ConZonotope> split(int64_t dim) const;

    /// Structural equality; a proof of emptiness (a true proves it); whether the sets intersect,
    /// proved only when disjoint (the value false with proved set).
    bool isequal(const ConZonotope &cz2, double tol = 0) const;
    bool representsa_empty() const;
    Certified is_intersecting(const ConZonotope &cz2, double tol = 0) const;

    /// A zonotope or a box as a constrained zonotope without constraints.
    static ConZonotope from_zonotope(const Zonotope &Z);
    static ConZonotope from_interval(const Interval &I);

    bool is_bounded() const;

    /// An upper bound of the support function along dir (a linear program's bound).
    double support_function(const Tensor &dir) const;

    /// Whether the oracle proves that the point (n, 1) is in the set, and that the set has an
    /// interior point: the certificate is a right inverse of G; a set with an equality row is
    /// never proved (false says only that no proof was found).
    bool contains(const Tensor &point, double tol = 0) const;
    bool is_full_dim(double tol = 0) const;

    Json to_json() const;
    static ConZonotope from_json(const std::string &dtype, const Json &j);

  private:
    ConZonotope(std::string dtype, Json json) : dtype_(std::move(dtype)), json_(std::move(json)) {}

    std::string dtype_;
    Json json_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
