// mink_diff - the Minkowski difference of a zonotope and a zonotope, a box or a point
//
// Syntax:   Zres = Z.mink_diff(S);   Zres = Z.mink_diff(S, "outer");
// Inputs:   S - the subtrahend: a zonotope, a box (enclosed by a zonotope, with a warning) or a
//           point;  method - the generators of Z are scaled by factors alpha: "approx" (default,
//           least squares on the facets), "inner" and "exact" (linear programs; exact for two
//           dimensions or when the generators of both are aligned), "outer", "outer:coarse";
//           "inner:conZonotope" needs ConZonotope, "inner:RaghuramanKoeln" and "outer:scaling"
//           are not supported. A minuend that is not full-dimensional is not supported
// Outputs:  Zres - the center c - c2, the generators of Z scaled; "inner" and "exact" give a set
//           with Zres + S in Z, "outer" one that contains the difference. Where CORA returns the
//           empty set, CORA.cpp raises empty_set. Not a batch
// See also: plus, contains
//
// CORA: contSet/@zonotope/minkDiff.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Tobias Ladner
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   06-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/linprog.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <limits>
#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

[[noreturn]] void aux_empty(const std::string &why) {
    CORA_FAIL(Error::empty_set, "mink_diff: " + why + " and CORA.cpp has no empty set yet");
}

/// Whether the generators of both are the same up to a common scaling (CORA's aux_areAligned):
/// the same shape, and equal columns after dividing by the longest.
bool aux_are_aligned(const Tensor &Gmin, const Tensor &Gsub) {
    if (Gmin.shape() != Gsub.shape()) return false;
    const double norm_min = Gmin.vector_norm(Tensor::Norm::two, -2).max().item(),
                 norm_sub = Gsub.vector_norm(Tensor::Norm::two, -2).max().item();
    if (within_tol(norm_min, 0) && within_tol(norm_sub, 0)) return true;
    if (within_tol(norm_min, 0) || within_tol(norm_sub, 0)) return false;
    return (Gmin / norm_min - Gsub / norm_sub).abs().max().item() <= machine_eps(Gmin.dtype());
}

// The programs ------------------------------------------------------------------------------------

/// A matrix on the host, row-major.
struct HostMatrix {
    int64_t rows = 0, cols = 0;
    std::vector<double> data;
};

HostMatrix aux_host(const Tensor &T) { return {T.shape()[0], T.shape()[1], T.data()}; }

/// The program min f'x with A x <= b (rows of A given as `rows`) and free x; the minimizer, or
/// nothing if there is none.
std::optional<std::vector<double>> aux_solve(const std::vector<double> &f, const HostMatrix &A,
                                             const std::vector<double> &b, double *fval = nullptr) {
    LinProg lp;
    lp.n = int(A.cols);
    lp.f = f;
    lp.Aineq = A.data;
    lp.bineq = b;
    const LinProgResult r = linprog(lp);
    if (r.status != LinProgResult::Status::optimal) return std::nullopt;
    if (fval) *fval = r.fval;
    return r.x;
}

/// The factors alpha, a row (1, m), of the minuend's generators that the method picks. The facets
/// are C (k, n) and their offsets K_new (2k, 1) with the subtrahend taken off; the slack of the
/// first half d of a set of offsets is d - C c + C c2 (CORA's delta_d).
Tensor aux_scalings(const std::string &method, const Zonotope &minuend, const Zonotope &subtrahend,
                    const Tensor &C, const Tensor &K_new) {
    const Tensor &Gm = minuend.generators();
    const int64_t k = C.shape()[0], m = Gm.shape().back(), n = C.shape()[1];
    const Tensor A_abs = C.matmul(Gm).abs();
    const HostMatrix A = aux_host(A_abs);
    const auto slack = [&](const Tensor &d) {
        return d.slice(0, 0, k) - C.matmul(minuend.center()) + C.matmul(subtrahend.center());
    };
    if (method == "approx") return A_abs.pinv().matmul(slack(K_new)).matrix_transpose();
    const auto row = [&](const std::vector<double> &alpha) {
        return Tensor::like(Gm, alpha, {1, m});
    };

    std::vector<double> f = Gm.vector_norm(Tensor::Norm::two, -2).data();
    // the rows of [sign A_abs; -I] and the vector [sign delta; 0]
    const auto stacked = [&](double sign, const std::vector<double> &delta) {
        HostMatrix S{k + m, m, std::vector<double>((k + m) * m, 0.0)};
        std::vector<double> b(k + m, 0.0);
        for (int64_t i = 0; i < k; ++i) {
            for (int64_t j = 0; j < m; ++j) S.data[i * m + j] = sign * A.data[i * m + j];
            b[i] = sign * delta[i];
        }
        for (int64_t j = 0; j < m; ++j) S.data[(k + j) * m + j] = -1.0;
        return std::make_pair(S, b);
    };
    if (method == "inner" || method == "exact") {
        for (double &x : f) x = -x;
        const auto [S, b] = stacked(1.0, slack(K_new).data());
        if (const auto alpha = aux_solve(f, S, b)) return row(*alpha);
        aux_empty("the inner approximation does not exist");
    }
    // "outer": each offset tightened to the maximum of its row over the polytope; "outer:coarse"
    // keeps them
    Tensor offsets = K_new;
    if (method == "outer") {
        const HostMatrix H = aux_host(Tensor::concat({C, C.negative()}, 0));
        const std::vector<double> K = K_new.data();
        std::vector<double> tight(2 * k);
        for (int64_t i = 0; i < 2 * k; ++i) {
            std::vector<double> objective(n);
            for (int64_t j = 0; j < n; ++j) objective[j] = -H.data[i * n + j];
            double fval = 0;
            if (!aux_solve(objective, H, K, &fval)) aux_empty("the difference is empty");
            tight[i] = -fval;
        }
        offsets = Tensor::like(K_new, tight, {2 * k, 1});
    }
    const auto [S, b] = stacked(-1.0, slack(offsets).data());
    if (const auto alpha = aux_solve(f, S, b)) return row(*alpha);
    aux_empty("the outer approximation has no solution");
}

// The algorithms ----------------------------------------------------------------------------------

/// CORA's aux_minkDiffZono: the generators of the minuend scaled so that the difference fits
/// the facets.
Zonotope aux_mink_diff_zono(const Zonotope &minuend, const Zonotope &subtrahend,
                            const std::string &method) {
    const Zonotope compacted = minuend.compact();
    const Tensor C = priv_zonotope_facet_normals(compacted);
    const Tensor &Gm = minuend.generators(), &cm = minuend.center();
    const Tensor &Gs = subtrahend.generators(), &cs = subtrahend.center();
    // the offsets of the facets of the minuend, and what the subtrahend takes off them
    const Tensor Cc = C.matmul(cm), width = C.matmul(compacted.generators()).abs().sum(-1, true);
    const Tensor K = Tensor::concat({Cc + width, Cc.negative() + width}, 0);
    const Tensor H = Tensor::concat({C, C.negative()}, 0);
    Tensor delta_K = H.matmul(cs);
    for (int64_t i = 0; i < Gs.shape().back(); ++i)
        delta_K = delta_K + H.matmul(Gs.slice(-1, i, i + 1)).abs();

    const Tensor alpha = aux_scalings(method, minuend, subtrahend, C, K - delta_K);
    const Tensor scaled = Gm.multiply(alpha);
    // CORA drops the generators that came out exactly zero
    const Tensor values = scaled.detach();
    const Tensor kept = values.not_equal(values.zeros_like()).any(-2);
    return {cm - cs, priv_select_generators(scaled, kept.nonzero())};
}

// The options -------------------------------------------------------------------------------------

/// Why the method is refused, if it is: CORA's list of methods.
Refusal aux_refusal(const std::string &method) {
    for (const char *known : {"exact", "inner", "outer", "outer:coarse", "outer:scaling", "approx",
                              "inner:conZonotope", "inner:RaghuramanKoeln"})
        if (method == known) return std::nullopt;
    return std::pair{Error::wrong_value,
                     "the method must be one of exact, inner, outer, outer:coarse, "
                     "outer:scaling, approx, inner:conZonotope, inner:RaghuramanKoeln, got '" +
                         method + "'"};
}

/// CORA's minkDiff of two zonotopes, after the checks.
Zonotope aux_mink_diff(const Zonotope &minuend, const Zonotope &subtrahend,
                       const std::string &method) {
    if (subtrahend.generators().shape().back() == 0) return minuend - subtrahend.center();
    if (!minuend.is_full_dim()) {
        if (subtrahend.is_full_dim())
            aux_empty("the difference of a flat set and a full-dimensional one is empty");
        CORA_FAIL(Error::not_supported,
                  "mink_diff: a minuend that is not full-dimensional is not supported");
    }
    // two dimensions always take the exact algorithm
    const std::string used = minuend.dim() == 2 ? "exact" : method;
    if (used == "inner:conZonotope")
        CORA_FAIL(Error::not_supported,
                  "mink_diff: the method 'inner:conZonotope' needs ConZonotope");
    if (used == "inner:RaghuramanKoeln" || used == "outer:scaling")
        CORA_FAIL(Error::not_supported, "mink_diff: the method '" + used + "' is not supported");
    if (used != "exact") return aux_mink_diff_zono(minuend, subtrahend, used);
    if (aux_are_aligned(minuend.generators(), subtrahend.generators()))
        return {minuend.center() - subtrahend.center(),
                minuend.generators() - subtrahend.generators()};
    if (minuend.dim() != 2)
        CORA_FAIL(Error::wrong_value, "mink_diff: no exact algorithm found: the sets have to be "
                                      "2-dimensional or aligned");
    return aux_mink_diff_zono(minuend, subtrahend, "exact");
}

/// The subtrahend as a zonotope, warning when a box has to be enclosed.
Zonotope aux_subtrahend_of_box(const Interval &I) {
    warn("CORA:contSet", "mink_diff: the subtrahend is not a zonotope; enclosing it with one");
    return I.zonotope();
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::mink_diff(const Zonotope &subtrahend, const std::string &method) const {
    detail::Entry entry;
    // the dimension, the method, then the batch
    CORA_CHECK(subtrahend.dim() == dim(), Error::dimension_mismatch, [&] {
        return "mink_diff: the zonotopes differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(subtrahend.dim());
    });
    const Refusal refusal = aux_refusal(method);
    CORA_CHECK(!refusal, refusal->first, [&] { return "mink_diff: " + refusal->second; });
    CORA_CHECK(c_.ndim() == 2 && subtrahend.center().ndim() == 2, Error::not_supported, [] {
        return std::string("mink_diff: facets of one zonotope at a time; it supports single "
                           "zonotopes, not a batch");
    });
    return aux_mink_diff(*this, subtrahend, method);
}

Zonotope Zonotope::mink_diff(const Interval &subtrahend, const std::string &method) const {
    detail::Entry entry;
    // a box is enclosed by a zonotope
    CORA_CHECK(subtrahend.dim() == dim() && subtrahend.inf().shape().back() == 1 &&
                   subtrahend.inf().ndim() == 2,
               Error::dimension_mismatch, [&] {
                   return "mink_diff: the interval must be a single box of the dimension " +
                          std::to_string(dim()) + " of the zonotope";
               });
    const Refusal refusal = aux_refusal(method);
    CORA_CHECK(!refusal, refusal->first, [&] { return "mink_diff: " + refusal->second; });
    CORA_CHECK(c_.ndim() == 2, Error::not_supported, [] {
        return std::string("mink_diff: facets of one zonotope at a time; it supports a single "
                           "zonotope, not a batch");
    });
    CORA_CHECK(!subtrahend.is_empty(), Error::empty_set,
               [] { return std::string("mink_diff: the subtrahend is the empty set"); });
    return aux_mink_diff(*this, aux_subtrahend_of_box(subtrahend), method);
}

Zonotope Zonotope::mink_diff(const Tensor &p, const std::string &method) const {
    detail::Entry entry;
    // a point shifts the center whatever the method
    CORA_CHECK(p.ndim() == 2 && p.shape()[0] == dim() && p.shape()[1] == 1,
               Error::dimension_mismatch, [&] {
                   return "mink_diff: the point must be a column (" + std::to_string(dim()) +
                          ", 1), got the shape " + shape_text(p.shape());
               });
    const Refusal refusal = aux_refusal(method);
    CORA_CHECK(!refusal, refusal->first, [&] { return "mink_diff: " + refusal->second; });
    return *this - p;
}

Set Zonotope::minkowski_difference_impl(const ContSet &S2,
                                        const std::optional<std::string> &type) const {
    const std::string method = type.value_or("approx");
    if (S2.class_id() == ClassId::zonotope)
        return mink_diff(static_cast<const Zonotope &>(S2), method);
    if (S2.class_id() == ClassId::interval)
        return mink_diff(static_cast<const Interval &>(S2), method);
    CORA_FAIL(Error::noops,
              "minkowski_difference is not implemented for Zonotope and " + S2.class_name());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
