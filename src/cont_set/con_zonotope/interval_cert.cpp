// interval_cert - the box of a constrained zonotope from dual certificates, in floating point
//
// CORA's interval solves max / min e_i'(c + G x) over the rows and the unit box, two linear
// programs per coordinate. Here the programs only propose: their dual solution (y for the rows
// C x <= d, zu and zl for the two halves of A x = b) goes into weak duality, checked in the order
// of CORALean's `ConZonotope.boundDotWeights`: weights clamped at 0, the residual of each factor's
// column paid by the unit box, the sum of the right-hand sides weighted. The bound is outer for
// every multiplier, so a poor or wrong solution loosens the box and never makes it unsound; all
// zero is the box of the zonotope. The result is CORALean's `ConZonotope.intervalCertWith` at the
// same multipliers, bit for bit.
//
// Syntax:   Interval I = cut.interval_cert();   Interval I = cut.interval_cert(w);
//           Interval I = cut.interval_cert_tight(passes);   Interval J = cz.interval_cert();
// Inputs:   w - the multipliers of both ends of every coordinate (ConZonotope::Certificate)
//           passes - the Fourier-Motzkin passes of the box that is intersected
// Outputs:  I - a box that holds the set; empty if a program has no solution
// See also: interval, interval_tight, cut
//
// CORA.cpp only: CORALean's float definition checks the dual solutions of CORA's programs
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/linprog.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Vec = std::vector<double>;

// The cut on the host -----------------------------------------------------------------------------

/// The cut on the host: c (n), G (n, h), the rows C (m, h) and d, the equalities A (p, h) and b.
struct Host {
    int64_t n = 0, h = 0, m = 0, p = 0;
    Vec c, G, C, d, A, b;
};

Host aux_host_cut(const ConZonotope::Cut &K) {
    // the sizes, then the values row-major
    Host H;
    H.n = K.c.shape()[0];
    H.h = K.G.shape()[1];
    H.m = K.C.shape()[0];
    H.p = K.A.shape()[0];
    H.c = K.c.data();
    H.G = priv_host_rows(K.G);
    H.C = priv_host_rows(K.C);
    H.d = priv_host_rows(K.d);
    H.A = priv_host_rows(K.A);
    H.b = priv_host_rows(K.b);
    return H;
}

// The bound ---------------------------------------------------------------------------------------

/// The dot product of two strided vectors, summed from the last term on (CORALean's `dotUp`
/// and `dotDown`, which agree at nearest): a_0 b_0 + (a_1 b_1 + (... + 0)).
double aux_dot(const double *a, std::ptrdiff_t sa, const double *b, std::ptrdiff_t sb, int64_t k) {
    double acc = 0.0;
    for (int64_t l = k; l-- > 0;) acc = a[l * sa] * b[l * sb] + acc;
    return acc;
}

/// A weight made admissible: `maximum 0 w`.
double aux_clamp(double w) { return priv_float_max(0.0, w); }

/// The clamped weights of one end.
ConZonotope::CertWeights aux_clamped(const ConZonotope::CertWeights &w) {
    ConZonotope::CertWeights out;
    for (const double x : w.y) out.y.push_back(aux_clamp(x));
    for (const double x : w.zu) out.zu.push_back(aux_clamp(x));
    for (const double x : w.zl) out.zl.push_back(aux_clamp(x));
    return out;
}

/// The bound along the direction a at the multipliers w, `boundDotWeights`: the weighted
/// right-hand sides and the residuals R_j of the columns, the error box (zero) added upward.
double aux_bound(const Host &H, const Vec &a, const ConZonotope::CertWeights &weights) {
    const ConZonotope::CertWeights w = aux_clamped(weights);
    const std::ptrdiff_t h = H.h;
    // the residual of each factor (its up and down ends agree at nearest): R_j = max(r, -r)
    Vec R(std::size_t(H.h));
    for (int64_t j = 0; j < H.h; ++j) {
        const double ga = aux_dot(a.data(), 1, H.G.data() + j, h, H.n);
        const double cy = aux_dot(w.y.data(), 1, H.C.data() + j, h, H.m);
        const double azu = aux_dot(w.zu.data(), 1, H.A.data() + j, h, H.p);
        const double azl = aux_dot(w.zl.data(), 1, H.A.data() + j, h, H.p);
        const double r = ga + (-cy + (-azu + azl));
        R[std::size_t(j)] = priv_float_max(r, -r);
    }
    // the right-hand sides, each with its slack (zero) rounded up
    Vec bs(std::size_t(H.m)), beu(std::size_t(H.p)), bel(std::size_t(H.p));
    for (int64_t k = 0; k < H.m; ++k) bs[std::size_t(k)] = H.d[std::size_t(k)] + 0.0;
    for (int64_t k = 0; k < H.p; ++k) {
        beu[std::size_t(k)] = H.b[std::size_t(k)] + 0.0;
        bel[std::size_t(k)] = -H.b[std::size_t(k)] + 0.0;
    }
    // the residuals summed from the last on (CORALean's `sumUp`)
    double sum = 0.0;
    for (std::size_t j = R.size(); j-- > 0;) sum = R[j] + sum;
    const double val = aux_dot(a.data(), 1, H.c.data(), 1, H.n) +
                       (aux_dot(w.y.data(), 1, bs.data(), 1, H.m) +
                        (aux_dot(w.zu.data(), 1, beu.data(), 1, H.p) +
                         (aux_dot(w.zl.data(), 1, bel.data(), 1, H.p) + sum)));
    // the error box's reach, a zero sum
    return val + 0.0;
}

/// The direction s e_i.
Vec aux_axis(int64_t n, int64_t i, double s) {
    Vec a(std::size_t(n), 0.0);
    a[std::size_t(i)] = s;
    return a;
}

/// The box at the multipliers: the upper end i bounds e_i'x, the lower end i is the negated
/// bound of -e_i'x.
Interval aux_box(const Host &H, const ConZonotope::Certificate &w, const Tensor &like) {
    Vec inf(std::size_t(H.n)), sup(std::size_t(H.n));
    for (int64_t i = 0; i < H.n; ++i) {
        const std::size_t ii = std::size_t(i);
        sup[ii] = aux_bound(H, aux_axis(H.n, i, 1.0), w.up[ii]);
        inf[ii] = -aux_bound(H, aux_axis(H.n, i, -1.0), w.lo[ii]);
    }
    return {Tensor::like(like, inf, {H.n, 1}), Tensor::like(like, sup, {H.n, 1})};
}

/// The set as a cut without rows.
ConZonotope::Cut aux_cut(const ConZonotope &cz) {
    return {cz.c(),
            cz.generators(),
            cz.A(),
            cz.b(),
            priv_zeros(cz.c(), 0, cz.generators().shape().back()),
            priv_zeros(cz.c(), 0, 1)};
}

// The programs ------------------------------------------------------------------------------------

/// The multipliers of max s e_i'(c + G x) over the rows and the unit box, from the dual solution
/// of min -s G_i'x; none if the program has no solution.
std::optional<ConZonotope::CertWeights> aux_dual(const Host &H, int64_t i, double s) {
    ConZonotope::CertWeights w;
    w.y.assign(std::size_t(H.m), 0.0);
    w.zu.assign(std::size_t(H.p), 0.0);
    w.zl.assign(std::size_t(H.p), 0.0);
    // without factors there is no program: the weights stay zero
    if (H.h == 0) return w;
    // the program min -s G_i'x over the rows, the equalities and the unit box
    LinProg lp;
    lp.n = int(H.h);
    for (int64_t j = 0; j < H.h; ++j) lp.f.push_back(-s * H.G[std::size_t(i * H.h + j)]);
    lp.Aineq = H.C;
    lp.bineq = H.d;
    lp.Aeq = H.A;
    lp.beq = H.b;
    lp.lb.assign(std::size_t(H.h), -1.0);
    lp.ub.assign(std::size_t(H.h), 1.0);
    const LinProgResult r = linprog(lp);
    if (r.status != LinProgResult::Status::optimal) return std::nullopt;
    // the equality multiplier is free, split in its two halves; a non-number stays 0 (looser)
    for (std::size_t k = 0; k < r.dual_ineq.size(); ++k)
        if (std::isfinite(r.dual_ineq[k])) w.y[k] = r.dual_ineq[k];
    for (std::size_t k = 0; k < r.dual_eq.size(); ++k) {
        if (!std::isfinite(r.dual_eq[k])) continue;
        w.zu[k] = std::max(r.dual_eq[k], 0.0);
        w.zl[k] = std::max(-r.dual_eq[k], 0.0);
    }
    return w;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The multipliers and the boxes -------------------------------------------------------------------

std::optional<ConZonotope::Certificate> ConZonotope::Cut::certificate() const {
    detail::Entry entry;
    CORA_CHECK(c.ndim() == 2, Error::not_supported,
               [] { return std::string("certificate: a single set, not a batch"); });
    const Host H = aux_host_cut(*this);
    // both ends of every coordinate, from a program each
    Certificate w;
    for (int64_t i = 0; i < H.n; ++i) {
        auto up = aux_dual(H, i, 1.0), lo = aux_dual(H, i, -1.0);
        if (!up || !lo) return std::nullopt;
        w.up.push_back(std::move(*up));
        w.lo.push_back(std::move(*lo));
    }
    return w;
}

Interval ConZonotope::Cut::interval_cert(const Certificate &w) const {
    detail::Entry entry;
    CORA_CHECK(c.ndim() == 2, Error::not_supported,
               [] { return std::string("interval_cert: a single set, not a batch"); });
    const Host H = aux_host_cut(*this);
    // a multiplier per end of each coordinate, of the sizes of the rows
    CORA_CHECK(int64_t(w.up.size()) == H.n && int64_t(w.lo.size()) == H.n,
               Error::dimension_mismatch, [&] {
                   return "interval_cert: " + std::to_string(H.n) + " coordinates need " +
                          std::to_string(H.n) + " multipliers per end, got " +
                          std::to_string(w.up.size()) + " and " + std::to_string(w.lo.size());
               });
    for (const auto *ends : {&w.up, &w.lo})
        for (const CertWeights &e : *ends)
            CORA_CHECK(int64_t(e.y.size()) == H.m && int64_t(e.zu.size()) == H.p &&
                           int64_t(e.zl.size()) == H.p,
                       Error::dimension_mismatch, [&] {
                           return "interval_cert: the multipliers need " + std::to_string(H.m) +
                                  " values for the rows and " + std::to_string(H.p) +
                                  " for each half of the equalities";
                       });
    // the bounds at the multipliers
    return aux_box(H, w, c);
}

Interval ConZonotope::Cut::interval_cert() const {
    detail::Entry entry;
    const std::optional<Certificate> w = certificate();
    return w ? interval_cert(*w) : Interval::empty(c.shape()[0]);
}

Interval ConZonotope::Cut::interval_cert_tight(int passes) const {
    detail::Entry entry;
    // the two boxes, empty if either is
    const Interval cert = interval_cert(), tight = interval_tight(passes);
    if (cert.is_empty() || tight.is_empty()) return Interval::empty(c.shape()[0]);
    // the larger lower end and the smaller upper end, the cert box first as in CORALean
    Vec inf = cert.inf().data(), sup = cert.sup().data();
    const Vec tinf = tight.inf().data(), tsup = tight.sup().data();
    for (std::size_t i = 0; i < inf.size(); ++i) {
        inf[i] = priv_float_max(inf[i], tinf[i]);
        sup[i] = priv_float_min(sup[i], tsup[i]);
        if (inf[i] > sup[i]) return Interval::empty(c.shape()[0]);
    }
    return {Tensor::like(c, inf, {c.shape()[0], 1}), Tensor::like(c, sup, {c.shape()[0], 1})};
}

Interval ConZonotope::interval_cert(const Certificate &w) const {
    return aux_cut(*this).interval_cert(w);
}

Interval ConZonotope::interval_cert() const { return aux_cut(*this).interval_cert(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
