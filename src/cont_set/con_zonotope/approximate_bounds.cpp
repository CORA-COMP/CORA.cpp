// approximate_bounds - bounds of a batch of constrained zonotopes with inequality constraints
//
// The sets are c + G x + [-dr, dr] with |x|_inf <= 1 and A x <= b (q >= m: the factors without
// generators are padded). "fourier-motzkin" tightens the factor bounds row by row, at most
// max_iter times, and the box follows from them; "exact" is a linear program per bound on the
// host. batch_union treats each union row as a batch member (else a loop), as CORA's options.nn.
//
// NaN and 0 for a member without a point. A union takes the least and the largest bound of its
// rows (CORA's loop over the union rows starts from -1 and 1, so it never tightens: docs/cora-
// differences.md). Any batch shape works, the result has it too.
//
// Syntax:   Bounds r = ConZonotope::approximate_bounds(c, G, A, b);
//           Bounds r = ConZonotope::approximate_bounds(c, G, A, b, dr, num_union_const, options);
// Inputs:   c (B, n, 1), G (B, n, m), A (B, p, q), b (B, p, 1) - the batch of the sets
//           dr (B, n, 1) - the interval error (default: 0)
//           num_union_const - the first rows of A are a union, the others an intersection
//           options - the method: "fourier-motzkin" (default) or "exact"
// Outputs:  r - the box (l, u) (B, n, 1) and the bounds (bl, bu) (B, q, 1) of the factors
// See also: interval, Koller, ICLR 2026, Prop. 3
//
// CORA: contSet/@conZonotope/approximateBoundsWithGPU.m
// CORA authors: Lukas Koller
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/linprog.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

using Bounds = ConZonotope::Bounds;

/// Whether some entry of a 0/1 mask is set.
bool aux_any(const Tensor &mask) { return mask.max().item() > 0; }

// Fourier Motzkin ---------------------------------------------------------------------------------

/// Bounds of the factors of {|x|_inf <= 1, A x <= b} (A (B, p, q), b (B, p, 1)) as rows (B, 1, q):
/// every row gives each factor the bound that the least value of the other terms leaves.
std::pair<Tensor, Tensor> aux_fourier_motzkin(const Tensor &A, const Tensor &b, int max_iter) {
    const int64_t B = A.shape()[0], q = A.shape()[2];
    Tensor bl = A.full_like({B, 1, q}, -1.0), bu = A.full_like({B, 1, q}, 1.0);
    const Tensor zero = A.zeros_like();
    const Tensor positive = A.greater(zero), negative = A.less(zero);
    // a row without a coefficient bounds nothing: divide by one, the quotient is not used
    const Tensor safe = Tensor::where(A.equal(zero), A.ones_like(), A);
    bool tighter = true;
    for (int iter = 1; tighter && iter <= max_iter; ++iter) {
        // the least value of each term a_ij x_j and of the others in its row
        const Tensor terms =
            A.multiply(Tensor::where(positive, bl, Tensor::where(negative, bu, zero)));
        const Tensor others = terms.sum(-1, true) - terms;
        const Tensor implied = (b - others).divide(safe).maximum(bl).minimum(bu);
        // a negative coefficient gives a lower bound, a positive one an upper bound
        const Tensor new_bl = Tensor::where(negative, implied, bl).max(-2, true);
        const Tensor new_bu = Tensor::where(positive, implied, bu).min(-2, true);
        const Tensor lower_moved = (bl + numerics::kBoundTighter).less(new_bl);
        const Tensor upper_moved = new_bu.less(bu - numerics::kBoundTighter);
        tighter = aux_any(lower_moved.maximum(upper_moved).multiply(new_bl.less_equal(new_bu)));
        bl = new_bl;
        bu = new_bu;
    }
    return {bl.matrix_transpose(), bu.matrix_transpose()};
}

// Linear programs ---------------------------------------------------------------------------------

/// The bounds by a linear program per bound, for every member of the batch.
Bounds aux_exact(const Tensor &c, const Tensor &G, const Tensor &A, const Tensor &b,
                 const Tensor &dr) {
    const int64_t B = c.shape()[0], n = c.shape()[1], q = A.shape()[2], p = A.shape()[1];
    const std::vector<double> host_c = c.data(), host_G = G.data(), host_A = A.data();
    const std::vector<double> host_b = b.data();
    std::vector<double> l(std::size_t(B * n), kNaN), u(std::size_t(B * n), kNaN),
        bl(std::size_t(B * q), kNaN), bu(std::size_t(B * q), kNaN);
    for (int64_t i = 0; i < B; ++i) {
        LinProg lp;
        lp.n = int(q);
        lp.Aineq.assign(host_A.begin() + i * p * q, host_A.begin() + (i + 1) * p * q);
        lp.bineq.assign(host_b.begin() + i * p, host_b.begin() + (i + 1) * p);
        lp.lb.assign(std::size_t(q), -1.0);
        lp.ub.assign(std::size_t(q), 1.0);
        const auto minimum = [&](const std::vector<double> &f, double &value) {
            lp.f = f;
            const LinProgResult r = linprog(lp);
            value = r.fval;
            return r.status == LinProgResult::Status::optimal;
        };
        // the box of the coordinates, then the bounds of the factors; the first failure ends it
        bool feasible = true;
        for (int64_t j = 0; j < n && feasible; ++j) {
            const auto row = host_G.begin() + (i * n + j) * q;
            std::vector<double> f(row, row + q);
            double lo = 0, hi = 0;
            // the minimum of the row, then of its negation (the maximum)
            feasible = minimum(f, lo);
            for (double &x : f) x = -x;
            feasible = feasible && minimum(f, hi);
            if (feasible) {
                l[i * n + j] = host_c[i * n + j] + lo;
                u[i * n + j] = host_c[i * n + j] - hi;
            }
        }
        // the same for the bounds of each factor
        for (int64_t j = 0; j < q && feasible; ++j) {
            std::vector<double> f(std::size_t(q), 0.0);
            f[j] = 1.0;
            double lo = 0, hi = 0;
            feasible = minimum(f, lo);
            f[j] = -1.0;
            feasible = feasible && minimum(f, hi);
            if (feasible) {
                bl[i * q + j] = lo;
                bu[i * q + j] = -hi;
            }
        }
    }
    const Tensor lt = Tensor::like(c, l, {B, n, 1}), ut = Tensor::like(c, u, {B, n, 1});
    return {lt - dr, ut + dr, Tensor::like(c, bl, {B, q, 1}), Tensor::like(c, bu, {B, q, 1})};
}

/// The bounds of one batch of sets by the method of the options.
Bounds aux_bound(const Tensor &c, const Tensor &G, const Tensor &dr, const Tensor &A,
                 const Tensor &b, const ConZonotope::BoundsOptions &options) {
    if (options.method == "exact") return aux_exact(c, G, A, b, dr);
    if (options.method != "fourier-motzkin")
        CORA_FAIL(Error::not_supported, "approximate_bounds: the method '" + options.method +
                                            "' is not supported; use 'fourier-motzkin' or 'exact'");
    const auto [bl, bu] = aux_fourier_motzkin(A, b, options.max_iter);
    // the box of the factors, mapped by G: center and radius
    const Tensor center = c + G.matmul((bu + bl) * 0.5);
    const Tensor radius = dr + G.abs().matmul((bu - bl) * 0.5);
    return {center - radius, center + radius, bl, bu};
}

// Unions ------------------------------------------------------------------------------------------

/// The smallest (largest) of the members of axis 0, which ignores NaN (an empty member).
Tensor aux_reduce(const Tensor &x, bool smallest) {
    const Tensor missing = x.isnan();
    const Tensor filler = x.full_like(x.shape(), smallest ? kInf : -kInf);
    const Tensor values = Tensor::where(missing, filler, x);
    const Tensor reduced = smallest ? values.min(0) : values.max(0);
    return Tensor::where(reduced.isinf().multiply(missing.min(0)),
                         reduced.full_like(reduced.shape(), kNaN), reduced);
}

/// The bounds of the union of the rows: the first `u` rows of A each with all the others, the
/// batch of the sets moved into the batch (batched) or looped over.
Bounds aux_union(const Tensor &c, const Tensor &G, const Tensor &dr, const Tensor &A,
                 const Tensor &b, int u, const ConZonotope::BoundsOptions &options) {
    const int64_t B = c.shape()[0], p = A.shape()[1];
    const Tensor rest_A = A.slice(1, u, p), rest_b = b.slice(1, u, p);
    // the bounds of every union row against the intersection rows, as (u, B, ., 1)
    Bounds s;
    if (options.batch_union) {
        // the union rows become members: member k * B + i is the member i with the union row k
        const Tensor union_A =
            A.slice(1, 0, u).permute_dims({1, 0, 2}).reshape({u * B, 1, A.shape()[2]});
        const Tensor union_b = b.slice(1, 0, u).permute_dims({1, 0, 2}).reshape({u * B, 1, 1});
        s = aux_bound(c.tile({u, 1, 1}), G.tile({u, 1, 1}), dr.tile({u, 1, 1}),
                      Tensor::concat({union_A, rest_A.tile({u, 1, 1})}, 1),
                      Tensor::concat({union_b, rest_b.tile({u, 1, 1})}, 1), options);
        const auto split = [&](const Tensor &x) { return x.reshape({u, B, x.shape()[1], 1}); };
        s = {split(s.l), split(s.u), split(s.bl), split(s.bu)};
    } else {
        std::vector<Tensor> l, u_, bl, bu;
        for (int k = 0; k < u; ++k) {
            const Bounds r = aux_bound(c, G, dr, Tensor::concat({A.slice(1, k, k + 1), rest_A}, 1),
                                       Tensor::concat({b.slice(1, k, k + 1), rest_b}, 1), options);
            l.push_back(r.l);
            u_.push_back(r.u);
            bl.push_back(r.bl);
            bu.push_back(r.bu);
        }
        s = {Tensor::stack(l, 0), Tensor::stack(u_, 0), Tensor::stack(bl, 0), Tensor::stack(bu, 0)};
    }
    // the four bounds are reduced
    return {aux_reduce(s.l, true), aux_reduce(s.u, false), aux_reduce(s.bl, true),
            aux_reduce(s.bu, false)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope::Bounds
ConZonotope::approximate_bounds(const Tensor &c, const Tensor &G, const Tensor &A, const Tensor &b,
                                const std::optional<Tensor> &dr_arg, int num_union_const,
                                const std::optional<BoundsOptions> &options_arg) {
    detail::Entry entry;
    const BoundsOptions options = options_arg.value_or(BoundsOptions{});
    const auto batch = [](const Tensor &t) {
        return std::vector<int64_t>(t.shape().begin(), t.shape().end() - 2);
    };
    CORA_CHECK(c.ndim() >= 2 && c.shape().back() == 1 && G.ndim() >= 2 && A.ndim() >= 2 &&
                   b.ndim() >= 2 && b.shape().back() == 1,
               Error::dimension_mismatch, [&] {
                   return "approximate_bounds: c (..., n, 1), G (..., n, m), A (..., p, q) and b "
                          "(..., p, 1) are needed, got the shapes " +
                          shape_text(c.shape()) + ", " + shape_text(G.shape()) + ", " +
                          shape_text(A.shape()) + " and " + shape_text(b.shape());
               });
    // the sets share one batch shape
    CORA_CHECK(batch(c) == batch(G) && batch(c) == batch(A) && batch(c) == batch(b) &&
                   (!dr_arg ||
                    (batch(*dr_arg) == batch(c) && is_column(*dr_arg, c.shape()[c.ndim() - 2]))),
               Error::dimension_mismatch, [] {
                   return std::string("approximate_bounds: the sets need one batch shape and dr a "
                                      "column of the dimension");
               });
    // the shapes of the parts fit each other
    CORA_CHECK(G.shape()[G.ndim() - 2] == c.shape()[c.ndim() - 2] &&
                   A.shape().back() >= G.shape().back() &&
                   b.shape()[b.ndim() - 2] == A.shape()[A.ndim() - 2],
               Error::dimension_mismatch, [&] {
                   return "approximate_bounds: G needs the rows of c, A at least the columns of G, "
                          "b the rows of A; got " +
                          shape_text(G.shape()) + ", " + shape_text(A.shape()) + " and " +
                          shape_text(b.shape());
               });
    const int64_t p = A.shape()[A.ndim() - 2];
    CORA_CHECK(p == 0 || (num_union_const >= 1 && num_union_const <= p), Error::wrong_value, [&] {
        return "approximate_bounds: num_union_const must be in [1, " + std::to_string(p) +
               "], got " + std::to_string(num_union_const);
    });
    // the method and the number of passes
    CORA_CHECK(options.method == "fourier-motzkin" || options.method == "exact" ||
                   options.method == "dual-iter",
               Error::wrong_field_value, [&] {
                   return "approximate_bounds: the method '" + options.method +
                          "' is unknown; use 'fourier-motzkin', 'dual-iter' or 'exact'";
               });
    CORA_CHECK(options.method != "dual-iter", Error::not_supported, [] {
        return std::string("approximate_bounds: the method 'dual-iter' (a quasi-Newton iteration "
                           "on the dual) is not ported");
    });
    CORA_CHECK(options.max_iter >= 0, Error::wrong_value,
               [] { return std::string("approximate_bounds: max_iter must be nonnegative"); });

    // the batch as one axis
    const std::vector<int64_t> lead = batch(c);
    int64_t B = 1;
    for (const int64_t d : lead) B *= d;
    const int64_t n = c.shape()[c.ndim() - 2], m = G.shape().back(), q = A.shape().back();
    const auto shaped = [&](const Tensor &t, int64_t rows, int64_t cols) {
        return t.reshape({B, rows, cols});
    };
    const Tensor c3 = shaped(c, n, 1), G3 = shaped(G, n, m);
    const Tensor dr3 = dr_arg ? shaped(*dr_arg, n, 1) : c3.zeros_like();
    Bounds r;
    if (p == 0) {
        // no constraint: the zonotope's box and the unit box
        const Tensor radius = dr3 + G3.abs_sum();
        r = {c3 - radius, c3 + radius, c3.full_like({B, m, 1}, -1.0), c3.full_like({B, m, 1}, 1.0)};
    } else {
        const Tensor padded = Tensor::concat({G3, priv_zeros(G3, n, q - m)}, -1);
        r = aux_union(c3, padded, dr3, shaped(A, p, q), shaped(b, p, 1), num_union_const, options);
        // a member without a point: the factor bounds cross, or the box is not a number
        const Tensor crossed = r.bl.greater(r.bu).any(-2, true);
        const Tensor missing = r.l.isnan().maximum(r.u.isnan()).any(-2, true).maximum(crossed);
        const Tensor nan = r.l.full_like(r.l.shape(), kNaN);
        r = {Tensor::where(missing, nan, r.l), Tensor::where(missing, nan, r.u),
             Tensor::where(missing, r.bl.zeros_like(), r.bl),
             Tensor::where(missing, r.bu.zeros_like(), r.bu)};
    }
    // back to the batch shape of the sets
    const auto back = [&](const Tensor &t) {
        Shape shape = lead;
        shape.push_back(t.shape()[1]);
        shape.push_back(1);
        return t.reshape(shape);
    };
    return {back(r.l), back(r.u), back(r.bl), back(r.bu)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
