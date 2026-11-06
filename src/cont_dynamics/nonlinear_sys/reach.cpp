// reach - the reachable sets of x' = f(x), as CORA's nonlinearSys.reach with algorithm "lin"
//
// Every step abstracts f around the center xs of the current set X_k by the affine system
//     x' = f(xs) + A (x - xs) + w,   w in L,
// where A is the Jacobian at xs and L the Lagrange remainder over a box that encloses all states
// of the step. The affine system is solved with a set of inputs U = f(xs) + L:
//     X_{k+1} = xs + e^{A dt} (X_k - xs) + sum_i A^i dt^{i+1}/(i+1)! U + (Taylor remainder),
// and the sets in between with the enclosure of linear_sys.reach plus the analogous input terms.
//
// Syntax:   R = sys.reach(problem, alg);
//           R = sys.reach({.initial_set = X0, .t_final = 1}, alg::Linearization{.time_step =
//           0.01});
// Inputs:   problem - initial_set (one zonotope), t_final; no input_set
//           alg - alg::Linearization: time_step, taylor_terms (order of the Taylor series of the
//                 linearized system), zonotope_order (a set keeps at most that times n generators)
// Outputs:  R - time_interval[k] encloses the states over [k*time_step, (k+1)*time_step];
//           time_point[k] is the set at k*time_step
// See also: simulate, LinearSys::reach
//
// CORA: contDynamics/@contDynamics/reach.m
// CORA: contDynamics/@contDynamics/linReach.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/global/tensor/tolerance.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The bounds of a box, each a column (n, 1).
using Box = Range;

/// The entries of a column (n, 1) as n tensors of shape (1, 1).
std::vector<Tensor> aux_entries(const Tensor &column) {
    const Tensor row = column.matrix_transpose();
    std::vector<Tensor> out;
    out.reserve(column.shape()[0]);
    for (int64_t i = 0; i < column.shape()[0]; ++i) out.push_back(row.slice(-1, i, i + 1));
    return out;
}

// The box of a set --------------------------------------------------------------------------------
/// The interval hull of a single zonotope.
Box aux_box(const Zonotope &Z) {
    const Interval I = Z.interval();
    return {I.inf(), I.sup()};
}

/// The box hull of two boxes, made 10% (and a little) larger.
Box aux_inflated_hull(const Box &a, const Box &b) {
    const Tensor lo = a.lo.minimum(b.lo), hi = a.hi.maximum(b.hi);
    const Tensor margin = (hi - lo) * 0.1 + numerics::kBoxMargin;
    return {lo - margin, hi + margin};
}

/// The box x0 + [0, dt] f(guess): where the states of a step end up if they stay in the guess.
Box aux_picard(const NonlinearSys &sys, const Box &x0, const Box &guess, double dt) {
    const Box f = sys.enclosure(guess);
    // [0, dt] * [a, b] is [min(0, dt a), max(0, dt b)] for a step dt > 0.
    return {x0.lo + (f.lo * dt).minimum(0.0), x0.hi + (f.hi * dt).maximum(0.0)};
}

/// Whether the box inner lies in outer (false for a NaN).
bool aux_inside(const Box &inner, const Box &outer) {
    const std::vector<double> il = inner.lo.data(), ih = inner.hi.data();
    const std::vector<double> ol = outer.lo.data(), oh = outer.hi.data();
    for (std::size_t i = 0; i < il.size(); ++i)
        if (!(il[i] >= ol[i] && ih[i] <= oh[i])) return false;
    return true;
}

/// A box that contains every state over the step [0, dt] from the box x0: a guess that the
/// Picard operator maps into itself (then no trajectory can leave it).
Box aux_enclosure_of_step(const NonlinearSys &sys, const Box &x0, double dt) {
    Box guess = aux_inflated_hull(x0, aux_picard(sys, x0, x0, dt));
    for (int iteration = 0; iteration < 20; ++iteration) {
        const Box image = aux_picard(sys, x0, guess, dt);
        if (aux_inside(image, guess)) return image;
        guess = aux_inflated_hull(guess, image);
    }
    CORA_FAIL(Error::special_error, "cora: no enclosure of the states of a step was found; "
                                    "the time step " +
                                        std::to_string(dt) + " is too large");
}

// The linearization error -------------------------------------------------------------------------
/// The Lagrange remainder 1/2 (x - xs)' H_i(xi) (x - xs) over the box, a column of one range per
/// component; the Hessian is symmetric, so the mixed terms count twice, and an entry that is
/// identically 0 adds nothing.
Box aux_lagrange_remainder(const NonlinearSys &sys, const Box &box, const Tensor &xs) {
    const int64_t n = xs.shape()[0];
    const std::vector<std::optional<Range>> H = sys.hessian_enclosure(box);
    const std::vector<Tensor> lo = aux_entries(box.lo), hi = aux_entries(box.hi);
    const std::vector<Tensor> c = aux_entries(xs);
    std::vector<Range> delta;
    delta.reserve(n);
    for (int64_t j = 0; j < n; ++j) delta.emplace_back(lo[j] - c[j], hi[j] - c[j]);

    std::vector<Tensor> los, his;
    for (int64_t i = 0; i < n; ++i) {
        Range sum(c[0].zeros_like());
        for (int64_t j = 0; j < n; ++j) {
            if (const auto &h = H[(i * n + j) * n + j]) sum = sum + square(delta[j]) * *h;
            for (int64_t k = j + 1; k < n; ++k)
                if (const auto &h = H[(i * n + j) * n + k])
                    sum = sum + delta[j] * delta[k] * *h * 2.0;
        }
        sum = sum * 0.5;
        los.push_back(sum.lo);
        his.push_back(sum.hi);
    }
    return {Tensor::concat(los, -2), Tensor::concat(his, -2)};
}

// The affine system with its input ----------------------------------------------------------------
/// The zonotope of the input u = f(xs) + w, w in L: the center f(xs) + mid(L), and one axis-aligned
/// generator per component (a zero one where L is a point).
Zonotope aux_input_set(const Tensor &f0, const Box &L) { return {f0 + L.mid(), L.rad().diag()}; }

/// The largest row sum of |A|, the infinity norm of the matrix, as a tensor (1, 1).
Tensor aux_norm(const Tensor &A) { return A.abs().sum(-1, true).matrix_transpose().max(-1, true); }

/// The bound on every entry of the input solution past the Taylor terms 0..taylor_terms:
/// dt (|A| dt)^(q+1) / (q+2)! / (1 - |A| dt / (q+3)) with q = taylor_terms.
Tensor aux_series_remainder(const Tensor &norm_a, double dt, int taylor_terms) {
    const Tensor r = norm_a * dt;
    const int q = taylor_terms;
    if (r.data()[0] >= q + 3)
        CORA_FAIL(Error::special_error,
                  "cora: the time step is too large for the Taylor series of the "
                  "linearized system; use a smaller step or more taylor_terms");
    Tensor term = r.full_like(r.shape(), dt); // dt * r^(q+1) / (q+2)!
    for (int i = 1; i <= q + 1; ++i) term = term.multiply(r) * (1.0 / (i + 1));
    return term.divide(1.0 - r * (1.0 / (q + 3))) * (1.0 / (q + 2));
}

/// Z shifted by the column p.
Zonotope aux_shifted(const Zonotope &Z, const Tensor &p) {
    return {Z.center() + p, Z.generators()};
}

} // namespace


// ===========================================  MAIN  =========================================== //

ReachSet NonlinearSys::reach(const ReachProblem &problem, const alg::NonlinearReach &alg) const {
    detail::Entry entry;
    // one initial zonotope of the dimension, no input, a positive horizon
    CORA_CHECK(problem.initial_set.has_value(), Error::wrong_value,
               [] { return std::string("reach: the problem needs an initial_set"); });
    const Zonotope &X0 = *problem.initial_set;
    CORA_CHECK(X0.center().shape() == std::vector<int64_t>({n_, 1}) &&
                   X0.generators().shape().size() == 2,
               Error::not_supported, [&] {
                   return "reach: the initial set of a nonlinear system is one zonotope of the "
                          "dimension " +
                          std::to_string(n_) + "; batches are not supported";
               });
    CORA_CHECK(!problem.input_set.has_value(), Error::not_supported, [] {
        return std::string("reach: a NonlinearSys has no inputs; leave input_set empty");
    });
    CORA_CHECK(problem.t_final > 0 && problem.t_final < 1e300, Error::wrong_value, [&] {
        return "reach: t_final must be positive and finite, got " + std::to_string(problem.t_final);
    });
    if (entry.check()) std::visit([](const auto &a) { a.validate(); }, alg);
    // the options of the algorithm and the number of steps
    const alg::Linearization &options = std::get<alg::Linearization>(alg);
    const double time_step = options.time_step;
    const int taylor_terms = options.taylor_terms, zonotope_order = options.zonotope_order;
    const int steps = priv_num_steps(problem.t_final, time_step);
    ReachSet R;
    Zonotope X = X0.reduce(zonotope_order);

    for (int k = 0; k < steps; ++k) {
        // (i) the abstraction around the center of the set: A, the input set U and F
        const Tensor xs = X.center();
        const Tensor A = jacobian(xs);
        const Box step_box = aux_enclosure_of_step(*this, aux_box(X), time_step);
        const Zonotope U = aux_input_set(dynamics(xs), aux_lagrange_remainder(*this, step_box, xs));
        const Interval F = LinearSys(A).correction_matrix_state(time_step, taylor_terms);
        const Tensor e_adt = (A * time_step).expm();

        // (ii) the input solution: sum_i A^i dt^{i+1}/(i+1)! U over the step, and over [0, dt]
        // where the factor t^{i+1}/(i+1)! in [0, dt^{i+1}/(i+1)!] turns U into the zonotope Ubar
        const Zonotope Ubar(U.center() * 0.5,
                            Tensor::concat({U.center() * 0.5, U.generators()}, -1));
        Tensor Ai = A.eye_like();
        double factor = time_step; // dt^{i+1} / (i+1)!
        Zonotope Ppoint = U.linear_map(Ai * factor), Pinterval = Ubar.linear_map(Ai * factor);
        for (int i = 1; i <= taylor_terms; ++i) {
            Ai = Ai.matmul(A);
            factor *= time_step / (i + 1);
            Ppoint = Ppoint.plus(U.linear_map(Ai * factor));
            Pinterval = Pinterval.plus(Ubar.linear_map(Ai * factor));
        }
        // The rest of the series moves every point of the input solution by at most eps |u|.
        const Tensor eps = aux_series_remainder(aux_norm(A), time_step, taylor_terms);
        const Interval hull = U.interval();
        const Tensor abs_max = hull.inf().abs().maximum(hull.sup().abs());
        const Tensor reach_of_u =
            xs.ones_like().matmul(eps.multiply(abs_max.matrix_transpose().sum(-1, true)));
        const Zonotope remainder(xs.zeros_like(), reach_of_u.diag());

        // (iii) the sets, from the homogeneous solution of the deviation from xs
        const Zonotope deviation(xs.zeros_like(), X.generators());
        const Zonotope next = deviation.linear_map(e_adt);
        const Zonotope between = deviation.linear_combination(next).plus(deviation.linear_map(F));
        R.time_interval.push_back(aux_shifted(between.plus(Pinterval).plus(remainder), xs));
        R.time_point.push_back(X);
        X = aux_shifted(next.plus(Ppoint).plus(remainder), xs).reduce(zonotope_order);
    }
    R.time_point.push_back(X);
    R.time_step = time_step;
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
