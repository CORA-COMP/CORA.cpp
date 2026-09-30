// reach - the reachable sets of x' = f(x), as CORA's nonlinearSys.reach with algorithm "lin"
//
// Every step abstracts f around the center xs of the current set X_k by the affine system
//     x' = f(xs) + A (x - xs) + w,   w in L,
// where A is the Jacobian at xs and L the Lagrange remainder over a box that encloses all states
// of the step. The affine system is solved with a set of inputs U = f(xs) + L:
//     X_{k+1} = xs + e^{A dt} (X_k - xs) + sum_i A^i dt^{i+1}/(i+1)! U + (Taylor remainder),
// and the sets in between with the enclosure of linearSys.reach plus the analogous input terms.
//
// Syntax:   R = sys.reach(X0, timeStep, tFinal, taylorTerms, zonotopeOrder);
// Inputs:   X0 - initial set (zonotope);  timeStep, tFinal - step size and time horizon
//           taylorTerms - order of the Taylor series of the linearized system
//           zonotopeOrder - a set keeps at most zonotopeOrder * n generators
// Outputs:  R - timeInt[k] encloses the states over [k*timeStep, (k+1)*timeStep];
//           timePoint[k] is the set at k*timeStep
// See also: simulate, LinearSys::reach

#include "contDynamics/nonlinearSys/nonlinearSys.h"
#include "contDynamics/linearSys/private/priv.h"

#include <optional>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The bounds of a box, each a column (n, 1).
using Box = Range;

/// A tensor of the shape and backend of `like` filled with `value`.
Tensor aux_filled(const Tensor &like, double value) {
    int64_t count = 1;
    for (const int64_t d : like.shape()) count *= d;
    return Tensor::like(like, std::vector<double>(count, value), like.shape());
}

/// The entries of a column (n, 1) as n tensors of shape (1, 1).
std::vector<Tensor> aux_entries(const Tensor &column) {
    const Tensor row = column.transpose();
    std::vector<Tensor> out;
    for (int64_t i = 0; i < column.shape()[0]; ++i) out.push_back(row.selectCols({i}));
    return out;
}

// The box of a set --------------------------------------------------------------------------------
/// The interval hull of a single zonotope.
Box aux_box(const Zonotope &Z) {
    const Interval I = Z.interval();
    return {I.inf, I.sup};
}

/// The box hull of two boxes, made 10% (and a little) larger.
Box aux_inflatedHull(const Box &a, const Box &b) {
    const Tensor lo = Tensor::minimum(a.lo, b.lo), hi = Tensor::maximum(a.hi, b.hi);
    const Tensor margin = (hi - lo) * 0.1 + aux_filled(lo, 1e-10);
    return {lo - margin, hi + margin};
}

/// The box x0 + [0, dt] f(guess): where the states of a step end up if they stay in the guess.
Box aux_picard(const NonlinearSys &sys, const Box &x0, const Box &guess, double dt) {
    const Box f = sys.enclosure(guess);
    // [0, dt] * [a, b] is [min(0, dt a), max(0, dt b)] for a step dt > 0.
    return {x0.lo + (f.lo * dt).neg(), x0.hi + (f.hi * dt).pos()};
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
Box aux_enclosureOfStep(const NonlinearSys &sys, const Box &x0, double dt) {
    Box guess = aux_inflatedHull(x0, aux_picard(sys, x0, x0, dt));
    for (int iteration = 0; iteration < 20; ++iteration) {
        const Box image = aux_picard(sys, x0, guess, dt);
        if (aux_inside(image, guess)) return image;
        guess = aux_inflatedHull(guess, image);
    }
    throw std::runtime_error("cora: no enclosure of the states of a step was found; "
                             "the time step " + std::to_string(dt) + " is too large");
}

// The linearization error -------------------------------------------------------------------------
/// The Lagrange remainder 1/2 (x - xs)' H_i(xi) (x - xs) over the box, a column of one range per
/// component; the Hessian is symmetric, so the mixed terms count twice, and an entry that is
/// identically 0 adds nothing.
Box aux_lagrangeRemainder(const NonlinearSys &sys, const Box &box, const Tensor &xs) {
    const int64_t n = xs.shape()[0];
    const std::vector<std::optional<Range>> H = sys.hessianEnclosure(box);
    const std::vector<Tensor> lo = aux_entries(box.lo), hi = aux_entries(box.hi);
    const std::vector<Tensor> c = aux_entries(xs);
    std::vector<Range> delta;
    for (int64_t j = 0; j < n; ++j) delta.emplace_back(lo[j] - c[j], hi[j] - c[j]);

    std::vector<Tensor> los, his;
    for (int64_t i = 0; i < n; ++i) {
        Range sum(aux_filled(c[0], 0.0));
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
    return {Tensor::catRows(los), Tensor::catRows(his)};
}

// The affine system with its input ----------------------------------------------------------------
/// The zonotope of the input u = f(xs) + w, w in L: the center f(xs) + mid(L), and one axis-aligned
/// generator per component (a zero one where L is a point).
Zonotope aux_inputSet(const Tensor &f0, const Box &L) { return {f0 + L.mid(), L.rad().diag()}; }

/// The largest row sum of |A|, the infinity norm of the matrix, as a tensor (1, 1).
Tensor aux_norm(const Tensor &A) { return A.abs().sumLast().transpose().maxLast(); }

/// The bound on every entry of the input solution past the Taylor terms 0..taylorTerms:
/// dt (|A| dt)^(q+1) / (q+2)! / (1 - |A| dt / (q+3)) with q = taylorTerms.
Tensor aux_seriesRemainder(const Tensor &normA, double dt, int taylorTerms) {
    const Tensor r = normA * dt;
    const int q = taylorTerms;
    if (r.data()[0] >= q + 3)
        throw std::runtime_error("cora: the time step is too large for the Taylor series of the "
                                 "linearized system; use a smaller step or more taylorTerms");
    Tensor term = aux_filled(r, dt);                        // dt * r^(q+1) / (q+2)!
    for (int i = 1; i <= q + 1; ++i) term = term.mul(r) * (1.0 / (i + 1));
    return term.div(aux_filled(r, 1.0) - r * (1.0 / (q + 3))) * (1.0 / (q + 2));
}

/// Z shifted by the column p.
Zonotope aux_shifted(const Zonotope &Z, const Tensor &p) { return {Z.c + p, Z.G}; }

} // namespace


// ===========================================  MAIN  =========================================== //

Reach NonlinearSys::reach(const Zonotope &X0, double timeStep, double tFinal, int taylorTerms,
                          int zonotopeOrder) const {
    if (timeStep <= 0) throw std::invalid_argument("cora: the time step must be positive");
    if (X0.c.shape() != std::vector<int64_t>({n_, 1}) || X0.G.shape().size() != 2)
        throw std::invalid_argument("cora: the initial set of a nonlinear system is one zonotope "
                                    "of dimension " + std::to_string(n_) +
                                    "; batches are not supported");
    const int steps = priv_numSteps(tFinal, timeStep);
    Reach R;
    Zonotope X = X0.reduce(zonotopeOrder);

    for (int k = 0; k < steps; ++k) {
        // (i) the abstraction around the center of the set: A, the input set U and F
        const Tensor xs = X.c;
        const Tensor A = jacobian(xs);
        const Box stepBox = aux_enclosureOfStep(*this, aux_box(X), timeStep);
        const Zonotope U = aux_inputSet(dynamics(xs), aux_lagrangeRemainder(*this, stepBox, xs));
        const Interval F = LinearSys(A).correctionMatrixState(timeStep, taylorTerms);
        const Tensor eAdt = (A * timeStep).expm();

        // (ii) the input solution: sum_i A^i dt^{i+1}/(i+1)! U over the step, and over [0, dt]
        // where the factor t^{i+1}/(i+1)! in [0, dt^{i+1}/(i+1)!] turns U into the zonotope Ubar
        const Zonotope Ubar(U.c * 0.5, Tensor::catLast({U.c * 0.5, U.G}));
        Tensor Ai = A.eyeLike();
        double factor = timeStep;  // dt^{i+1} / (i+1)!
        Zonotope Ppoint = U.mtimes(Ai * factor), Pinterval = Ubar.mtimes(Ai * factor);
        for (int i = 1; i <= taylorTerms; ++i) {
            Ai = Ai.matmul(A);
            factor *= timeStep / (i + 1);
            Ppoint = Ppoint.plus(U.mtimes(Ai * factor));
            Pinterval = Pinterval.plus(Ubar.mtimes(Ai * factor));
        }
        // The rest of the series moves every point of the input solution by at most eps |u|.
        const Tensor eps = aux_seriesRemainder(aux_norm(A), timeStep, taylorTerms);
        const Interval hull = U.interval();
        const Tensor absMax = Tensor::maximum(hull.inf.abs(), hull.sup.abs());
        const Tensor reachOfU = aux_filled(xs, 1.0).matmul(eps.mul(absMax.transpose().sumLast()));
        const Zonotope remainder(aux_filled(xs, 0.0), reachOfU.diag());

        // (iii) the sets, from the homogeneous solution of the deviation from xs
        const Zonotope deviation(aux_filled(xs, 0.0), X.G);
        const Zonotope next = deviation.mtimes(eAdt);
        const Zonotope between = deviation.linComb(next).plus(deviation.mtimes(F));
        R.timeInt.push_back(aux_shifted(between.plus(Pinterval).plus(remainder), xs));
        R.timePoint.push_back(X);
        X = aux_shifted(next.plus(Ppoint).plus(remainder), xs).reduce(zonotopeOrder);
    }
    R.timePoint.push_back(X);
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
