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

#include <algorithm>
#include <cmath>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Box = std::vector<Range>;

// The box of a set --------------------------------------------------------------------------------
/// The interval hull of a single zonotope as one Range per dimension.
Box aux_box(const Zonotope &Z) {
    const Interval I = Z.interval();
    const std::vector<double> lo = I.inf.data(), hi = I.sup.data();
    Box box;
    for (std::size_t i = 0; i < lo.size(); ++i) box.emplace_back(lo[i], hi[i]);
    return box;
}

/// The box hull of two boxes, made 10% (and a little) larger.
Box aux_inflatedHull(const Box &a, const Box &b) {
    Box out;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double lo = std::min(a[i].lo, b[i].lo), hi = std::max(a[i].hi, b[i].hi);
        const double margin = 0.1 * (hi - lo) + 1e-10;
        out.emplace_back(lo - margin, hi + margin);
    }
    return out;
}

/// The box x0 + [0, dt] f(guess): where the states of a step end up if they stay in the guess.
Box aux_picard(const NonlinearSys &sys, const Box &x0, const Box &guess, double dt) {
    const Box f = sys.enclosure(guess);
    Box out;
    for (std::size_t i = 0; i < x0.size(); ++i) out.push_back(x0[i] + Range(0, dt) * f[i]);
    return out;
}

bool aux_inside(const Box &inner, const Box &outer) {
    for (std::size_t i = 0; i < inner.size(); ++i)
        if (inner[i].lo < outer[i].lo || inner[i].hi > outer[i].hi) return false;
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
/// The square of a range: a range that contains 0 squares to [0, ...], not to [-..., ...].
Range aux_square(const Range &r) {
    const double a = r.lo * r.lo, b = r.hi * r.hi;
    if (r.lo <= 0 && r.hi >= 0) return Range(0, std::max(a, b));
    return Range(std::min(a, b), std::max(a, b));
}

/// The Lagrange remainder 1/2 (x - xs)' H_i(xi) (x - xs) over the box, one Range per component;
/// the Hessian is symmetric, so the mixed terms count twice.
Box aux_lagrangeRemainder(const NonlinearSys &sys, const Box &box, const std::vector<double> &xs) {
    const std::size_t n = xs.size();
    const Box H = sys.hessianEnclosure(box);
    Box delta;
    for (std::size_t j = 0; j < n; ++j) delta.push_back(box[j] - Range(xs[j]));
    Box L;
    for (std::size_t i = 0; i < n; ++i) {
        Range sum(0.0);
        for (std::size_t j = 0; j < n; ++j) {
            sum = sum + aux_square(delta[j]) * H[(i * n + j) * n + j];
            for (std::size_t k = j + 1; k < n; ++k)
                sum = sum + Range(2.0) * delta[j] * delta[k] * H[(i * n + j) * n + k];
        }
        L.push_back(sum * Range(0.5));
    }
    return L;
}

// The affine system with its input ----------------------------------------------------------------
/// The zonotope of the input u = f(xs) + w, w in L: the center f(xs) + mid(L), and one axis-aligned
/// generator per component with a nonzero radius (one zero generator if there is none).
Zonotope aux_inputSet(const Tensor &like, const std::vector<double> &f0, const Box &L) {
    const int64_t n = f0.size();
    std::vector<double> center(n);
    std::vector<int64_t> wide;
    for (int64_t i = 0; i < n; ++i) {
        center[i] = f0[i] + L[i].mid();
        if (L[i].rad() > 0) wide.push_back(i);
    }
    const int64_t m = std::max<int64_t>(1, wide.size());
    std::vector<double> G(n * m, 0.0);
    for (std::size_t k = 0; k < wide.size(); ++k) G[wide[k] * m + k] = L[wide[k]].rad();
    return {Tensor::like(like, center, {n, 1}), Tensor::like(like, G, {n, m})};
}

/// The largest row sum of |A|, the infinity norm of the row-major matrix.
double aux_norm(const std::vector<double> &A, int64_t n) {
    double norm = 0;
    for (int64_t i = 0; i < n; ++i) {
        double row = 0;
        for (int64_t j = 0; j < n; ++j) row += std::abs(A[i * n + j]);
        norm = std::max(norm, row);
    }
    return norm;
}

/// The bound on every entry of the input solution past the Taylor terms 0..taylorTerms:
/// dt (|A| dt)^(q+1) / (q+2)! / (1 - |A| dt / (q+3)) with q = taylorTerms.
double aux_seriesRemainder(double normA, double dt, int taylorTerms) {
    const double r = normA * dt;
    const int q = taylorTerms;
    if (r >= q + 3)
        throw std::runtime_error("cora: the time step is too large for the Taylor series of the "
                                 "linearized system; use a smaller step or more taylorTerms");
    double term = dt;                                       // dt * r^(q+1) / (q+2)!
    for (int i = 1; i <= q + 1; ++i) term *= r / (i + 1);
    return term / (q + 2) / (1 - r / (q + 3));
}

/// The box of half-width w in every dimension as a zonotope with the given backend.
Zonotope aux_centeredBox(const Tensor &like, const std::vector<double> &w) {
    const int64_t n = w.size();
    std::vector<double> G(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i) G[i * n + i] = w[i];
    return {Tensor::like(like, std::vector<double>(n, 0.0), {n, 1}), Tensor::like(like, G, {n, n})};
}

/// Z shifted by the point p.
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
        const std::vector<double> xs = X.c.data();
        const Tensor A = Tensor::like(X.c, jacobian(xs), {n_, n_});
        const Box stepBox = aux_enclosureOfStep(*this, aux_box(X), timeStep);
        const Zonotope U = aux_inputSet(X.c, values(xs), aux_lagrangeRemainder(*this, stepBox, xs));
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
        const double eps = aux_seriesRemainder(aux_norm(A.data(), n_), timeStep, taylorTerms);
        const Interval hull = U.interval();
        std::vector<double> reachOfU(n_, 0.0);
        const std::vector<double> lo = hull.inf.data(), hi = hull.sup.data();
        for (int64_t i = 0; i < n_; ++i)
            for (int64_t j = 0; j < n_; ++j)
                reachOfU[i] += eps * std::max(std::abs(lo[j]), std::abs(hi[j]));
        const Zonotope remainder = aux_centeredBox(X.c, reachOfU);

        // (iii) the sets, from the homogeneous solution of the deviation from xs
        const Tensor center = Tensor::like(X.c, xs, {n_, 1});
        const Zonotope deviation(X.c - center, X.G);
        const Zonotope next = deviation.mtimes(eAdt);
        const Zonotope between = deviation.linComb(next).plus(deviation.mtimes(F));
        R.timeInt.push_back(aux_shifted(between.plus(Pinterval).plus(remainder), center));
        R.timePoint.push_back(X);
        X = aux_shifted(next.plus(Ppoint).plus(remainder), center).reduce(zonotopeOrder);
    }
    R.timePoint.push_back(X);
    return R;
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
