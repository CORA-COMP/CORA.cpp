#include "contDynamics/linear_sys.h"

#include <cmath>

namespace cora::ct {

Reach LinearSys::reach(const Zonotope &X0, double time_step, double t_final, int taylor_terms,
                       Algorithm algorithm) const {
    const int steps = static_cast<int>(std::ceil(t_final / time_step - 1e-9));
    const Tensor eAdt = (A_ * time_step).expm();
    const IntervalMatrix F = correction_matrix_state(time_step, taylor_terms);

    Reach out;
    if (algorithm == Algorithm::Standard) {
        Zonotope X = X0;
        for (int k = 0; k < steps; ++k) {
            const Zonotope Xnext = X.mtimes(eAdt); // propagate the time-point set
            const Zonotope H = X.lin_comb(Xnext);  // enclose both
            const Zonotope C = X.mtimes(F);        // curvature enlargement
            out.time_int.push_back(H.plus(C));
            out.time_point.push_back(X);
            X = Xnext;
        }
        out.time_point.push_back(X);
    } else {
        const Zonotope R0 = X0.lin_comb(X0.mtimes(eAdt)).plus(X0.mtimes(F));
        Tensor eAkdt = A_.eye_like(); // e^{A k Δt}
        for (int k = 0; k <= steps; ++k) {
            if (k < steps) out.time_int.push_back(R0.mtimes(eAkdt));
            out.time_point.push_back(X0.mtimes(eAkdt));
            eAkdt = eAkdt.matmul(eAdt);
        }
    }
    return out;
}

IntervalMatrix LinearSys::correction_matrix_state(double time_step, int taylor_terms) const {
    const Tensor Aabs = A_.abs();
    Tensor Ai = A_, Ai_abs = Aabs; // A^i and |A|^i
    Tensor Fneg = A_.zeros_like(), Fpos = A_.zeros_like();
    Tensor M = A_.eye_like() + Aabs * time_step; // Σ |A|^i Δt^i / i!
    double dt_over_fac = time_step;              // Δt^i / i!
    for (int i = 2; i <= taylor_terms; ++i) {
        Ai = Ai.matmul(A_);
        Ai_abs = Ai_abs.matmul(Aabs);
        dt_over_fac *= time_step / i;
        M = M + Ai_abs * dt_over_fac;
        // The factor is negative, so the negative part of A^i bounds F from above.
        const double factor =
            (std::pow(i, -double(i) / (i - 1)) - std::pow(i, -1.0 / (i - 1))) * dt_over_fac;
        Fpos = Fpos + Ai.neg() * factor;
        Fneg = Fneg + Ai.pos() * factor;
    }
    // Elementwise bound on the series past order η: E = [-W, W].
    const Tensor W = ((Aabs * time_step).expm() - M).abs();
    return IntervalMatrix::from_bounds(Fneg - W, Fpos + W);
}

} // namespace cora::ct
