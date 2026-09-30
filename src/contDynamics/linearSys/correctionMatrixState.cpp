// correctionMatrixState - the matrix F of the curvature enlargement, as CORA's taylorMatrices
//
// The trajectories between two time points curve away from the chord between them; F X_k
// encloses that. F sums the Taylor terms i >= 2 of e^{A dt}, each weighted by where t^i - t is
// extremal on [0, dt], and adds the remainder of the series past order taylorTerms.
//
// Syntax:   Interval F = sys.correctionMatrixState(timeStep, taylorTerms);
// Inputs:   timeStep - step size;  taylorTerms - order of the Taylor series
// Outputs:  F - interval matrix (..., n, n)
// See also: reach

#include "contDynamics/linearSys/linearSys.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The elementwise bound W on the series past the last term: |e^{|A| dt} - series|.
Tensor aux_remainder(const Tensor &Aabs, double timeStep, const Tensor &series) {
    return ((Aabs * timeStep).expm() - series).abs();
}

/// The weight (i^(-i/(i-1)) - i^(-1/(i-1))) dt^i / i! of the i-th Taylor term.
double aux_weight(int i, double dtOverFac) {
    return (std::pow(i, -double(i) / (i - 1)) - std::pow(i, -1.0 / (i - 1))) * dtOverFac;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval LinearSys::correctionMatrixState(double timeStep, int taylorTerms) const {
    const Tensor Aabs = A_.abs();
    Tensor Ai = A_, Aiabs = Aabs;                // A^i and |A|^i
    Tensor Fneg = A_.zerosLike(), Fpos = A_.zerosLike();
    Tensor series = A_.eyeLike() + Aabs * timeStep;  // sum_i |A|^i dt^i / i!, up to i = 1
    double dtOverFac = timeStep;                 // dt^i / i!

    for (int i = 2; i <= taylorTerms; ++i) {
        Ai = Ai.matmul(A_);
        Aiabs = Aiabs.matmul(Aabs);
        dtOverFac *= timeStep / i;
        series = series + Aiabs * dtOverFac;
        // The weight is negative: the negative part of A^i bounds F from above.
        const double weight = aux_weight(i, dtOverFac);
        Fpos = Fpos + Ai.neg() * weight;
        Fneg = Fneg + Ai.pos() * weight;
    }
    const Tensor W = aux_remainder(Aabs, timeStep, series);
    return {Fneg - W, Fpos + W};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
