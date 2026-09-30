// correctionMatrixInput - the matrix G of the input's curvature enlargement, as in CORA
//
// Between two time points the effect of a constant input u curves away from the chord: G u
// encloses that. G sums the Taylor terms A^(i-1) dt^i / i! for i >= 2, each weighted by where
// t^i - t is extremal on [0, dt], and adds the remainder of the series past taylorTerms times dt.
//
// Syntax:   Interval G = sys.correctionMatrixInput(timeStep, taylorTerms);
// Inputs:   timeStep - step size;  taylorTerms - order of the Taylor series
// Outputs:  G - interval matrix (n, n)
// See also: correctionMatrixState, reach

#include "contDynamics/linearSys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval LinearSys::correctionMatrixInput(double timeStep, int taylorTerms) const {
    Tensor Ai = A_;  // A^(i-1)
    Tensor Gneg = A_.zerosLike(), Gpos = A_.zerosLike();
    double dtOverFac = timeStep;  // dt^i / i!, after the update in the loop

    for (int i = 2; i <= taylorTerms + 1; ++i) {
        dtOverFac *= timeStep / i;
        // The weight is negative: the negative part of A^(i-1) bounds G from above.
        const double weight = priv_extremalWeight(i, dtOverFac);
        Gpos = Gpos + Ai.neg() * weight;
        Gneg = Gneg + Ai.pos() * weight;
        Ai = Ai.matmul(A_);
    }
    const Interval E = priv_expmRemainder(A_, timeStep, taylorTerms);
    return {Gneg + E.inf * timeStep, Gpos + E.sup * timeStep};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
