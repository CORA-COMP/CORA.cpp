// priv_expmRemainder - the error of the truncated series of e^{A dt}, as CORA's function
//
// The series sum_{i=0}^{taylorTerms} |A|^i dt^i / i! is compared with e^{|A| dt}; the absolute
// value keeps the difference free of cancellation.
//
// Syntax:   Interval E = priv_expmRemainder(A, timeStep, taylorTerms);
// Inputs:   A - system matrix;  timeStep - step size;  taylorTerms - order of the series
// Outputs:  E - the box [-W, W] around zero with W = |e^{|A| dt} - series|
// See also: LinearSys::correctionMatrixInput, priv_inputSolution

#include "contDynamics/linearSys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval priv_expmRemainder(const Tensor &A, double timeStep, int taylorTerms) {
    const Tensor Aabs = A.abs();
    Tensor Ai = A.eyeLike(), series = A.eyeLike();  // |A|^i and the series up to i
    double dtOverFac = 1.0;                         // dt^i / i!

    for (int i = 1; i <= taylorTerms; ++i) {
        Ai = Ai.matmul(Aabs);
        dtOverFac *= timeStep / i;
        series = series + Ai * dtOverFac;
    }
    const Tensor W = ((Aabs * timeStep).expm() - series).abs();
    return {W * -1.0, W};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
