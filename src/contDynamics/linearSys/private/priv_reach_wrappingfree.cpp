// priv_reach_wrappingfree - the "wrapping-free" algorithm of linearSys.reach
//
// The first step's enclosure is computed once and mapped forward:
//    R_0 = linComb(X0, e^{A dt} X0) + F X0;   R_k = e^{A k dt} R_0
//
// Syntax:   R = priv_reach_wrappingfree(X0, eAdt, F, steps);
// Inputs:   X0 - initial set;  eAdt - e^{A dt};  F - correction matrix;  steps - number of steps
// Outputs:  R - the reachable sets, see LinearSys::reach
// See also: priv_reach_standard

#include "contDynamics/linearSys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {


// ===========================================  MAIN  =========================================== //

Reach priv_reach_wrappingfree(const Zonotope &X0, const Tensor &eAdt, const Interval &F,
                              int steps) {
    Reach R;
    const Zonotope R0 = X0.linComb(X0.mtimes(eAdt)).plus(X0.mtimes(F));

    Tensor eAkdt = eAdt.eyeLike();  // e^{A k dt}, starting at k = 0
    for (int k = 0; k <= steps; ++k) {
        if (k < steps) R.timeInt.push_back(R0.mtimes(eAkdt));
        R.timePoint.push_back(X0.mtimes(eAkdt));
        eAkdt = eAkdt.matmul(eAdt);
    }
    return R;
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
