// priv_reach_standard - the "standard" algorithm of linearSys.reach, as CORA's private function
//
// Every step propagates the time-point set X_k and encloses it again:
//    X_{k+1} = e^{A dt} X_k;   H = linComb(X_k, X_{k+1});   C = F X_k;   R_k = H + C
//
// Syntax:   R = priv_reach_standard(X0, eAdt, F, steps);
// Inputs:   X0 - initial set;  eAdt - e^{A dt};  F - correction matrix;  steps - number of steps
// Outputs:  R - the reachable sets, see LinearSys::reach
// See also: priv_reach_wrappingfree

#include "contDynamics/linearSys/private/priv.h"

namespace cora::ct {

Reach priv_reach_standard(const Zonotope &X0, const Tensor &eAdt, const Interval &F, int steps) {
    Reach R;
    // The identity map broadcasts X0 to a batch of systems, so every step has the batch shape.
    Zonotope X = X0.mtimes(eAdt.eyeLike());

    for (int k = 0; k < steps; ++k) {
        const Zonotope Xnext = X.mtimes(eAdt);  // (i) propagate the time-point set
        const Zonotope H = X.linComb(Xnext);    // (ii) enclose both sets
        const Zonotope C = X.mtimes(F);         // (iii) curvature enlargement
        R.timeInt.push_back(H.plus(C));
        R.timePoint.push_back(X);
        X = Xnext;
    }
    R.timePoint.push_back(X);
    return R;
}

} // namespace cora::ct
