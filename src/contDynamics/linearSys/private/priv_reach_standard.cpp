// priv_reach_standard - the "standard" algorithm of linearSys.reach, as CORA's private function
//
// Every step propagates the time-point set X_k and encloses it again:
//    X_{k+1} = e^{A dt} X_k;   H = linComb(X_k, X_{k+1});   C = F X_k;   R_k = H + C
//
// With inputs it is CORA's algorithm: the affine solution Hti of the first step is propagated and
// the time-varying input solution PU accumulates as a zonotope, both reduced to zonotopeOrder:
//    Htp <- e^{A dt} Htp + Pu;  Hti <- e^{A dt} Hti + Pu;  PU <- PU + e^{A dt k} PU_1
//    R_k = Hti + PU + C_input;  timePoint_k = Htp + PU
//
// Syntax:   R = priv_reach_standard(X0, eAdt, F, steps);
//           R = priv_reach_standard(X0, eAdt, F, in, steps, zonotopeOrder);
// Inputs:   X0 - initial set;  eAdt - e^{A dt};  F - correction matrix;  steps - number of steps
//           in - the input solution of one step;  zonotopeOrder - order that bounds the sets
// Outputs:  R - the reachable sets, see LinearSys::reach
// See also: priv_reach_wrappingfree

#include "contDynamics/linearSys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The reduction to zonotopeOrder; a batch keeps all generators, reduction takes a single set.
Zonotope aux_reduce(const Zonotope &Z, int zonotopeOrder) {
    return Z.G.shape().size() == 2 ? Z.reduce(zonotopeOrder) : Z;
}

} // namespace


// ===========================================  MAIN  =========================================== //

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

Reach priv_reach_standard(const Zonotope &X0, const Tensor &eAdt, const Interval &F,
                          const InputSolution &in, int steps, int zonotopeOrder) {
    Reach R;
    const Zonotope X = X0.mtimes(eAdt.eyeLike());  // the identity map broadcasts a batch of sets
    if (steps < 1) return {{}, {X}};

    // First step: the affine solution of the time point and of the interval, without C_input.
    Zonotope Htp = X.mtimes(eAdt) + in.Pu;
    Zonotope Hti = X.linComb(Htp).plus(X.mtimes(F));
    Zonotope PUnext = in.PU, PU = in.PU;
    R.timePoint.push_back(X);
    R.timeInt.push_back(Hti.plus(PU).plus(in.Cinput));
    R.timePoint.push_back(Htp.plus(PU));

    for (int k = 1; k < steps; ++k) {
        Htp = aux_reduce(Htp.mtimes(eAdt) + in.Pu, zonotopeOrder);
        Hti = aux_reduce(Hti.mtimes(eAdt) + in.Pu, zonotopeOrder);
        PUnext = PUnext.mtimes(eAdt);
        PU = aux_reduce(PU.plus(PUnext), zonotopeOrder);
        R.timeInt.push_back(Hti.plus(PU).plus(in.Cinput));
        R.timePoint.push_back(Htp.plus(PU));
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
