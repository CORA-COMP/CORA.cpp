// priv_reach_wrappingfree - the "wrapping-free" algorithm of linearSys.reach
//
// The first step's enclosure is computed once and mapped forward:
//    R_0 = linComb(X0, e^{A dt} X0) + F X0;   R_k = e^{A k dt} R_0
//
// With inputs it is Girard's algorithm as in CORA: Hti and Htp are propagated, and the time-varying
// input solution accumulates as a box, so the sets keep their size:
//    PU <- PU + [e^{A dt k} PU_1];  R_k = Hti + PU + C_input;  timePoint_k = Htp + PU
//
// Syntax:   R = priv_reach_wrappingfree(X0, eAdt, F, steps);
//           R = priv_reach_wrappingfree(X0, eAdt, F, in, steps);
// Inputs:   X0 - initial set;  eAdt - e^{A dt};  F - correction matrix;  steps - number of steps
//           in - the input solution of one step
// Outputs:  R - the reachable sets, see LinearSys::reach
// See also: priv_reach_standard

#include "contDynamics/linearSys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


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

Reach priv_reach_wrappingfree(const Zonotope &X0, const Tensor &eAdt, const Interval &F,
                              const InputSolution &in, int steps) {
    Reach R;
    const Zonotope X = X0.mtimes(eAdt.eyeLike());  // the identity map broadcasts a batch of sets
    if (steps < 1) return {{}, {X}};

    // First step: the affine solution of the time point and of the interval, without C_input.
    Zonotope Htp = X.mtimes(eAdt) + in.Pu;
    Zonotope Hti = X.linComb(Htp).plus(X.mtimes(F));
    Zonotope PUnext = in.PU;
    Tensor PUrad = in.PU.interval().rad();  // the time-varying solution as a box around zero
    R.timePoint.push_back(X);
    R.timeInt.push_back(Hti.plus(in.PU).plus(in.Cinput));
    R.timePoint.push_back(Htp.plus(in.PU));

    for (int k = 1; k < steps; ++k) {
        Hti = Hti.mtimes(eAdt) + in.Pu;
        Htp = Htp.mtimes(eAdt) + in.Pu;
        PUnext = PUnext.mtimes(eAdt);
        PUrad = PUrad + PUnext.interval().rad();
        const Zonotope PU = priv_box(PUrad);
        R.timeInt.push_back(Hti.plus(PU).plus(in.Cinput));
        R.timePoint.push_back(Htp.plus(PU));
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
