// priv_reach_wrappingfree - the "wrapping-free" algorithm of linear_sys.reach
//
// The first step's enclosure is computed once and mapped forward:
//    R_0 = linear_combination(X0, e^{A dt} X0) + F X0;   R_k = e^{A k dt} R_0
//
// With inputs it is Girard's algorithm as in CORA: Hti and Htp are propagated, and the time-varying
// input solution accumulates as a box, so the sets keep their size:
//    PU <- PU + [e^{A dt k} PU_1];  R_k = Hti + PU + C_input;  timePoint_k = Htp + PU
//
// Syntax:   R = priv_reach_wrappingfree(X0, e_adt, F, steps);
//           R = priv_reach_wrappingfree(X0, e_adt, F, in, steps);
// Inputs:   X0 - initial set
//           e_adt - e^{A dt}
//           F - correction matrix
//           steps - number of steps
//           in - the input solution of one step
// Outputs:  R - the reachable sets, see LinearSys::reach
// See also: priv_reach_standard
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ReachSet priv_reach_wrappingfree(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                                 int steps) {
    ReachSet R;
    const Zonotope R0 = X0.linear_combination(X0.linear_map(e_adt)).plus(X0.linear_map(F));

    Tensor e_akdt = e_adt.eye_like(); // e^{A k dt}, starting at k = 0
    for (int k = 0; k <= steps; ++k) {
        if (k < steps) R.time_interval.push_back(R0.linear_map(e_akdt));
        R.time_point.push_back(X0.linear_map(e_akdt));
        e_akdt = e_akdt.matmul(e_adt);
    }
    return R;
}

ReachSet priv_reach_wrappingfree(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                                 const InputSolution &in, int steps) {
    ReachSet R;
    const Zonotope X =
        X0.linear_map(e_adt.eye_like()); // the identity map broadcasts a batch of sets
    if (steps < 1) return {{}, {X}};

    // First step: the affine solution of the time point and of the interval, without C_input.
    Zonotope Htp = X.linear_map(e_adt) + in.Pu;
    Zonotope Hti = X.linear_combination(Htp).plus(X.linear_map(F));
    Zonotope PUnext = in.PU;
    Tensor PUrad = in.PU.interval().rad(); // the time-varying solution as a box around zero
    R.time_point.push_back(X);
    R.time_interval.push_back(Hti.plus(in.PU).plus(in.Cinput));
    R.time_point.push_back(Htp.plus(in.PU));

    for (int k = 1; k < steps; ++k) {
        Hti = Hti.linear_map(e_adt) + in.Pu;
        Htp = Htp.linear_map(e_adt) + in.Pu;
        PUnext = PUnext.linear_map(e_adt);
        PUrad = PUrad + PUnext.interval().rad();
        const Zonotope PU = priv_box(PUrad);
        R.time_interval.push_back(Hti.plus(PU).plus(in.Cinput));
        R.time_point.push_back(Htp.plus(PU));
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
