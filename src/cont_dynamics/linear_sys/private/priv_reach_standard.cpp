// priv_reach_standard - the "standard" algorithm of linear_sys.reach, as CORA's private function
//
// Every step propagates the time-point set X_k and encloses it again:
//    X_{k+1} = e^{A dt} X_k;   H = linear_combination(X_k, X_{k+1});   C = F X_k;   R_k = H + C
//
// With inputs it is CORA's algorithm: the affine solution Hti of the first step is propagated and
// the time-varying input solution PU accumulates as a zonotope, both reduced to zonotope_order:
//    Htp <- e^{A dt} Htp + Pu;  Hti <- e^{A dt} Hti + Pu;  PU <- PU + e^{A dt k} PU_1
//    R_k = Hti + PU + C_input;  timePoint_k = Htp + PU
//
// Syntax:   R = priv_reach_standard(X0, e_adt, F, steps);
//           R = priv_reach_standard(X0, e_adt, F, in, steps, zonotope_order);
// Inputs:   X0 - initial set;  e_adt - e^{A dt};  F - correction matrix;  steps - number of steps
//           in - the input solution of one step;  zonotope_order - order that bounds the sets
// Outputs:  R - the reachable sets, see LinearSys::reach
// See also: priv_reach_wrappingfree

#include "cont_dynamics/linear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The reduction to zonotope_order; a batch keeps all generators, reduction takes a single set.
Zonotope aux_reduce(const Zonotope &Z, int zonotope_order) {
    return Z.generators().shape().size() == 2 ? Z.reduce(zonotope_order) : Z;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ReachSet priv_reach_standard(const Zonotope &X0, const Tensor &e_adt, const Interval &F, int steps) {
    ReachSet R;
    // The identity map broadcasts X0 to a batch of systems, so every step has the batch shape.
    Zonotope X = X0.linear_map(e_adt.eye_like());

    for (int k = 0; k < steps; ++k) {
        const Zonotope Xnext = X.linear_map(e_adt);  // (i) propagate the time-point set
        const Zonotope H = X.linear_combination(Xnext);    // (ii) enclose both sets
        const Zonotope C = X.linear_map(F);         // (iii) curvature enlargement
        R.time_interval.push_back(H.plus(C));
        R.time_point.push_back(X);
        X = Xnext;
    }
    R.time_point.push_back(X);
    return R;
}

ReachSet priv_reach_standard(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                          const InputSolution &in, int steps, int zonotope_order) {
    ReachSet R;
    const Zonotope X = X0.linear_map(e_adt.eye_like());  // the identity map broadcasts a batch of sets
    if (steps < 1) return {{}, {X}};

    // First step: the affine solution of the time point and of the interval, without C_input.
    Zonotope Htp = X.linear_map(e_adt) + in.Pu;
    Zonotope Hti = X.linear_combination(Htp).plus(X.linear_map(F));
    Zonotope PUnext = in.PU, PU = in.PU;
    R.time_point.push_back(X);
    R.time_interval.push_back(Hti.plus(PU).plus(in.Cinput));
    R.time_point.push_back(Htp.plus(PU));

    for (int k = 1; k < steps; ++k) {
        Htp = aux_reduce(Htp.linear_map(e_adt) + in.Pu, zonotope_order);
        Hti = aux_reduce(Hti.linear_map(e_adt) + in.Pu, zonotope_order);
        PUnext = PUnext.linear_map(e_adt);
        PU = aux_reduce(PU.plus(PUnext), zonotope_order);
        R.time_interval.push_back(Hti.plus(PU).plus(in.Cinput));
        R.time_point.push_back(Htp.plus(PU));
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
