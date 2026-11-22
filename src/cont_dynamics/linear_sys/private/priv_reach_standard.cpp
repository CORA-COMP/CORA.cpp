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
// Inputs:   X0 - initial set
//           e_adt - e^{A dt}
//           F - correction matrix
//           steps - number of steps
//           in - the input solution of one step
//           zonotope_order - order that bounds the sets
// Outputs:  R - the reachable sets, see LinearSys::reach
// See also: priv_reach_wrappingfree
//
// CORA: contDynamics/@linearSys/private/priv_reach_standard.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ReachSet priv_reach_standard(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                             int steps) {
    ReachSet R;
    R.time_interval.reserve(steps);
    R.time_point.reserve(steps + 1);
    // The identity map broadcasts X0 to a batch of systems, so every step has the batch shape.
    Zonotope X = X0.linear_map(e_adt.eye_like());

    for (int k = 0; k < steps; ++k) {
        const Zonotope Xnext = X.linear_map(e_adt);     // (i) propagate the time-point set
        const Zonotope H = X.linear_combination(Xnext); // (ii) enclose both sets
        const Zonotope C = X.linear_map(F);             // (iii) curvature enlargement
        R.time_interval.push_back(H.plus(C));
        R.time_point.push_back(X);
        X = Xnext;
    }
    R.time_point.push_back(X);
    return R;
}

StandardSteps::StandardSteps(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                             const InputSolution &in, int zonotope_order)
    : e_adt_(e_adt), F_(F), in_(in), zonotope_order_(zonotope_order),
      Htp_(X0.linear_map(e_adt.eye_like())), Hti_(Htp_), PUnext_(in.PU), PU_(in.PU),
      time_point_(Htp_) {} // the identity map broadcasts a batch of sets

Zonotope StandardSteps::next() {
    if (k_++ == 0) {
        // First step: the affine solution of the time point and of the interval, without C_input.
        const Zonotope &X = time_point_;
        Htp_ = X.linear_map(e_adt_) + in_.Pu;
        Hti_ = X.linear_combination(Htp_).plus(X.linear_map(F_));
    } else {
        // later steps map both affine solutions and add the time-varying input of one more step
        Htp_ = (Htp_.linear_map(e_adt_) + in_.Pu).reduce(zonotope_order_);
        Hti_ = (Hti_.linear_map(e_adt_) + in_.Pu).reduce(zonotope_order_);
        PUnext_ = PUnext_.linear_map(e_adt_);
        PU_ = PU_.plus(PUnext_).reduce(zonotope_order_);
    }
    time_point_ = Htp_.plus(PU_);
    return Hti_.plus(PU_).plus(in_.Cinput);
}

ReachSet priv_reach_standard(const Zonotope &X0, const Tensor &e_adt, const Interval &F,
                             const InputSolution &in, int steps, int zonotope_order) {
    StandardSteps step(X0, e_adt, F, in, zonotope_order);
    if (steps < 1) return {{}, {step.time_point()}};
    ReachSet R;
    R.time_interval.reserve(steps);
    R.time_point.reserve(steps + 1);
    R.time_point.push_back(step.time_point());
    // one step at a time: the enclosure of the step, then the set at its end
    for (int k = 0; k < steps; ++k) {
        R.time_interval.push_back(step.next());
        R.time_point.push_back(step.time_point());
    }
    return R;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
