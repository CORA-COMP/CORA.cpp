// output_set - the output sets y = C x of reachable sets, as CORA's outputSet
//
// Syntax:   ReachSet Y = sys.output_set(R);
// Inputs:   R - reachable sets in the state space
// Outputs:  Y - the same sets mapped by C (R itself if the system has no C)
// See also: reach, priv_outputSet_canonicalForm in CORA
//
// CORA: contDynamics/@linearSys/outputSet.m

#include "cora/cont_dynamics/linear_sys/linear_sys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ReachSet LinearSys::output_set(const ReachSet &R) const {
    detail::Entry entry;
    if (!C_) return R;

    ReachSet Y;
    for (const Zonotope &Z : R.time_interval) Y.time_interval.push_back(Z.linear_map(*C_));
    for (const Zonotope &Z : R.time_point) Y.time_point.push_back(Z.linear_map(*C_));
    return Y;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
