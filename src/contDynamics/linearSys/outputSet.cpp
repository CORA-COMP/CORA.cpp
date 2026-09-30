// outputSet - the output sets y = C x of reachable sets, as CORA's outputSet
//
// Syntax:   Reach Y = sys.outputSet(R);
// Inputs:   R - reachable sets in the state space
// Outputs:  Y - the same sets mapped by C (R itself if the system has no C)
// See also: reach, priv_outputSet_canonicalForm in CORA

#include "contDynamics/linearSys/linearSys.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Reach LinearSys::outputSet(const Reach &R) const {
    if (!C_) return R;

    Reach Y;
    for (const Zonotope &Z : R.timeInt) Y.timeInt.push_back(Z.mtimes(*C_));
    for (const Zonotope &Z : R.timePoint) Y.timePoint.push_back(Z.mtimes(*C_));
    return Y;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
