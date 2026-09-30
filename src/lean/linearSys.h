// linearSys - lean::LinearSys, the reachability of x' = A x computed by CORALean
//
// Syntax:     lean::LinearSys sys(A);   auto R = sys.reach(X0, timeStep, tFinal, taylorTerms, 1);
// Outputs:    R.timePoint (steps + 1 sets) and R.timeInt (steps sets): lean zonotopes whose
//             nominal part and error box together enclose the reachable set
// Scheme:     CORALean's sound float reach (no input term yet: B and U are not supported)
// See also:   lean/zonotope.h, contDynamics/linearSys/linearSys.h

#pragma once

#include "lean/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

struct ReachSet {
    std::vector<Zonotope> timePoint, timeInt;
};

class LinearSys {
  public:
    /// The dynamics matrix (n, n) in the current dtype.
    explicit LinearSys(const cora::Tensor &A);

    /// The reachable sets from X0 over [0, tFinal] in steps of timeStep (tFinal / timeStep must
    /// be an integer); the Taylor series has taylorTerms terms, zonotopes are reduced to
    /// zonotopeOrder.
    ReachSet reach(const Zonotope &X0, double timeStep, double tFinal, int taylorTerms,
                   int zonotopeOrder) const;

  private:
    Tensor A_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
