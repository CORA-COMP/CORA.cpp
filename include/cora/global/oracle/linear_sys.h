// linear_sys - lean::LinearSys, the reachability of x' = A x computed by CORALean
//
// Syntax:     lean::LinearSys sys(A);   auto R = sys.reach(X0, t_final, alg::Standard{.time_step =
// 0.1}); Outputs:    R.time_point (steps + 1 sets) and R.time_interval (steps sets): lean zonotopes
// whose
//             nominal part and error box together enclose the reachable set
// Scheme:     CORALean's sound float reach (no input term yet: B and U are not supported)
// See also:   lean/zonotope.h, contDynamics/linear_sys/linear_sys.h
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/global/oracle/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

struct ReachSet {
    std::vector<Zonotope> time_point, time_interval;
};

class LinearSys {
  public:
    /// The dynamics matrix (n, n) in the current dtype.
    explicit LinearSys(const cora::Tensor &A);

    /// The reachable sets from X0 over [0, t_final] by the algorithm "standard" in steps of
    /// alg.time_step (t_final / time_step must be an integer); the Taylor series has
    /// alg.taylor_terms terms, zonotopes are reduced to alg.zonotope_order.
    ReachSet reach(const Zonotope &X0, double t_final, const alg::Standard &alg) const;

  private:
    Tensor A_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
