// transition - a jump of a hybrid automaton, as CORA's transition
//
// When the state reaches the guard, it may jump to the location `target` (0-based), mapped by the
// reset. A guard is a polytope; a box is held as the polytope of its sides, and a hyperplane with
// inequalities (CORA's constrained hyperplane) is a polytope with one equality.
//
// Syntax:     Transition trans(guard, reset, target);
// Operations: the constructor checks its input (transition.cpp)
// See also:   cora/hybrid_dynamics/location/location.h, linear_reset.h
//
// CORA: hybridDynamics/@transition/transition.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/hybrid_dynamics/linear_reset/linear_reset.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Transition {
  public:
    /// The jump to the location `target` (0-based) through `guard`, mapped by `reset`; the
    /// dimension of the guard is that of the state before the jump.
    Transition(Polytope guard, LinearReset reset, int64_t target);
    Transition(const Interval &guard, LinearReset reset, int64_t target);

    const Polytope &guard() const noexcept { return guard_; }
    const LinearReset &reset() const noexcept { return reset_; }
    int64_t target() const noexcept { return target_; }

  private:
    Polytope guard_;
    LinearReset reset_;
    int64_t target_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
