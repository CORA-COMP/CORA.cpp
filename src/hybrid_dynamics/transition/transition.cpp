// transition - the constructor of a transition, as CORA's transition(guard, reset, target)
//
// A box guard becomes the polytope of its sides.
//
// Syntax:   Transition trans(guard, reset, target);
// Inputs:   guard - a polytope or a box of the pre-state dimension of the reset
//           reset - the linear reset of the jump
//           target - the location after the jump, 0-based
// Outputs:  trans - the transition
// See also: LinearReset, Location
//
// CORA: hybridDynamics/@transition/transition.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/hybrid_dynamics/transition/transition.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Transition::Transition(Region guard, LinearReset reset, int64_t target)
    : guard_(std::move(guard)), reset_(std::move(reset)), target_(target) {
    detail::Entry entry;
    // the guard lives where the reset starts, and the target is a location number
    CORA_CHECK(
        region_dim(guard_) == reset_.pre_state_dim(), Error::wrong_input_in_constructor, [&] {
            return "Transition: the guard has the dimension " + std::to_string(region_dim(guard_)) +
                   ", the reset maps " + std::to_string(reset_.pre_state_dim());
        });
    CORA_CHECK(target_ >= 0, Error::wrong_input_in_constructor, [&] {
        return "Transition: the target must be a location number from 0, got " +
               std::to_string(target_);
    });
}

Transition::Transition(const Interval &guard, LinearReset reset, int64_t target)
    : Transition(Polytope(guard), std::move(reset), target) {}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
