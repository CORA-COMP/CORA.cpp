// location - the constructor of a location, as CORA's location(name, inv, trans, sys)
//
// Syntax:   Location loc(name, invariant, transitions, flow, c);
// Inputs:   name - the name of the location
//           invariant - the polytope the state stays in
//           transitions - the jumps out of the location
//           flow - the linear system x' = A x + B u, without an output map C
//           c - the constant input (n, 1) of the flow, optional
// Outputs:  loc - the location
// See also: Transition, HybridAutomaton
//
// CORA: hybridDynamics/@location/location.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/hybrid_dynamics/location/location.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Location::Location(std::string name, Polytope invariant, std::vector<Transition> transitions,
                   LinearSys flow, std::optional<Tensor> c)
    : name_(std::move(name)), invariant_(std::move(invariant)),
      transitions_(std::move(transitions)), flow_(std::move(flow)),
      c_(c ? std::move(*c) : flow_.A().zeros_like({flow_.dim(), 1})) {
    detail::Entry entry;
    // one flow of the dimension of the invariant and of every guard
    const int64_t n = flow_.dim();
    CORA_CHECK(flow_.A().ndim() == 2 && !flow_.C(), Error::wrong_input_in_constructor, [] {
        return std::string("Location: the flow must be one system without an output map C");
    });
    CORA_CHECK(c_.shape() == std::vector<int64_t>({n, 1}), Error::wrong_input_in_constructor, [&] {
        return "Location: c must be a column (" + std::to_string(n) + ", 1), got the shape " +
               shape_text(c_.shape());
    });
    CORA_CHECK(invariant_.dim() == n, Error::wrong_input_in_constructor, [&] {
        return "Location: the invariant has the dimension " + std::to_string(invariant_.dim()) +
               ", the flow " + std::to_string(n);
    });
    for (const Transition &t : transitions_)
        CORA_CHECK(t.guard().dim() == n && t.reset().post_state_dim() == n,
                   Error::wrong_input_in_constructor, [&] {
                       return "Location: every guard and reset must have the dimension " +
                              std::to_string(n) + " of the flow";
                   });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
