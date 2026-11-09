// location - the constructor of a location, as CORA's location(name, inv, trans, sys)
//
// Syntax:   Location loc(name, invariant, transitions, flow, c);
//           Location loc(name, invariant, transitions, param_flow);
// Inputs:   name - the name of the location
//           invariant - the polytope the state stays in
//           transitions - the jumps out of the location
//           flow - the linear system x' = A x + B u, without an output map C
//           c - the constant input (n, 1) of the flow, optional
//           param_flow - a LinParamSys x' = A x + B u + c, its c the location's
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

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The invariant and every guard and reset of the dimension of the flow.
void aux_validate(const Location &loc) {
    detail::Validation entry;
    const int64_t n = loc.dim();
    CORA_CHECK(loc.invariant().dim() == n, Error::wrong_input_in_constructor, [&] {
        return "Location: the invariant has the dimension " +
               std::to_string(loc.invariant().dim()) + ", the flow " + std::to_string(n);
    });
    for (const Transition &t : loc.transitions())
        CORA_CHECK(t.guard().dim() == n && t.reset().post_state_dim() == n,
                   Error::wrong_input_in_constructor, [&] {
                       return "Location: every guard and reset must have the dimension " +
                              std::to_string(n) + " of the flow";
                   });
}

} // namespace


// ===========================================  MAIN  =========================================== //

Location::Location(std::string name, Polytope invariant, std::vector<Transition> transitions,
                   LinearSys flow, std::optional<Tensor> c)
    : name_(std::move(name)), invariant_(std::move(invariant)),
      transitions_(std::move(transitions)), flow_(std::move(flow)) {
    detail::Entry entry;
    // one flow without an output map, and c of its states
    const auto &sys = std::get<LinearSys>(flow_);
    const int64_t n = sys.dim();
    c_ = c ? std::move(*c) : sys.A().zeros_like({n, 1});
    CORA_CHECK(sys.A().ndim() == 2 && !sys.C(), Error::wrong_input_in_constructor, [] {
        return std::string("Location: the flow must be one system without an output map C");
    });
    CORA_CHECK(c_.shape() == std::vector<int64_t>({n, 1}), Error::wrong_input_in_constructor, [&] {
        return "Location: c must be a column (" + std::to_string(n) + ", 1), got the shape " +
               shape_text(c_.shape());
    });
    aux_validate(*this);
}

Location::Location(std::string name, Polytope invariant, std::vector<Transition> transitions,
                   LinParamSys flow)
    : name_(std::move(name)), invariant_(std::move(invariant)),
      transitions_(std::move(transitions)), flow_(std::move(flow)),
      c_(std::get<LinParamSys>(flow_).c()) {
    detail::Entry entry;
    // the system checked its c; the invariant and the guards remain
    aux_validate(*this);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
