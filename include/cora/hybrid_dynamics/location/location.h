// location - a location of a hybrid automaton, as CORA's location
//
// The state flows by x' = A x + B u + c (a LinearSys and its constant input c, or a LinParamSys
// with its own c), or by x' = f(x, u) (a NonlinearSys), while it is in the invariant, and it may
// leave by the transitions whose guards it reaches. CORA's linearSys holds c itself; LinearSys
// has no constant input, so the location holds it (docs/cora-differences.md).
//
// Syntax:     Location loc("S1", invariant, {trans1, trans2}, LinearSys(A, B), c);
//             Location loc("S1", invariant, {trans1}, LinParamSys(A, B, c));
//             Location loc("S1", invariant, {trans1}, NonlinearSys(f, n));
// Operations: the constructor checks its input (location.cpp); reach and the guard intersections
//             are private to the hybrid automaton (private/)
// See also:   cora/hybrid_dynamics/hybrid_automaton/hybrid_automaton.h, transition.h
//
// CORA: hybridDynamics/@location/location.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/lin_param_sys/lin_param_sys.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/hybrid_dynamics/transition/transition.h"

#include <optional>
#include <string>
#include <variant>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The continuous dynamics of a location.
using Flow = std::variant<LinearSys, LinParamSys, NonlinearSys>;

class Location {
  public:
    /// The flow x' = A x + B u + c of `flow` (without an output map C) and the constant input c
    /// (n, 1; none: zero) inside `invariant`, left by `transitions`.
    Location(std::string name, Region invariant, std::vector<Transition> transitions,
             LinearSys flow, std::optional<Tensor> c = std::nullopt);

    /// The flow x' = A x + B u + c of a linear parametric system, with its own c.
    Location(std::string name, Region invariant, std::vector<Transition> transitions,
             LinParamSys flow);

    /// The nonlinear flow x' = f(x, u); c is zero.
    Location(std::string name, Region invariant, std::vector<Transition> transitions,
             NonlinearSys flow);

    const std::string &name() const noexcept { return name_; }
    const Region &invariant() const noexcept { return invariant_; }
    const std::vector<Transition> &transitions() const noexcept { return transitions_; }
    const Flow &flow() const noexcept { return flow_; }
    const Tensor &c() const noexcept { return c_; }

    /// The input matrix B of a linear flow, if it has one (a nonlinear flow has none).
    const std::optional<Tensor> &B() const;

    /// The number of states.
    int64_t dim() const {
        return std::visit([](const auto &f) { return f.dim(); }, flow_);
    }

    /// The number of inputs of the flow (the columns of B, 0 without B; the inputs of f).
    int64_t input_dim() const;

  private:
    std::string name_;
    Region invariant_;
    std::vector<Transition> transitions_;
    Flow flow_;
    Tensor c_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
