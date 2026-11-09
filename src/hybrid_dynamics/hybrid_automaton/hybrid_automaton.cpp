// hybrid_automaton - the constructor of a hybrid automaton, as CORA's hybridAutomaton(name, loc)
//
// Syntax:   HybridAutomaton HA(name, locations);
// Inputs:   name - the name of the automaton
//           locations - its locations, numbered from 0 in this order
// Outputs:  HA - the automaton
// See also: Location, reach
//
// CORA: hybridDynamics/@hybridAutomaton/hybridAutomaton.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/hybrid_dynamics/hybrid_automaton/hybrid_automaton.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

HybridAutomaton::HybridAutomaton(std::string name, std::vector<Location> locations)
    : name_(std::move(name)), locations_(std::move(locations)) {
    detail::Entry entry;
    CORA_CHECK(!locations_.empty(), Error::wrong_input_in_constructor,
               [] { return std::string("HybridAutomaton: give at least one location"); });
    // the locations share the state, and every target is one of them
    const int64_t n = locations_.front().dim(), count = int64_t(locations_.size());
    for (const Location &loc : locations_) {
        CORA_CHECK(loc.dim() == n, Error::wrong_input_in_constructor, [&] {
            return "HybridAutomaton: the location " + loc.name() + " has " +
                   std::to_string(loc.dim()) + " states, the first " + std::to_string(n);
        });
        for (const Transition &t : loc.transitions())
            CORA_CHECK(t.target() < count, Error::wrong_input_in_constructor, [&] {
                return "HybridAutomaton: a transition of " + loc.name() + " targets the location " +
                       std::to_string(t.target()) + ", but there are " + std::to_string(count);
            });
    }
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
