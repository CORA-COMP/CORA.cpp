// hybrid_automaton - lean::HybridAutomaton, the reach loop of a hybrid automaton by CORALean
//
// Sends `hybridAutomaton.reach` with the locations as a list of objects (the flow's fields at the
// top level of each, as the linear-system ops, then invariant, timeStep and transitions) and
// reads the branches (README, "hybridAutomaton.").
//
// Syntax:   auto branches = lean::HybridAutomaton::reach(locations, 0, R0, 0, t_final, 10, 50);
// Inputs:   see hybrid_automaton.h
// Outputs:  see hybrid_automaton.h
// See also: request.h
//
// CORA.cpp only: the wrapper of the oracle, CORA has no such class
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/hybrid_automaton.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A location as the oracle reads it; its dtype is that of its first lean member.
Wire aux_location(const HybridLocation &loc) {
    Json json = Json::object();
    std::string dtype;
    const auto put = [&](const char *key, const Wire &w) {
        json.set(key, w.json);
        if (dtype.empty()) dtype = w.dtype;
    };
    put("A", Tensor::from(loc.flow.A));
    if (loc.flow.B) put("B", Tensor::from(*loc.flow.B));
    if (loc.flow.c) put("c", Tensor::from(*loc.flow.c));
    if (loc.flow.U) put("U", Zonotope(*loc.flow.U));
    put("invariant", Polytope(loc.invariant));
    put("timeStep", Wire(loc.time_step));
    // the transitions: guard, reset x -> W x + w and target
    Json transitions = Json::array();
    for (const HybridTransition &t : loc.transitions) {
        Json tr = Json::object();
        tr.set("guard", Wire(Polytope(t.guard)).json)
            .set("W", Wire(Tensor::from(t.W)).json)
            .set("w", Wire(Tensor::from(t.w)).json)
            .set("target", Json(t.target));
        transitions.push(std::move(tr));
    }
    json.set("transitions", std::move(transitions));
    return {std::move(json), dtype};
}

/// Fails with not_supported unless `ok`.
void aux_require(bool ok, const char *message) {
    if (!ok) CORA_FAIL(Error::not_supported, message);
}

/// The time m*2^e of the oracle, rounded to the nearest double: a time k dt of a step size that is
/// not a power of two is exact there and has more than 53 bits.
double aux_time(const std::string &text) {
    const std::size_t star = text.find("*2^");
    if (star == std::string::npos)
        CORA_FAIL(Error::wrong_value, "lean: expected a time m*2^e, got '" + text + "'");
    return static_cast<double>(
        std::ldexp(std::stold(text.substr(0, star)), std::stoi(text.substr(star + 3))));
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The locations -----------------------------------------------------------------------------------

std::vector<HybridBranch> HybridAutomaton::reach(const std::vector<HybridLocation> &locations,
                                                 int64_t start_loc, const cora::Zonotope &R0,
                                                 double t_start, double t_final,
                                                 int64_t taylor_terms, int64_t zonotope_order,
                                                 std::optional<int64_t> final_loc,
                                                 bool check_flow) {
    // the locations travel as a list of objects
    Json list = Json::array();
    std::string dtype;
    for (const HybridLocation &loc : locations) {
        const Wire w = aux_location(loc);
        if (dtype.empty()) dtype = w.dtype;
        list.push(w.json);
    }
    Json enclose = Json::array();
    enclose.push(Json("box"));
    std::vector<Operand> operands = {{"locations", Wire(std::move(list), dtype)},
                                     {"startLoc", start_loc},
                                     {"R0", Zonotope(R0)},
                                     {"tStart", t_start},
                                     {"tFinal", t_final},
                                     {"taylorTerms", taylor_terms},
                                     {"zonotopeOrder", zonotope_order},
                                     {"guardIntersect", "zonoGirard"},
                                     {"enclose", Wire(std::move(enclose), "")}};
    if (final_loc) operands.push_back({"finalLoc", *final_loc});
    if (check_flow) operands.push_back({"checkFlow", true});
    // one object per branch
    std::vector<HybridBranch> branches;
    for (const Reply &b : ask("hybridAutomaton.reach", operands).list("branches")) {
        std::vector<cora::Zonotope> sets;
        for (const Zonotope &Z : b.get<std::vector<Zonotope>>("timeInterval"))
            sets.push_back(Z.gather().first);
        const std::vector<std::string> t = b.get<std::vector<std::string>>("tStart");
        branches.push_back({b.get<int64_t>("loc"),
                            b.get<std::optional<int64_t>>("parent"),
                            std::move(sets),
                            {aux_time(t.at(0)), aux_time(t.at(1))}});
    }
    return branches;
}

// The automaton -----------------------------------------------------------------------------------

std::vector<HybridBranch> HybridAutomaton::reach(const cora::HybridAutomaton &HA,
                                                 const cora::HybridReachProblem &problem,
                                                 const cora::alg::HybridStandard &alg) {
    // the problem must be one the op serves
    const auto &given = HA.locations();
    aux_require(problem.initial_set && problem.final_locations.size() <= 1 &&
                    (problem.input_set.empty() || problem.input_set.size() == 1 ||
                     problem.input_set.size() == given.size()),
                "lean reach: a zonotope start set, at most one final location and one input "
                "set per location are served");
    // the locations with the input set and the step size of each
    std::vector<HybridLocation> locations;
    for (std::size_t i = 0; i < given.size(); ++i) {
        const cora::Location &loc = given[i];
        const auto *flow = std::get_if<cora::LinearSys>(&loc.flow());
        const auto *invariant = std::get_if<cora::Polytope>(&loc.invariant());
        aux_require(flow && invariant, "lean reach: linear flows and polytope invariants");
        HybridLocation out{{flow->A(), loc.B(), loc.c(), std::nullopt}, *invariant, {}, 0};
        if (!problem.input_set.empty())
            out.flow.U = problem.input_set[problem.input_set.size() == 1 ? 0 : i];
        out.time_step = alg.time_step[alg.time_step.size() == 1 ? 0 : i];
        // its transitions
        for (const cora::Transition &t : loc.transitions()) {
            const auto *guard = std::get_if<cora::Polytope>(&t.guard());
            aux_require(guard && !t.reset().B(),
                        "lean reach: polytope guards, resets x -> W x + w");
            const cora::Tensor w = t.reset().c() ? *t.reset().c() : loc.c().zeros_like();
            out.transitions.push_back({*guard, t.reset().A(), w, t.target()});
        }
        locations.push_back(std::move(out));
    }
    // the final location and the call
    std::optional<int64_t> final_loc;
    if (!problem.final_locations.empty()) final_loc = problem.final_locations.front();
    return reach(locations, problem.start_location, *problem.initial_set, problem.t_start,
                 problem.t_final, alg.taylor_terms, alg.zonotope_order, final_loc, alg.check_flow);
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
