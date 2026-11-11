// location - lean::Location and lean::Transition, the hybrid-automaton steps computed by CORALean
//
// Each method sends the op of the same name with the flow's fields at the top level of the request
// (A, B, c, U) and reads the answer of the oracle (README, "location. and transition.").
//
// Syntax:   auto hits = lean::Location::pot_int(guards, targets, final_loc, R);
// Inputs:   see location.h
// Outputs:  see location.h
// See also: request.h
//
// CORA.cpp only: the wrapper of the oracle, CORA has no such class
// Authors:       Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/location.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The operands of a flow: its fields that are given.
std::vector<Operand> aux_flow(const Flow &flow) {
    std::vector<Operand> operands{{"A", Tensor::from(flow.A)}};
    if (flow.B) operands.push_back({"B", Tensor::from(*flow.B)});
    if (flow.c) operands.push_back({"c", Tensor::from(*flow.c)});
    if (flow.U) operands.push_back({"U", Zonotope(*flow.U)});
    return operands;
}

/// The lean zonotopes of cora zonotopes.
std::vector<Zonotope> aux_lean(const std::vector<cora::Zonotope> &R) {
    return std::vector<Zonotope>(R.begin(), R.end());
}

/// The members every polytope guard op takes: the bases, the sets, the guard and the box method.
std::vector<Operand> aux_cut_operands(const std::vector<cora::Tensor> &Bs, const cora::Zonotope &R0,
                                      const std::vector<cora::Zonotope> &R,
                                      const cora::Polytope &guard, const std::string &box,
                                      int64_t fm_passes) {
    // the bases cross as lean tensors, then the sets and the guard
    std::vector<Tensor> bases;
    bases.reserve(Bs.size());
    for (const cora::Tensor &B : Bs) bases.push_back(Tensor::from(B));
    std::vector<Operand> operands = {{"Bs", bases},
                                     {"R0", Zonotope(R0)},
                                     {"R", aux_lean(R)},
                                     {"guard", Polytope(guard)},
                                     {"box", box}};
    // the box "cert" has no passes
    if (box != "cert") operands.push_back({"fmPasses", fm_passes});
    return operands;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<std::vector<int64_t>> Location::pot_int(const std::vector<cora::Polytope> &guards,
                                                    const std::vector<int64_t> &targets,
                                                    int64_t final_loc,
                                                    const std::vector<cora::Zonotope> &R) {
    // the guards and the sets travel as lists; the answer is a list of index pairs
    std::vector<Polytope> lean_guards(guards.begin(), guards.end());
    const Reply r = ask("location.potInt", {{"guards", lean_guards},
                                            {"targets", targets},
                                            {"finalLoc", final_loc},
                                            {"R", aux_lean(R)}});
    return r.get<std::vector<std::vector<int64_t>>>("hits");
}

// The groups come as objects; each is flattened to its transition followed by its sets.
std::vector<std::vector<int64_t>>
Location::guard_groups(const std::vector<std::vector<int64_t>> &hits) {
    const Reply r = ask("location.guardGroups", {{"hits", hits}});
    // one group per object: the transition, then its sets
    std::vector<std::vector<int64_t>> out;
    for (const Reply &g : r.list("groups")) {
        std::vector<int64_t> group{g.get<int64_t>("transition")};
        for (int64_t j : g.get<std::vector<int64_t>>("sets")) group.push_back(j);
        out.push_back(std::move(group));
    }
    return out;
}

std::vector<cora::Tensor> Location::calc_basis(int64_t n, const std::vector<std::string> &enclose) {
    Json names = Json::array();
    for (const std::string &s : enclose) names.push(Json(s));
    const Reply r = ask("location.calcBasis", {{"n", n}, {"enclose", Wire(std::move(names), "")}});
    std::vector<cora::Tensor> bases;
    for (const Tensor &B : r.get<std::vector<Tensor>>("B")) bases.push_back(B.gather());
    return bases;
}

std::vector<int64_t> Location::check_flow(const Flow &flow, const cora::Tensor &normal, double d,
                                          const std::vector<cora::Zonotope> &R) {
    // the hyperplane is {normal, d}
    Json hyperplane = Json::object();
    hyperplane.set("normal", Tensor::from(normal).to_json()).set("d", Wire(d).json);
    std::vector<Operand> operands = aux_flow(flow);
    operands.push_back({"hyperplane", Wire(std::move(hyperplane), Tensor::from(normal).dtype())});
    operands.push_back({"R", aux_lean(R)});
    return ask("location.checkFlow", operands).get<std::vector<int64_t>>("keep");
}

// The bases, the sets, the normal and the offset of the hyperplane.
ZonoBundle Location::guard_intersect_zono_girard(const std::vector<cora::Tensor> &Bs,
                                                 const cora::Zonotope &R0,
                                                 const std::vector<cora::Zonotope> &R,
                                                 const cora::Tensor &normal, double gamma) {
    std::vector<Tensor> bases;
    bases.reserve(Bs.size());
    for (const cora::Tensor &B : Bs) bases.push_back(Tensor::from(B));
    return ask("location.guardIntersect_zonoGirard", {{"Bs", bases},
                                                      {"R0", Zonotope(R0)},
                                                      {"R", aux_lean(R)},
                                                      {"normal", Tensor::from(normal)},
                                                      {"gamma", gamma}})
        .get<ZonoBundle>("zB");
}

// The polytope guards ------------------------------------------------------------------------


// The bases, the sets, the polytope guard, the box method and the passes of the box.
ZonoBundle Location::guard_intersect_nondet_guard(const std::vector<cora::Tensor> &Bs,
                                                  const cora::Zonotope &R0,
                                                  const std::vector<cora::Zonotope> &R,
                                                  const cora::Polytope &guard, int64_t fm_passes,
                                                  const std::string &box) {
    return ask("location.guardIntersect_nondetGuard",
               aux_cut_operands(Bs, R0, R, guard, box, fm_passes))
        .get<ZonoBundle>("zB");
}

ZonoBundle Location::guard_intersect_con_zonotope(const std::vector<cora::Tensor> &Bs,
                                                  const cora::Zonotope &R0,
                                                  const std::vector<cora::Zonotope> &R,
                                                  const cora::Polytope &guard, int64_t guard_order,
                                                  int64_t fm_passes, const std::string &box) {
    // the guard order is the reduction of the sets before they are cut
    std::vector<Operand> operands = aux_cut_operands(Bs, R0, R, guard, box, fm_passes);
    operands.push_back({"guardOrder", guard_order});
    return ask("location.guardIntersect_conZonotope", operands).get<ZonoBundle>("zB");
}

// The reset ---------------------------------------------------------------------------------

// The reset image of the guard-cut set.
Zonotope Transition::jump(const cora::Tensor &W, const cora::Tensor &w, const cora::Zonotope &Rg) {
    return ask("transition.jump",
               {{"W", Tensor::from(W)}, {"w", Tensor::from(w)}, {"Rg", Zonotope(Rg)}})
        .get<Zonotope>("Z");
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
