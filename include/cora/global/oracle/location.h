// location - lean::Location and lean::Transition, the hybrid-automaton steps computed by CORALean
//
// The steps of a location that CORA.cpp's hybrid reach uses, each the oracle's op of that name:
// the guards a set may meet (pot_int), the bases of the enclosure (calc_basis), the sets whose flow
// may leave through a hyperplane (check_flow), the box of the cut by a hyperplane (the dual bound
// of zonoGirard) and the reset image (jump). The flow is x' = A x + B u + c with the input set U.
//
// Syntax:     auto hits = lean::Location::pot_int(guards, targets, final_loc, R);
//             lean::ZonoBundle zB = lean::Location::guard_intersect_zono_girard(Bs, R0, R, n, g);
//             lean::Zonotope Z = lean::Transition::jump(W, w, Rg);
// Operations: pot_int, calc_basis, check_flow, guard_intersect_zono_girard, jump
// Outputs:    the oracle's answers, the sets as lean sets with their error boxes
// See also:   lean/zonotope.h, lean/polytope.h, hybrid_dynamics/location/location.h
//
// CORA.cpp only: the wrapper of the oracle, CORA has no such class
// Authors:       Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/oracle/request.h"
#include "cora/global/oracle/zono_bundle.h"

#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

/// The flow of a location: x' = A x + B u + c, u in U (B, c and U are optional).
struct Flow {
    cora::Tensor A;
    std::optional<cora::Tensor> B, c;
    std::optional<cora::Zonotope> U;
};

struct Location {
    /// The (transition, set) pairs, 0-based, whose guard may meet the set; the transitions into
    /// final_loc are skipped (`location.potInt`).
    static std::vector<std::vector<int64_t>> pot_int(const std::vector<cora::Polytope> &guards,
                                                     const std::vector<int64_t> &targets,
                                                     int64_t final_loc,
                                                     const std::vector<cora::Zonotope> &R);

    /// One basis per entry of `enclose`; only "box" has a float definition (`location.calcBasis`).
    static std::vector<cora::Tensor> calc_basis(int64_t n, const std::vector<std::string> &enclose);

    /// The indices of the sets that are not certified to have no outward flow through the
    /// hyperplane normal'x = d (`location.checkFlow`).
    static std::vector<int64_t> check_flow(const Flow &flow, const cora::Tensor &normal, double d,
                                           const std::vector<cora::Zonotope> &R);

    /// The bundle with one box per basis of the cut of R0 and R by the hyperplane normal'x = gamma
    /// (`location.guardIntersect_zonoGirard`).
    static ZonoBundle guard_intersect_zono_girard(const std::vector<cora::Tensor> &Bs,
                                                  const cora::Zonotope &R0,
                                                  const std::vector<cora::Zonotope> &R,
                                                  const cora::Tensor &normal, double gamma);
};

struct Transition {
    /// W Rg + w (`transition.jump`).
    static Zonotope jump(const cora::Tensor &W, const cora::Tensor &w, const cora::Zonotope &Rg);
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
