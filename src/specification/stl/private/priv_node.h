// priv_node - the nodes of STL expressions and formulas, and the access of the operations to them
//
// CORA.cpp only: the representation of CORA's stl objects as trees of immutable nodes
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/specification/stl.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The node of an arithmetic expression: a variable, a number, or an operation on children.
struct StlExpr::Node {
    StlExprKind kind = StlExprKind::number;
    double value = 0;
    std::string name;
    int64_t index = 0;
    int64_t exponent = 0;
    std::shared_ptr<const Node> l, r;
};

/// The node of a formula: its kind, the comparison with its sides, the children, the window, and
/// whether a temporal operator occurs below.
struct Stl::Node {
    StlKind kind = StlKind::true_;
    StlCmp op = StlCmp::lt;
    StlExpr e1, e2;
    std::shared_ptr<const Node> l, r;
    StlInterval interval;
    bool temporal = false;
    int64_t id = 0;
};

/// The access of the operations to the private nodes of the two classes.
struct StlAccess {
    static std::shared_ptr<Stl::Node> make(StlKind kind) {
        auto node = std::make_shared<Stl::Node>();
        node->kind = kind;
        return node;
    }
    static Stl wrap(std::shared_ptr<const Stl::Node> node) { return Stl(std::move(node)); }
    static std::shared_ptr<const Stl::Node> node(const Stl &phi) { return phi.node_; }
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
