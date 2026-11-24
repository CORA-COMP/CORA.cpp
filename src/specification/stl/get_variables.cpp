// get_variables - the variables of a formula
//
// Syntax:   names = phi.get_variables();
// Inputs:   none
// Outputs:  names - the names of the variables occurring, shortest first, then alphabetically
// See also: stl.h
//
// CORA: specification/@stl/getVariables.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The names of the variables below the node, with repetitions.
void aux_collect(const Stl &phi, std::vector<std::string> &names) {
    const StlKind k = phi.kind();
    if (k == StlKind::true_ || k == StlKind::false_) return;
    // a comparison names the variables of its sides, the operators those of their operands
    if (k == StlKind::comparison) {
        for (const StlExpr &e : {phi.expr_lhs(), phi.expr_rhs()})
            for (auto &name : e.variable_names()) names.push_back(name);
        return;
    }
    aux_collect(phi.lhs(), names);
    const bool unary =
        k == StlKind::not_ || k == StlKind::next || k == StlKind::finally || k == StlKind::globally;
    if (!unary) aux_collect(phi.rhs(), names);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<std::string> Stl::get_variables() const {
    detail::Entry entry;
    std::vector<std::string> names;
    aux_collect(*this, names);
    stl_sort_names(names);
    return names;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
