// get_clauses - the conjuncts or disjuncts of a formula
//
// Syntax:   list = phi.get_clauses("cnf");
// Inputs:   type - "cnf" (split at &) or "dnf" (split at |)
// Outputs:  list - the clauses, left to right
// See also: stl.h
//
// CORA: specification/@stl/getClauses.m
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

/// The sub-formulas below the nodes of the kind, left to right (aux_getClausesCNF / DNF).
void aux_split(const Stl &phi, StlKind kind, std::vector<Stl> &list) {
    if (phi.kind() != kind) {
        list.push_back(phi);
        return;
    }
    aux_split(phi.lhs(), kind, list);
    aux_split(phi.rhs(), kind, list);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Stl> Stl::get_clauses(const std::string &type) const {
    detail::Entry entry;
    CORA_CHECK(type == "cnf" || type == "dnf", Error::wrong_value,
               [] { return std::string("get_clauses: the type must be \"cnf\" or \"dnf\""); });
    std::vector<Stl> list;
    aux_split(*this, type == "cnf" ? StlKind::and_ : StlKind::or_, list);
    return list;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
