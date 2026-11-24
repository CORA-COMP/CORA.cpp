// assign_identifiers - the formula with its non-temporal parts numbered
//
// Syntax:   [res, list] = phi.assign_identifiers();
// Inputs:   none
// Outputs:  res - the formula, each maximal non-temporal sub-formula carrying its id 1, 2, ...
//           list - those sub-formulas in the order of the ids
// See also: stl.h
//
// CORA: specification/@stl/assignIdentifiers.m
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

/// aux_recursive: the formula with the ids following those in list; each numbered part is added.
Stl aux_number(const Stl &phi, std::vector<Stl> &list) {
    if (!phi.temporal()) {
        const Stl res = phi.with_id(static_cast<int64_t>(list.size()) + 1);
        list.push_back(res);
        return res;
    }
    const StlKind k = phi.kind();
    if (k == StlKind::next) return aux_number(phi.lhs(), list).next(phi.from());
    if (k == StlKind::globally) return aux_number(phi.lhs(), list).globally(phi.interval());
    if (k == StlKind::finally) return aux_number(phi.lhs(), list).finally(phi.interval());
    if (k == StlKind::not_) return aux_number(phi.lhs(), list).not_();
    // two operands: the left one is numbered first
    const Stl l = aux_number(phi.lhs(), list);
    const Stl r = aux_number(phi.rhs(), list);
    if (k == StlKind::until) return l.until(r, phi.interval());
    if (k == StlKind::release) return l.release(r, phi.interval());
    return k == StlKind::and_ ? l.and_(r) : l.or_(r);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<Stl, std::vector<Stl>> Stl::assign_identifiers() const {
    detail::Entry entry;
    std::vector<Stl> list;
    Stl res = aux_number(*this, list);
    return {res, list};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
