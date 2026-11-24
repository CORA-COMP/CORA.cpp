// isequal - whether specifications are equal, as CORA's isequal
//
// Syntax:   res = spec1.isequal(spec2, tol);   res = isequal(specs1, specs2, tol);
// Inputs:   tol - tolerance of the times and the sets
// Outputs:  res - whether they are equal (a list: up to the order)
// See also: project
//
// CORA: specification/@specification/isequal.m
// CORA: specification/@specification/eq.m
// CORA: specification/@specification/ne.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "cora/global/within_tol.h"
#include "cora/specification/specification.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Specification::isequal(const Specification &other, Tol tol) const {
    detail::Entry entry;
    resolve_tol(entry, tol, "isequal"); // refuses a negative tolerance
    CORA_CHECK(type_ != SpecType::custom && other.type_ != SpecType::custom, Error::not_supported,
               [] { return std::string("isequal: custom predicates cannot be compared"); });
    // two logic specifications are the same when their formulas are, and the window and locations
    if (type_ == SpecType::logic && other.type_ == SpecType::logic)
        return formula_->isequal(*other.formula_) && time_ == other.time_ &&
               locations_ == other.locations_;
    entry.release(); // the sets check their operands
    if (type_ != other.type_ || time_.has_value() != other.time_.has_value()) return false;
    if (time_ && !(within_tol(time_->first, other.time_->first, tol) &&
                   within_tol(time_->second, other.time_->second, tol)))
        return false;
    if (locations_ != other.locations_) return false;
    return cora::isequal(set_->get(), other.set_->get(), tol);
}

bool isequal(const std::vector<Specification> &specs1, const std::vector<Specification> &specs2,
             Tol tol) {
    detail::Entry entry;
    entry.release(); // each pair checks its own operands
    if (specs1.size() != specs2.size()) return false;
    // every specification of the first list has a match of its own in the second
    std::vector<bool> matched(specs2.size(), false);
    for (const Specification &spec : specs1) {
        bool found = false;
        for (std::size_t j = 0; j < specs2.size() && !found; ++j)
            if (!matched[j] && spec.isequal(specs2[j], tol)) matched[j] = found = true;
        if (!found) return false;
    }
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
