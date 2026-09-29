// specification - the constructors of a specification, as CORA's specification(set, type)
//
// Syntax:   Specification::safeSet({h1, h2, ...});   Specification::unsafeSet(h);
// Inputs:   halfspaces {x | a'x <= b}
// Outputs:  the specification; a safe set needs at least one halfspace
// See also: check

#include "specification/specification.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ===========================================  MAIN  =========================================== //

Specification Specification::safeSet(std::vector<Halfspace> halfspaces) {
    if (halfspaces.empty()) throw std::invalid_argument("Specification: no halfspace");
    return {SpecType::SafeSet, std::move(halfspaces)};
}

Specification Specification::unsafeSet(const Halfspace &halfspace) {
    return {SpecType::UnsafeSet, {halfspace}};
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
