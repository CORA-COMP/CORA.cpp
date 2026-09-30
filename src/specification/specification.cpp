// specification - the constructors of a specification, as CORA's specification(set, type)
//
// Syntax:   Specification::safeSet({h1, h2, ...});   Specification::unsafeSet({h1, h2, ...});
// Inputs:   halfspaces {x | a'x <= b}
// Outputs:  the specification; an unsafe set of several halfspaces is their intersection
// See also: check

#include "specification/specification.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Specification Specification::safeSet(std::vector<Halfspace> halfspaces) {
    if (halfspaces.empty()) throw std::invalid_argument("Specification: no halfspace");
    return {SpecType::SafeSet, std::move(halfspaces)};
}

Specification Specification::unsafeSet(std::vector<Halfspace> halfspaces) {
    if (halfspaces.empty()) throw std::invalid_argument("Specification: no halfspace");
    return {SpecType::UnsafeSet, std::move(halfspaces)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
