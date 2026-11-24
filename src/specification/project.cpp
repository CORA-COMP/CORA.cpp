// project - the specification on some dimensions, as CORA's project
//
// Syntax:   spec2 = spec.project(dims);   specs2 = project(specs, dims);
// Inputs:   dims - the dimensions to keep, 0-based
// Outputs:  spec2 - the specification with the projected set; time and locations stay
// See also: isequal
//
// CORA: specification/@specification/project.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/specification.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Specification Specification::project(const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    CORA_CHECK(type_ != SpecType::custom, Error::not_supported, [] {
        return std::string("project: the projection of a custom specification is not supported");
    });
    entry.release(); // the set checks its dimensions
    Specification out = *this;
    out.set_ = set_->get().project(dims);
    return out;
}

std::vector<Specification> project(const std::vector<Specification> &specs,
                                   const std::vector<int64_t> &dims) {
    detail::Entry entry;
    entry.release();
    std::vector<Specification> out;
    out.reserve(specs.size());
    for (const Specification &spec : specs) out.push_back(spec.project(dims));
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
