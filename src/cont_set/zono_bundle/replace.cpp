// replace - a zonotope bundle with one member replaced, as CORA's zonoBundle.replace
//
// Syntax:   ZonoBundle R = bundle.replace(index, Z);
// Inputs:   index - the member to replace (0-based)
//           Z - the new member, of the dimension and batch shape of the bundle
// Outputs:  R - the bundle with member `index` replaced by Z
// See also: ZonoBundle::ZonoBundle
//
// CORA: contSet/@zonoBundle/replace.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ZonoBundle ZonoBundle::replace(int64_t index, const Zonotope &Z) const {
    detail::Entry entry;
    CORA_CHECK(index >= 0 && index < parallel_sets(), Error::wrong_value, [&] {
        return "replace: the index " + std::to_string(index) + " is not in [0, " +
               std::to_string(parallel_sets()) + ")";
    });
    const detail::Refusal refusal = priv_operand_refusal(*this, Z.dim(), Z.center(), "replace");
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    std::vector<Zonotope> members = Z_;
    members[std::size_t(index)] = Z;
    return ZonoBundle(std::move(members));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
