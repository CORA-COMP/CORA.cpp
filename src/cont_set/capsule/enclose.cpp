// enclose - a capsule that encloses a capsule and a second one, as CORA's capsule.enclose
//
// Syntax:   Cres = C.enclose(S);   Cres = C.enclose(M, Cplus);
// Inputs:   S - capsule of the dimension of C;  M, Cplus - a matrix and a capsule for the second
//           form
// Outputs:  Cres - the convex hull of C and S. The second form is refused: CORA computes the hull
//           of C and M * Cplus and never adds Cplus, so it misses the documented M * C + Cplus
//           (CORALean's enclose_cora_counterexample); the sum waits for capsule plus
// See also: convex_hull
//
// CORA: contSet/@capsule/enclose.m

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/capsule/capsule.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Capsule Capsule::enclose(const Capsule &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "enclose: the capsules differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    return convex_hull(S);
}

Capsule Capsule::enclose(const Tensor &M, const Capsule &Cplus) const {
    detail::Entry entry;
    // M maps the space of Cplus into that of C
    CORA_CHECK(M.defined() && M.ndim() >= 2 && M.size(-2) == dim() && M.size(-1) == Cplus.dim(),
               Error::dimension_mismatch, [&] {
                   return "enclose: M must be (" + std::to_string(dim()) + ", " +
                          std::to_string(Cplus.dim()) + "), the dimensions of C and Cplus";
               });
    CORA_FAIL(Error::not_supported,
              "enclose(M, Cplus) is not supported: CORA returns the hull of C and M * Cplus, which "
              "does not enclose the documented M * C + Cplus, and the sum of capsules is not "
              "ported yet");
}

Set Capsule::enclose_impl(const ContSet &S2) const {
    if (S2.class_id() != ClassId::capsule) priv_noops("enclose", *this, &S2);
    return enclose(static_cast<const Capsule &>(S2));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
