// representsa - whether a box is also a set of another type, as CORA's interval.representsa_
//
// Syntax:   bool res = I.representsa(type);   bool res = I.representsa(type, tol);
// Inputs:   type - CORA's name of a set type ("origin", "point", "emptySet", "fullspace",
//           "interval", "zonotope", "capsule", ...);  tol - width that counts as zero (absolute,
//           or relative to the bound, as within_tol); default: the repository tolerance
// Outputs:  res - whether the set is exactly such a set, for a batch whether all members are. A
//           matrix of intervals is supported for "emptySet" and "origin" only. The converted set
//           (CORA's second output) is not returned: use the conversions. "polygon" is refused
//           (Error::not_supported), CORA's switch has no case for it. "capsule" and "ellipsoid":
//           at most one coordinate has a width (CORA counts the zero widths instead, and
//           accepts every box with fewer than two flat coordinates)
// See also: is_full_dim, is_bounded
//
// CORA: contSet/@interval/representsa_.m
// CORA: contSet/@contSet/representsa_emptyObject.m
// CORA: contSet/@contSet/representsa.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <algorithm>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The types of CORA's representsa.
const char *const kTypes[] = {"capsule", "conPolyZono", "conHyperplane", "conZonotope",
                              "ellipsoid", "halfspace", "interval", "levelSet", "polygon",
                              "polytope", "polyZonotope", "probZonotope", "zonoBundle",
                              "zonotope", "origin", "point", "hyperplane", "parallelotope",
                              "convexSet", "emptySet", "fullspace"};

/// The types that an empty interval does not represent (CORA's representsa_emptyObject).
bool aux_empty_is_not(const std::string &type) {
    return type == "origin" || type == "point" || type == "parallelotope" || type == "fullspace";
}

/// Whether the mask holds everywhere (for every member of a batch).
bool aux_all(const Tensor &mask) { return mask.numel() == 0 || mask.min().item() != 0; }

/// The number of entries of the mask that hold in each member of a batch.
Tensor aux_count(const Tensor &mask) { return mask.sum(-1).sum(-1); }

/// Whether the count of each member satisfies the comparison with the number s.
bool aux_count_is(const Tensor &mask, Tensor::Binary op, double s) {
    return aux_all(aux_count(mask).scalar_op(op, s));
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Interval::representsa(const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    CORA_CHECK(std::find(std::begin(kTypes), std::end(kTypes), type) != std::end(kTypes),
               Error::wrong_value,
               [&] { return "representsa: '" + type + "' is not a type of set that CORA knows"; });
    CORA_CHECK(type != "polygon", Error::not_supported,
               [] { return std::string("representsa: the type 'polygon' is not supported"); });
    const double tol = resolve_tol(entry, tol_arg, inf_.dtype(), "representsa");
    if (is_empty()) return !aux_empty_is_not(type);
    CORA_CHECK(type == "emptySet" || type == "origin" || inf_.size(-1) == 1 || inf_.size(-2) == 1,
               Error::not_supported, [] {
                   return std::string("representsa only supports vector interval objects "
                                      "(except type = 'emptySet', 'origin')");
               });
    const double kInf = std::numeric_limits<double>::infinity();
    const Tensor zero = inf_.zeros_like();
    const Tensor flat = within_tol(rad(), zero, tol); // the coordinates of width zero
    const Tensor sloped = flat.scalar_op(Tensor::Binary::equal, 0);
    const bool is_point = aux_all(flat), one_dim = dim() == 1;
    if (type == "origin")
        return aux_all(within_tol(inf_, zero, tol)) && aux_all(within_tol(sup_, zero, tol));
    if (type == "point") return is_point;
    if (type == "capsule" || type == "ellipsoid")
        return one_dim || is_point || aux_count_is(sloped, Tensor::Binary::less_equal, 1);
    if (type == "conHyperplane")
        return one_dim || is_point || aux_count_is(flat, Tensor::Binary::greater_equal, 1);
    if (type == "halfspace") {
        const Tensor infinite = inf_.isinf().sum(-1).sum(-1) + sup_.isinf().sum(-1).sum(-1);
        return aux_all(infinite.scalar_op(Tensor::Binary::equal, 2.0 * double(dim()) - 1));
    }
    if (type == "hyperplane") return aux_count_is(flat, Tensor::Binary::equal, 1);
    if (type == "fullspace")
        return aux_all(inf_.equal(inf_.full_like(inf_.shape(), -kInf))) &&
               aux_all(sup_.equal(sup_.full_like(sup_.shape(), kInf)));
    // the sets that every box is, and the one that none is
    return type != "probZonotope" && type != "emptySet";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
