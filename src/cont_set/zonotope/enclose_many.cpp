// enclose_many - a box, oriented along a direction, around many zonotopes
//
// Syntax:   [Zenclose, rot_inv] = Zonotope::enclose_many(Zs, direction);
// Inputs:   Zs - zonotopes of one dimension n;  direction - a nonzero column (n, 1)
// Outputs:  Zenclose - the smallest box in the frame whose first axis is the direction (the other
//           axes are standard axes, the one most aligned with the direction left out), as a
//           zonotope, containing every Z;  rot_inv - the inverse of that frame (n, n).
//           CORA reduces each Z to a parallelotope with reduce(Z, 'parallelpiped'), a method
//           that does not exist and makes CORA fail; CORA.cpp reduces to order 1 (Girard)
// See also: enclose, reduce
//
// CORA: contSet/@zonotope/encloseMany.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<Zonotope, Tensor> Zonotope::enclose_many(const std::vector<Zonotope> &Zs,
                                                   const Tensor &direction) {
    detail::Entry entry;
    CORA_CHECK(!Zs.empty(), Error::wrong_value,
               [] { return std::string("enclose_many: at least one zonotope is needed"); });
    const int64_t n = Zs.front().dim();
    CORA_CHECK(is_column(direction, n) && direction.ndim() == 2, Error::dimension_mismatch, [&] {
        return "enclose_many: the direction must be a column (" + std::to_string(n) +
               ", 1), got the shape " + shape_text(direction.shape());
    });
    for (const Zonotope &Z : Zs)
        CORA_CHECK(Z.dim() == n && Z.center().ndim() == 2, Error::dimension_mismatch, [&] {
            return "enclose_many: the zonotopes must be single sets of one dimension, " +
                   std::to_string(n);
        });
    const Tensor length = direction.vector_norm(Tensor::Norm::two, -2, true);
    CORA_CHECK(length.detach().item() > 0, Error::wrong_value,
               [] { return std::string("enclose_many: the direction must not be zero"); });

    // the frame: the direction, then the standard axes by growing alignment, without the last
    const Tensor new_gen = direction.divide(length);
    const Tensor order = new_gen.abs().reshape({n}).argsort();
    const Tensor axes = new_gen.eye_like(n).take(order.slice(0, 0, n - 1), -1);
    const Tensor rotation = Tensor::concat({new_gen, axes}, -1);
    const Tensor rotation_inverse = rotation.inv();

    // the box around the cells in that frame
    std::optional<Interval> hull;
    for (const Zonotope &Z : Zs) {
        const Interval box = Z.reduce(1).linear_map(rotation_inverse).interval();
        hull =
            hull ? Interval(hull->inf().minimum(box.inf()), hull->sup().maximum(box.sup())) : box;
    }
    return {hull->zonotope().linear_map(rotation), rotation_inverse};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
