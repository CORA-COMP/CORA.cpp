// split - a zonotope cut into zonotopes that together cover it, as CORA's zonotope.split
//
// Syntax:   pieces = Z.split();   [Z1, Z2] = Z.split(N);   [Z1, Z2] = Z.split(N, max_order);
//           [Z1, Z2] = Z.split(dir);   [Z1, Z2] = Z.split(orig_dir, aux_dir);
//           pieces = Z.split(a, d);
// Inputs:   N - a coordinate of the box around Z, or with max_order a generator of Z reduced to
//           that order (0-based);  dir - the normal of the cutting plane through the center;
//           orig_dir, aux_dir - the plane is normal to aux_dir minus its part along orig_dir;
//           a, d - the plane a'x = d
// Outputs:  pieces - one pair per coordinate of the box (no argument); the pair of halves
//           (c - g/2 and c + g/2, g halved) of the box or the generator; the pair of the
//           direction split, whose generators lose their part along dir, which becomes the
//           generator of the pieces; or for a plane both pieces, or Z alone if it misses Z.
//           The pieces of a box cover the box, the others cover Z. CORA indexes the nonzero
//           generators of the box by N, so a flat coordinate shifts it or breaks it; CORA.cpp
//           indexes the coordinates. The split with 'bundle' needs zonoBundle and is not
//           supported. Not a batch, except for N
// See also: split_first_gen, split_longest_gen, reduce
//
// CORA: contSet/@zonotope/split.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Boxes and cuts ----------------------------------------------------------------------------------

/// The box around Z as a zonotope with a generator for every coordinate, even a flat one.
Zonotope aux_box(const Zonotope &Z) {
    const Interval I = Z.interval();
    return {I.center(), I.rad().diag()};
}

/// The unit vector of dir (n, 1).
Tensor aux_unit(const Tensor &dir) {
    return dir.divide(dir.vector_norm(Tensor::Norm::two, -2, true));
}

/// The generators without their part along the unit vector u, and the generator with the sum of
/// the absolute parts, alignedVal * u (CORA's loop over the generators).
std::pair<Tensor, Tensor> aux_decompose(const Tensor &G, const Tensor &u) {
    const Tensor aligned = u.matrix_transpose().matmul(G); // (1, m): the part of each along u
    return {G - u.matmul(aligned), u.multiply(aligned.abs().sum(-1, true))};
}

/// The pieces of the cut at the fraction beta in (-1, 1) of the new generator.
std::pair<Zonotope, Zonotope> aux_cut(const Tensor &c, const Tensor &G, const Tensor &new_gen,
                                      double beta) {
    const Tensor g1 = new_gen * (0.5 * (beta + 1)), g2 = new_gen * (0.5 * (beta - 1));
    const Tensor shifted = c + new_gen * beta;
    return {{shifted - g1, Tensor::concat({g1, G}, -1)},
            {shifted - g2, Tensor::concat({g2, G}, -1)}};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// By coordinate or generator ----------------------------------------------------------------------

std::pair<Zonotope, Zonotope> Zonotope::split(int64_t N) const {
    detail::Entry entry;
    CORA_CHECK(N >= 0 && N < dim(), Error::wrong_value, [&] {
        return "split: the coordinate " + std::to_string(N) + " is not in [0, " +
               std::to_string(dim()) + ")";
    });
    // the halves of the box, which has a generator for every coordinate
    return priv_split_generator(aux_box(*this), N);
}

std::vector<std::pair<Zonotope, Zonotope>> Zonotope::split() const {
    detail::Entry entry;
    const Zonotope box = aux_box(*this);
    std::vector<std::pair<Zonotope, Zonotope>> pieces;
    pieces.reserve(dim());
    for (int64_t N = 0; N < dim(); ++N) pieces.push_back(priv_split_generator(box, N));
    return pieces;
}

std::pair<Zonotope, Zonotope> Zonotope::split(int64_t N, int max_order) const {
    detail::Entry entry;
    CORA_CHECK(max_order >= 1, Error::wrong_value, [&] {
        return "split: max_order must be at least 1, got " + std::to_string(max_order);
    });
    // the generators of the reduced zonotope are the ones that can be cut
    const Zonotope reduced = reduce(max_order);
    CORA_CHECK(N >= 0 && N < reduced.G_.shape().back(), Error::wrong_value, [&] {
        return "split: the generator " + std::to_string(N) + " is not in [0, " +
               std::to_string(reduced.G_.shape().back()) + ") of the reduced zonotope";
    });
    return priv_split_generator(reduced, N);
}

// By a plane --------------------------------------------------------------------------------------

std::pair<Zonotope, Zonotope> Zonotope::split(const Tensor &dir) const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 2, Error::not_supported, [] {
        return std::string("split: a split by a direction supports a single zonotope");
    });
    CORA_CHECK(is_column(dir, dim()), Error::dimension_mismatch, [&] {
        return "split: dir must be a column of " + std::to_string(dim()) +
               " entries, got the "
               "shape " +
               shape_text(dir.shape());
    });
    CORA_CHECK(dir.abs().sum().item() > 0, Error::wrong_value,
               [] { return std::string("split: dir must not be zero"); });
    const auto [rest, new_gen] = aux_decompose(G_, aux_unit(dir));
    return aux_cut(c_, rest, new_gen, 0.0);
}

std::pair<Zonotope, Zonotope> Zonotope::split(const Tensor &orig_dir, const Tensor &aux_dir) const {
    detail::Entry entry;
    CORA_CHECK(
        is_column(orig_dir, dim()) && is_column(aux_dir, dim()), Error::dimension_mismatch, [&] {
            return "split: the directions must be columns of " + std::to_string(dim()) + " entries";
        });
    // the split direction is not zero
    CORA_CHECK(orig_dir.abs().sum().item() > 0, Error::wrong_value,
               [] { return std::string("split: orig_dir must not be zero"); });
    // aux_dir minus its part along orig_dir, as CORA computes it
    const Tensor length = orig_dir.vector_norm(Tensor::Norm::two, -2, true);
    const Tensor part = aux_dir.matrix_transpose().matmul(orig_dir).divide(length);
    const Tensor perpendicular = aux_dir - orig_dir.multiply(part).divide(length);
    CORA_CHECK(perpendicular.abs().sum().item() > 0, Error::wrong_value,
               [] { return std::string("split: aux_dir must not be parallel to orig_dir"); });
    return split(perpendicular);
}

std::vector<Zonotope> Zonotope::split(const Tensor &a, double d) const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 2, Error::not_supported, [] {
        return std::string("split: a split by a halfspace supports a single zonotope");
    });
    CORA_CHECK(is_column(a, dim()), Error::dimension_mismatch, [&] {
        return "split: a must be a column of " + std::to_string(dim()) +
               " entries, got the "
               "shape " +
               shape_text(a.shape());
    });
    CORA_CHECK(a.abs().sum().item() > 0, Error::wrong_value,
               [] { return std::string("split: a must not be zero"); });
    // the position of the plane along the new generator, in units of that generator
    const auto [rest, new_gen] = aux_decompose(G_, aux_unit(a));
    const Tensor at = a.matrix_transpose();
    const double beta = (d - at.matmul(c_).item()) / at.matmul(new_gen).item();
    if (!(std::abs(beta) < 1)) return {*this};
    const auto [lower, upper] = aux_cut(c_, rest, new_gen, beta);
    return {lower, upper};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
