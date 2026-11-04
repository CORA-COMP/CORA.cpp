// generate_random - a random interval matrix, as CORA's intervalMatrix.generateRandom
//
// Syntax:   IntervalMatrix A = IntervalMatrix::generate_random(rng);
//           A = IntervalMatrix::generate_random(rng, dimension, center, max_radius);
// Inputs:   rng - random numbers (its seed makes the result repeatable);  dimension - {rows,
//           cols}, default the shape of center or of max_radius, else each random in 1..10;
//           center - matrix, default 10 N(0, 1) per entry;  max_radius - a matrix of radii >= 0,
//           or one number (a 1 x 1 tensor) for all entries, default 10 U[0, 1] per entry
// Outputs:  A - the interval matrix of the center and the radii: center +- radius
// See also: Interval::generate_random
//
// CORA: matrixSet/@intervalMatrix/generateRandom.m

#include "cora/global/rng.h"
#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A random rows x cols matrix: scale * N(0, 1), or scale * U[0, 1].
CORA_FROM_NUMBERS Tensor aux_random(Rng &rng, int64_t rows, int64_t cols, double scale,
                                    bool normal) {
    std::vector<double> values(rows * cols);
    if (normal)
        rng.normal(values.data(), values.size(), scale);
    else
        rng.uniform(values.data(), values.size(), 0.0, scale);
    return Tensor::from_data(values, {rows, cols});
}

/// A random size in 1..10, as randi(10).
int64_t aux_random_size(Rng &rng) {
    double u;
    rng.uniform(&u, 1, 0.0, 10.0);
    return std::min<int64_t>(10, int64_t(u) + 1);
}

} // namespace


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS IntervalMatrix IntervalMatrix::generate_random(Rng &rng, Dimension dimension,
                                                                 const Matrix &center,
                                                                 const Matrix &max_radius) {
    detail::Entry entry;
    CORA_CHECK(
        !dimension || (dimension->at(0) >= 1 && dimension->at(1) >= 1), Error::wrong_value,
        [] { return std::string("generate_random: the dimension needs rows, columns >= 1"); });
    CORA_CHECK(!max_radius || max_radius->min().item() >= 0, Error::wrong_value,
               [] { return std::string("generate_random: max_radius must be nonnegative"); });
    // the size: given, else that of the center or the radii, else random
    std::array<int64_t, 2> n{0, 0};
    if (dimension)
        n = *dimension;
    else if (center)
        n = {center->size(-2), center->size(-1)};
    else if (max_radius)
        n = {max_radius->size(-2), max_radius->size(-1)};
    else
        n = {aux_random_size(rng), aux_random_size(rng)};
    const Tensor c = center ? *center : aux_random(rng, n[0], n[1], 10.0, true);
    Tensor r = max_radius ? *max_radius : aux_random(rng, n[0], n[1], 10.0, false);
    if (max_radius && r.numel() == 1) r = r.reshape({1, 1}).tile({n[0], n[1]});
    CORA_CHECK(c.ndim() == 2 && c.shape() == r.shape(), Error::wrong_input_in_constructor, [&] {
        return "generate_random: the center and the radius must be matrices of one shape, got " +
               shape_text(c.shape()) + " and " + shape_text(r.shape());
    });
    return IntervalMatrix(c, r);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
