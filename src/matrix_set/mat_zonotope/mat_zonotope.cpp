// mat_zonotope - the constructors of a matrix zonotope, checked, as CORA's matZonotope
//
// Syntax:   MatZonotope A(C, G);   MatZonotope A(C);   MatZonotope A;
// Inputs:   C - center, a matrix (..., rows, cols) without NaN;  G - generators (..., h, rows,
//           cols) of the same rows, columns and batch
// Outputs:  A - the matrices C + sum_i b_i G_i with |b|_inf <= 1; without G no generators; without
//           arguments the empty 0 x 0 set
// See also: IntervalMatrix::IntervalMatrix, Zonotope::Zonotope
//
// CORA: matrixSet/@matZonotope/matZonotope.m

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "global/format.h"
#include "matrix_set/mat_zonotope/private/priv_stacking.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether t has no NaN entry (detached: the test carries no gradient).
bool aux_without_nan(const Tensor &t) {
    return t.numel() == 0 || t.detach().isnan().max().item() == 0;
}

} // namespace


// ===========================================  MAIN  =========================================== //

MatZonotope::MatZonotope(Tensor center, Tensor generators)
    : C_(std::move(center)), G_(std::move(generators)) {
    detail::Entry entry;
    CORA_CHECK(C_.defined() && G_.defined(), Error::wrong_input_in_constructor, [] {
        return std::string("MatZonotope: the center and the generators must be tensors");
    });
    CORA_CHECK(C_.ndim() >= 2 && G_.ndim() == C_.ndim() + 1, Error::wrong_input_in_constructor,
               [&] {
                   return "MatZonotope: the center must be (..., rows, cols) and the generators "
                          "(..., h, rows, cols), got the shapes " +
                          shape_text(C_.shape()) + " and " + shape_text(G_.shape());
               });
    // the generators share the batch, the rows and the columns of the center
    CORA_CHECK(G_.size(-2) == rows() && G_.size(-1) == cols() &&
                   priv_replace_tail(G_, 3, {}) == priv_replace_tail(C_, 2, {}),
               Error::wrong_input_in_constructor, [&] {
                   return "MatZonotope: dimension mismatch between the center " +
                          shape_text(C_.shape()) + " and the generators " + shape_text(G_.shape());
               });
    CORA_CHECK(aux_without_nan(C_) && aux_without_nan(G_), Error::wrong_value, [] {
        return std::string("MatZonotope: the center and the generators must not be NaN");
    });
}

MatZonotope::MatZonotope(Tensor center) : C_(std::move(center)) {
    detail::Entry entry;
    CORA_CHECK(C_.defined() && C_.ndim() >= 2, Error::wrong_input_in_constructor, [] {
        return std::string("MatZonotope: the center must be a matrix (..., rows, cols)");
    });
    CORA_CHECK(aux_without_nan(C_), Error::wrong_value,
               [] { return std::string("MatZonotope: the center must not be NaN"); });
    G_ = priv_no_generators(C_);
}

CORA_FROM_NUMBERS MatZonotope::MatZonotope() : MatZonotope(Tensor::zeros({0, 0})) {}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
