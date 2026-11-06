// polytope - the constructors of a polytope, checked, as CORA's polytope
//
// Syntax:   Polytope P(A, b);   Polytope P(A, b, Ae, be);   Polytope P(V);   Polytope P(I);
// Inputs:   A (..., m, n), b (..., m, 1), Ae (..., k, n), be (..., k, 1) - the halfspaces
//           A x <= b, Ae x = be, one batch shape; V (..., n, p) - the vertices, p = 0 for none,
//           finite unless n = 1; I - a box (n, 1) bounds
// Outputs:  P - the polytope; no rows is R^n. A row of A with the offset +Inf is dropped (it
//           holds everywhere; not in a batch); an offset -Inf in b or +-Inf in be makes the set
//           empty, which then also holds no vertices. In 1D only the least and the greatest
//           vertex are kept (one if they agree)
// See also: empty, inf, constraints
//
// CORA: contSet/@polytope/polytope.m
// CORA: contSet/@interval/polytope.m
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Tobias Ladner, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cont_set/polytope/private/priv.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The leading (batch) axes of a tensor.
std::vector<int64_t> aux_batch(const Tensor &t) {
    const std::vector<int64_t> shape = t.shape();
    return {shape.begin(), shape.end() - 2};
}

/// Whether the tensor is a matrix or a batch of them, with rows (the second to last axis).
int64_t aux_rows(const Tensor &t) { return t.shape()[t.ndim() - 2]; }

/// Whether any entry of t is NaN.
bool aux_has_nan(const Tensor &t) { return t.numel() > 0 && t.detach().isnan().sum().item() > 0.0; }

/// Whether any entry of t is infinite.
bool aux_has_inf(const Tensor &t) { return t.numel() > 0 && t.detach().isinf().sum().item() > 0.0; }

/// Why the halfspaces are no polytope, if they are none.
detail::Refusal aux_h_refusal(const Tensor &A, const Tensor &b, const Tensor &Ae,
                              const Tensor &be) {
    const auto fail = [](const std::string &text) {
        return detail::Refusal(std::pair{Error::wrong_input_in_constructor, "Polytope: " + text});
    };
    if (!A.defined() || !b.defined() || !Ae.defined() || !be.defined())
        return fail("the constraints must be tensors");
    if (A.ndim() < 2 || Ae.ndim() < 2)
        return fail("the constraint matrices must be matrices (..., rows, n), got " +
                    shape_text(A.shape()) + " and " + shape_text(Ae.shape()));
    if (b.ndim() < 2 || b.shape().back() != 1 || be.ndim() < 2 || be.shape().back() != 1)
        return fail("the offsets must be columns (..., rows, 1), got " + shape_text(b.shape()) +
                    " and " + shape_text(be.shape()));
    // the rows of the matrices and of their offsets, and the columns of both matrices, must agree
    if (aux_rows(A) != aux_rows(b))
        return fail("the number of rows does not hold between the arguments A and b: " +
                    shape_text(A.shape()) + " and " + shape_text(b.shape()));
    if (aux_rows(Ae) != aux_rows(be))
        return fail("the number of rows does not hold between the arguments Ae and be: " +
                    shape_text(Ae.shape()) + " and " + shape_text(be.shape()));
    if (A.shape().back() != Ae.shape().back())
        return fail("the number of columns does not hold between the arguments A and Ae: " +
                    shape_text(A.shape()) + " and " + shape_text(Ae.shape()));
    // a batch has one shape in all four parts
    if (aux_batch(A) != aux_batch(b) || aux_batch(A) != aux_batch(Ae) ||
        aux_batch(A) != aux_batch(be))
        return fail("the constraints must have one batch shape");
    return std::nullopt;
}

/// Why the vertices are no polytope, if they are none.
detail::Refusal aux_v_refusal(const Tensor &V) {
    const auto fail = [](const std::string &text) {
        return detail::Refusal(std::pair{Error::wrong_input_in_constructor, "Polytope: " + text});
    };
    if (!V.defined()) return fail("the vertices must be a tensor");
    if (V.ndim() < 2)
        return fail("the vertices must be a matrix (..., n, p), got " + shape_text(V.shape()));
    if (aux_has_nan(V)) return fail("vertices have to be non-nan");
    if (aux_rows(V) > 1 && aux_has_inf(V)) return fail("nD vertices for n > 1 have to be finite");
    return std::nullopt;
}

/// The rows of the halfspaces with an offset that is not +Inf (all, if the offsets are finite).
std::vector<int64_t> aux_finite_rows(const Tensor &b) {
    const std::vector<double> values = b.data();
    std::vector<int64_t> keep;
    for (std::size_t i = 0; i < values.size(); ++i)
        if (!(std::isinf(values[i]) && values[i] > 0)) keep.push_back(int64_t(i));
    return keep;
}

/// Whether an inequality offset is -Inf or an equality offset infinite: the set is empty.
bool aux_infeasible_offset(const Tensor &b, const Tensor &be) {
    for (const double x : b.data())
        if (std::isinf(x) && x < 0) return true;
    for (const double x : be.data())
        if (std::isinf(x)) return true;
    return false;
}

/// The text of the refusal of the halfspaces of a polytope that has only vertices.
std::string aux_no_halfspaces() {
    return "The halfspace representation is not available. Call the function 'constraints'.";
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Vertices ----------------------------------------------------------------------------------------

Polytope::Polytope(Tensor V) : ContSet(kClassId), V_(std::move(V)) {
    detail::Entry entry;
    const detail::Refusal refusal = aux_v_refusal(V_);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    if (aux_rows(V_) != 1 || V_.shape().back() == 0) return;
    // in 1D only the extremes matter; a single point if they agree
    const Tensor lo = V_.min(-1, true), hi = V_.max(-1, true);
    const bool single = V_.ndim() == 2 && within_tol(lo.item(), hi.item(), machine_eps());
    V_ = single ? lo : Tensor::concat({lo, hi}, -1);
}

// Halfspaces --------------------------------------------------------------------------------------

Polytope::Polytope(Tensor A, Tensor b, Tensor Ae, Tensor be)
    : ContSet(kClassId), A_(std::move(A)), b_(std::move(b)), Ae_(std::move(Ae)),
      be_(std::move(be)) {
    detail::Entry entry;
    const detail::Refusal refusal = aux_h_refusal(A_, b_, Ae_, be_);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    const bool batch = A_.ndim() > 2;
    // an offset +Inf is satisfied by every point: the row goes
    if (aux_rows(b_) > 0) {
        const std::vector<int64_t> keep = aux_finite_rows(b_);
        if (int64_t(keep.size()) != b_.numel()) {
            if (batch)
                CORA_FAIL(Error::not_supported,
                          "Polytope: a batch with an infinite offset is not supported");
            A_ = keep.empty() ? priv_zeros(A_, 0, A_.shape().back()) : A_.take(keep, -2);
            b_ = keep.empty() ? priv_zeros(b_, 0, 1) : b_.take(keep, -2);
        }
    }
    // an infeasible offset: the set is empty, and so has no vertices
    if (aux_infeasible_offset(b_, be_)) V_ = priv_zeros(A_, A_.shape().back(), 0);
}

Polytope::Polytope(const Tensor &A, Tensor b)
    : Polytope(A, std::move(b),
               A.defined() && A.ndim() >= 2 ? priv_zeros(A, 0, A.shape().back()) : Tensor(),
               A.defined() && A.ndim() >= 2 ? priv_zeros(A, 0, 1) : Tensor()) {}

Polytope::Polytope(Parts parts)
    : ContSet(kClassId), A_(std::move(parts.A)), b_(std::move(parts.b)), Ae_(std::move(parts.Ae)),
      be_(std::move(parts.be)), V_(std::move(parts.V)) {}

// A box -------------------------------------------------------------------------------------------

Polytope::Polytope(const Interval &I) : Polytope(priv_polytope_of_interval(I)) {}

// Data --------------------------------------------------------------------------------------------

// the halfspace form exists only if it is held; the vertices of the vertex form likewise
const Tensor &Polytope::A() const {
    detail::Entry entry;
    CORA_CHECK(is_h_rep(), Error::special_error, aux_no_halfspaces);
    return A_;
}

const Tensor &Polytope::b() const {
    detail::Entry entry;
    CORA_CHECK(is_h_rep(), Error::special_error, aux_no_halfspaces);
    return b_;
}

const Tensor &Polytope::Ae() const {
    detail::Entry entry;
    CORA_CHECK(is_h_rep(), Error::special_error, aux_no_halfspaces);
    return Ae_;
}

// the offset of the equalities, and then the vertices
const Tensor &Polytope::be() const {
    detail::Entry entry;
    CORA_CHECK(is_h_rep(), Error::special_error, aux_no_halfspaces);
    return be_;
}

const Tensor &Polytope::V() const {
    detail::Entry entry;
    CORA_CHECK(is_v_rep(), Error::special_error, [] {
        return std::string("The vertex representation is not available. "
                           "Call the function 'vertices'.");
    });
    return V_;
}

int64_t Polytope::dim_impl() const {
    return is_h_rep() ? A_.shape().back() : V_.shape()[V_.ndim() - 2];
}

// Not ported yet ----------------------------------------------------------------------------------

// Not yet ported: the support function, random points, projections and vertices.
Tensor Polytope::support_function_impl(const Tensor &) const {
    CORA_FAIL(Error::not_supported, "Polytope: support_function is not ported yet");
}

Tensor Polytope::rand_point_impl(int64_t, Rng &) const {
    CORA_FAIL(Error::not_supported, "Polytope: rand_point is not ported yet");
}

Set Polytope::project_impl(const std::vector<int64_t> &) const {
    CORA_FAIL(Error::not_supported, "Polytope: project is not ported yet");
}

std::vector<Polygon> Polytope::vertices_impl() const {
    CORA_FAIL(Error::not_supported, "Polytope: vertices is not ported yet");
}

std::vector<std::vector<Point3>> Polytope::vertices_3d_impl() const {
    CORA_FAIL(Error::not_supported, "Polytope: vertices_3d is not ported yet");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
