// priv_host - the host form of a polytope and the small predicates the operations share
//
// Syntax:   HostH H = priv_host_h(P.constraints());   Rows V = priv_polytope_vertices(P);
//           bool full = priv_is_fullspace(P);   Polytope B = priv_polytope_of_interval(I);
// Inputs:   P - a single polytope (the host form), any polytope (the predicates)
// Outputs:  R - the rows of the halfspaces and vertices, whether the set is R^n, a box's polytope
// See also: priv_compact, priv_vertices
//
// CORA.cpp only: MATLAB reads its matrices as they are; CORA's @interval/polytope.m is the
//                polytope of a box
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows of a single matrix (rows, cols) on the host.
Rows aux_rows_of(const Tensor &M) {
    const int64_t rows = M.shape()[0], cols = M.shape()[1];
    const std::vector<double> flat = M.data();
    Rows out(static_cast<std::size_t>(rows), Row(static_cast<std::size_t>(cols)));
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j) out[i][j] = flat[std::size_t(i * cols + j)];
    return out;
}

/// The matrix (rows, cols) of rows, on the backend of `like`.
Tensor aux_matrix(const Tensor &like, const Rows &rows, int64_t cols) {
    std::vector<double> flat;
    for (const Row &row : rows) flat.insert(flat.end(), row.begin(), row.end());
    return Tensor::like(like, flat, {int64_t(rows.size()), cols});
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Host form ---------------------------------------------------------------------------------------

HostH priv_host_h(const Polytope::Constraints &C) {
    if (C.A.ndim() != 2)
        CORA_FAIL(Error::not_supported, "Polytope: a linear program per set: it supports a "
                                        "single polytope, not a batch");
    // the rows and offsets as host numbers
    HostH H;
    H.n = C.A.shape()[1];
    H.A = aux_rows_of(C.A);
    H.b = C.b.data();
    H.Ae = aux_rows_of(C.Ae);
    H.be = C.be.data();
    return H;
}

Polytope::Constraints priv_tensors_h(const Tensor &like, const HostH &H) {
    // back to tensors of the backend of `like`
    return {aux_matrix(like, H.A, H.n), Tensor::like(like, H.b, {int64_t(H.b.size()), 1}),
            aux_matrix(like, H.Ae, H.n), Tensor::like(like, H.be, {int64_t(H.be.size()), 1})};
}

Rows priv_host_points(const Tensor &V) {
    if (V.ndim() != 2)
        CORA_FAIL(Error::not_supported,
                  "Polytope: the vertices of a batch are not supported by this operation");
    // column k of V (n, p) is the point k
    const int64_t n = V.shape()[0], p = V.shape()[1];
    const std::vector<double> flat = V.data();
    Rows points(static_cast<std::size_t>(p), Row(static_cast<std::size_t>(n)));
    for (int64_t k = 0; k < p; ++k)
        for (int64_t i = 0; i < n; ++i) points[k][i] = flat[std::size_t(i * p + k)];
    return points;
}

Tensor priv_points_tensor(const Tensor &like, int64_t n, const Rows &points) {
    // the points are the columns of (n, p)
    const int64_t p = int64_t(points.size());
    std::vector<double> flat(std::size_t(n * p));
    for (int64_t k = 0; k < p; ++k)
        for (int64_t i = 0; i < n; ++i) flat[std::size_t(i * p + k)] = points[k][i];
    return Tensor::like(like, flat, {n, p});
}

Tensor priv_column_tensor(const Tensor &like, const Row &values) {
    return Tensor::like(like, values, {int64_t(values.size()), 1});
}

// Small linear algebra ----------------------------------------------------------------------------

double priv_dot(const Row &a, const Row &b) {
    double sum = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) sum += a[k] * b[k];
    return sum;
}

double priv_norm(const Row &a) { return std::sqrt(priv_dot(a, a)); }

std::optional<Row> priv_solve(Rows M, Row r, double pivot_tol) {
    const std::size_t n = r.size();
    double scale = 0.0;
    for (const Row &row : M)
        for (const double x : row) scale = std::max(scale, std::abs(x));
    // forward elimination with the largest entry of the column as pivot
    for (std::size_t c = 0; c < n; ++c) {
        std::size_t pivot = c;
        for (std::size_t i = c + 1; i < n; ++i)
            if (std::abs(M[i][c]) > std::abs(M[pivot][c])) pivot = i;
        if (std::abs(M[pivot][c]) <= pivot_tol * scale) return std::nullopt;
        std::swap(M[c], M[pivot]);
        std::swap(r[c], r[pivot]);
        for (std::size_t i = c + 1; i < n; ++i) {
            const double f = M[i][c] / M[c][c];
            for (std::size_t j = c; j < n; ++j) M[i][j] -= f * M[c][j];
            r[i] -= f * r[c];
        }
    }
    // back substitution
    Row x(n);
    for (std::size_t i = n; i-- > 0;) {
        double sum = r[i];
        for (std::size_t j = i + 1; j < n; ++j) sum -= M[i][j] * x[j];
        x[i] = sum / M[i][i];
    }
    return x;
}

// Predicates --------------------------------------------------------------------------------------

bool priv_is_fullspace(const Polytope &P, double tol) {
    if (P.is_h_rep()) {
        // every row 0 <= b with b >= 0 (up to tol), every equality 0 = 0 (up to tol)
        const auto zero = [tol](const Tensor &t) {
            return t.numel() == 0 || t.detach().abs().max().item() <= tol;
        };
        const bool trivial = zero(P.A()) && zero(P.Ae()) && zero(P.be()) &&
                             (P.b().numel() == 0 || P.b().detach().min().item() >= -tol);
        if (trivial) return true;
    }
    // in 1D the vertices at both infinities are R
    if (P.is_v_rep() && P.dim() == 1 && P.V().numel() > 0) {
        const Tensor V = P.V().detach();
        constexpr double kInf = std::numeric_limits<double>::infinity();
        return V.min().item() == -kInf && V.max().item() == kInf;
    }
    return false;
}

bool priv_has_no_vertices(const Polytope &P) { return P.is_v_rep() && P.V().shape().back() == 0; }

bool priv_polytope_is_empty(const Polytope &P) {
    if (P.is_v_rep()) return P.V().shape().back() == 0;
    return priv_is_empty(priv_host_h(P.constraints()));
}

bool priv_polytope_is_bounded(const Polytope &P) {
    if (P.is_v_rep()) {
        // vertices are finite in several dimensions; in 1D a bound can be infinite
        return P.dim() > 1 || P.V().numel() == 0 || P.V().detach().isinf().max().item() == 0.0;
    }
    return priv_is_bounded(priv_host_h(P.constraints()));
}

// Vertices ----------------------------------------------------------------------------------------

Rows priv_polytope_vertices(const Polytope &P) {
    if (P.is_v_rep()) {
        Rows points = priv_host_points(P.V());
        // one vertex, or any number in 1D, are minimal; else the ones that are not extreme go
        if (points.size() <= 1 || P.dim() == 1) return points;
        Rows kept;
        for (const int64_t k : priv_extreme_points(points)) kept.push_back(points[std::size_t(k)]);
        return kept;
    }
    // a halfspace set: the endpoints in 1D, else the enumeration of the vertices
    const HostH H = priv_host_h(P.constraints());
    if (H.n == 1) {
        bool empty = false;
        const Row v = priv_vertices_1d(H, empty);
        Rows points;
        for (const double x : v) points.push_back({x});
        return points;
    }
    return priv_vertices(H);
}

void priv_require_enclosable(const Polytope &P, const std::string &op, const std::string &target) {
    // a batch and an unbounded set have no enclosure
    const Tensor &like = P.is_h_rep() ? P.A() : P.V();
    if (like.ndim() != 2)
        CORA_FAIL(Error::not_supported, op + ": it supports a single polytope, not a batch");
    if (priv_is_fullspace(P) || !priv_polytope_is_bounded(P))
        CORA_FAIL(Error::special_error,
                  "Polytope is unbounded and can therefore not be enclosed by " + target + ".");
}

Tensor priv_place(const Tensor &M, int64_t N, const std::vector<int64_t> &dims, int axis) {
    const int64_t n = int64_t(dims.size());
    std::vector<int64_t> source(std::size_t(N), n);
    for (int64_t i = 0; i < n; ++i) source[std::size_t(dims[std::size_t(i)])] = i;
    const std::vector<int64_t> shape = M.shape();
    const int64_t rows = shape[shape.size() - 2], cols = shape.back();
    // the zero line that the other positions take
    const Tensor zero = axis == -1 ? priv_zeros(M, rows, 1) : priv_zeros(M, 1, cols);
    return Tensor::concat({M, zero}, axis).take(source, axis);
}

Rows priv_gram_schmidt(const Row &a) {
    const std::size_t n = a.size();
    // the columns a, then the axes but the one a points to the most
    std::size_t skip = 0;
    for (std::size_t i = 1; i < n; ++i)
        if (std::abs(a[i]) > std::abs(a[skip])) skip = i;
    Rows B = {a};
    for (std::size_t i = 0; i < n; ++i) {
        if (i == skip) continue;
        Row e(n, 0.0);
        e[i] = 1.0;
        B.push_back(e);
    }
    // classical Gram-Schmidt on the columns, as CORA's gramSchmidt
    Rows Q;
    for (std::size_t j = 0; j < n; ++j) {
        Row v = B[j];
        for (const Row &q : Q) {
            const double r = priv_dot(q, B[j]);
            for (std::size_t i = 0; i < n; ++i) v[i] -= r * q[i];
        }
        const double norm = priv_norm(v);
        for (double &x : v) x /= norm;
        Q.push_back(v);
    }
    return Q;
}

// A box -------------------------------------------------------------------------------------------

Polytope priv_polytope_of_interval(const Interval &I) {
    const int64_t n = I.dim();
    if (I.is_empty()) return Polytope::empty(n);
    // x <= sup and -x <= -inf; the rows with an infinite offset are dropped by the constructor
    const Tensor eye = priv_identity(I.inf(), n);
    // the negative identity as 0 - I, so that its zeros are +0 as in CORALean
    const Tensor negative = priv_zeros(I.inf(), n, n) - eye;
    return {Tensor::concat({eye, negative}, -2), Tensor::concat({I.sup(), -I.inf()}, -2)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
