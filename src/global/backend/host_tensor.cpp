// host_tensor - the storage and the reference operations of the CPU backends (eigen and lean)
//
// Elementwise operations are the machine's IEEE operations; a sum is CORALean's sumUp, nested to
// the right from zero, `x0 + (x1 + (... + (x_{n-1} + 0)))`, and a product of matrices sums each
// entry that way. Rearrangements copy entries and round nothing. The factorizations (solve, qr,
// svd, eigh, cholesky) have no CORALean counterpart: they are Eigen's, matrix by matrix.
//
// See also: host_tensor.h, eigen.cpp, lean.cpp
//
// CORA.cpp only: the backends of Tensor replace MATLAB's matrices.
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   06-November-2026
// Reviewed:      ---

#include "global/backend/host_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

#include <Eigen/Dense>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::host {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Binary = Tensor::Binary;
using Unary = Tensor::Unary;
using Reduce = Tensor::Reduce;

template <class T>
using RowMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <class T>
using MapC = Eigen::Map<const RowMat<T>>;
template <class T>
using Map = Eigen::Map<RowMat<T>>;

/// The dtype of the entry type.
template <class T>
constexpr DType aux_dtype() {
    return std::is_same_v<T, double> ? DType::float64 : DType::float32;
}

std::string aux_text(const Shape &s) {
    std::string text = "(";
    for (int64_t i = 0; i < s.size(); ++i) text += (i ? ", " : "") + std::to_string(s[i]);
    return text + ")";
}

// Elementwise -------------------------------------------------------------------------------------

/// The larger of a and b, NaN if either is: torch.maximum.
template <class T>
T aux_max(T a, T b) {
    if (std::isnan(a)) return a;
    return a > b ? a : b;
}
template <class T>
T aux_min(T a, T b) {
    if (std::isnan(a)) return a;
    return a < b ? a : b;
}

/// fmod(a, b) moved to the sign of b, as torch.remainder and MATLAB's mod.
template <class T>
T aux_remainder(T a, T b) {
    T r = std::fmod(a, b);
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

/// Calls g(f) with the scalar function f of `op`: the one switch over the binary operations.
template <class T, class G>
auto aux_with_binary(Binary op, G &&g) {
    switch (op) {
    case Binary::add:
        return g([](T a, T b) { return a + b; });
    case Binary::subtract:
        return g([](T a, T b) { return a - b; });
    case Binary::multiply:
        return g([](T a, T b) { return a * b; });
    case Binary::divide:
        return g([](T a, T b) { return a / b; });
    case Binary::pow:
        return g([](T a, T b) { return std::pow(a, b); });
    // the extrema, the remainder and atan2
    case Binary::maximum:
        return g([](T a, T b) { return aux_max(a, b); });
    case Binary::minimum:
        return g([](T a, T b) { return aux_min(a, b); });
    case Binary::remainder:
        return g([](T a, T b) { return aux_remainder(a, b); });
    case Binary::atan2:
        return g([](T a, T b) { return std::atan2(a, b); });
    // the comparisons give 1 or 0 of the operand type
    case Binary::equal:
        return g([](T a, T b) { return T(a == b); });
    case Binary::not_equal:
        return g([](T a, T b) { return T(a != b); });
    case Binary::less:
        return g([](T a, T b) { return T(a < b); });
    case Binary::less_equal:
        return g([](T a, T b) { return T(a <= b); });
    case Binary::greater:
        return g([](T a, T b) { return T(a > b); });
    case Binary::greater_equal:
        return g([](T a, T b) { return T(a >= b); });
    }
    CORA_FAIL(Error::wrong_value, "CoraTensor: unknown elementwise operation");
}

/// -1, 0 or 1; NaN stays.
template <class T>
T aux_sign(T x) {
    return std::isnan(x) ? x : T((x > 0) - (x < 0));
}

/// Calls g(f) with the scalar function f of `op`.
template <class T, class G>
auto aux_with_unary(Unary op, G &&g) {
    switch (op) {
    case Unary::negative:
        return g([](T x) { return -x; });
    case Unary::abs:
        return g([](T x) { return std::fabs(x); });
    case Unary::sign:
        return g([](T x) { return aux_sign(x); });
    case Unary::floor:
        return g([](T x) { return std::floor(x); });
    case Unary::ceil:
        return g([](T x) { return std::ceil(x); });
    case Unary::round:
        return g([](T x) { return std::round(x) + T(0); }); // -0 becomes 0
    case Unary::sqrt:
        return g([](T x) { return std::sqrt(x); });
    case Unary::exp:
        return g([](T x) { return std::exp(x); });
    case Unary::log:
        return g([](T x) { return std::log(x); });
    // the C library: CORALean has no counterpart of these
    case Unary::sin:
        return g([](T x) { return std::sin(x); });
    case Unary::cos:
        return g([](T x) { return std::cos(x); });
    case Unary::tan:
        return g([](T x) { return std::tan(x); });
    case Unary::asin:
        return g([](T x) { return std::asin(x); });
    case Unary::acos:
        return g([](T x) { return std::acos(x); });
    case Unary::atan:
        return g([](T x) { return std::atan(x); });
    case Unary::sinh:
        return g([](T x) { return std::sinh(x); });
    case Unary::cosh:
        return g([](T x) { return std::cosh(x); });
    case Unary::tanh:
        return g([](T x) { return std::tanh(x); });
    case Unary::asinh:
        // the elementwise functions, one case each
        return g([](T x) { return std::asinh(x); });
    case Unary::acosh:
        return g([](T x) { return std::acosh(x); });
    case Unary::atanh:
        return g([](T x) { return std::atanh(x); });
    // masks: 1 or 0
    case Unary::isnan:
        return g([](T x) { return T(std::isnan(x)); });
    case Unary::isinf:
        return g([](T x) { return T(std::isinf(x)); });
    case Unary::isfinite:
        return g([](T x) { return T(std::isfinite(x)); });
    }
    CORA_FAIL(Error::wrong_value, "CoraTensor: unknown elementwise function");
}

/// The strides of `s` aligned to the last axes of `out`, zero where s is broadcast.
Shape aux_broadcast_strides(const Shape &s, const Shape &out) {
    const Shape st = strides_of(s);
    Shape r;
    const int64_t shift = out.size() - s.size();
    for (int64_t d = 0; d < out.size(); ++d)
        r.push_back(d < shift || s[d - shift] == 1 ? 0 : st[d - shift]);
    return r;
}

/// Calls f(offsets...) for every position of `out` in row-major order, with the offsets of
/// each operand by its strides: the odometer behind broadcasting, permutation and slicing.
template <std::size_t N, class F>
void aux_walk(const Shape &out, const std::array<Shape, N> &strides, F f) {
    const int64_t rank = out.size();
    if (out.numel() == 0) return;
    if (rank == 0) {
        std::array<int64_t, N> zero{};
        std::apply(f, zero);
        return;
    }
    Shape idx;
    for (int64_t d = 0; d < rank; ++d) idx.push_back(0);
    std::array<int64_t, N> base{};
    const int64_t inner = out.back();
    while (true) {
        for (int64_t j = 0; j < inner; ++j) {
            std::array<int64_t, N> at;
            for (std::size_t k = 0; k < N; ++k) at[k] = base[k] + j * strides[k].back();
            std::apply(f, at);
        }
        // advance the outer axes like an odometer
        int64_t d = rank - 2;
        for (; d >= 0; --d) {
            for (std::size_t k = 0; k < N; ++k) base[k] += strides[k][d];
            if (++idx[d] < out[d]) break;
            for (std::size_t k = 0; k < N; ++k) base[k] -= strides[k][d] * out[d];
            idx[d] = 0;
        }
        if (d < 0) return;
    }
}

/// out = f(a, b) elementwise with broadcasting; the common cases without the odometer.
template <class T, class F>
void aux_binary_into(const HostTensor<T> &a, const HostTensor<T> &b, T *out, const Shape &shape,
                     F f) {
    const int64_t n = shape.numel();
    if (a.shape == shape && b.shape == shape) {
        for (int64_t i = 0; i < n; ++i) out[i] = f(a.p[i], b.p[i]);
    } else if (b.shape.numel() == 1 && a.shape == shape) {
        const T s = b.p[0];
        for (int64_t i = 0; i < n; ++i) out[i] = f(a.p[i], s);
    } else if (a.shape.numel() == 1 && b.shape == shape) {
        const T s = a.p[0];
        for (int64_t i = 0; i < n; ++i) out[i] = f(s, b.p[i]);
    } else {
        const std::array<Shape, 3> st = {aux_broadcast_strides(a.shape, shape),
                                         aux_broadcast_strides(b.shape, shape), strides_of(shape)};
        aux_walk<3>(shape, st,
                    [&](int64_t ia, int64_t ib, int64_t io) { out[io] = f(a.p[ia], b.p[ib]); });
    }
}

/// The entries of `src` at the strided positions (offset + sum idx_d * strides_d) of `out`.
template <class T>
void aux_gather(const T *src, int64_t offset, const Shape &strides, const Shape &out, T *dst) {
    const std::array<Shape, 2> st = {strides, strides_of(out)};
    aux_walk<2>(out, st, [&](int64_t is, int64_t io) { dst[io] = src[offset + is]; });
}

// Reductions --------------------------------------------------------------------------------------

/// The extents of a reduction along `axis`: the axes before, the axis, the axes after.
struct Extents {
    int64_t outer, len, inner;
};
Extents aux_extents(const Shape &s, int axis) {
    Extents e{1, s[axis], 1};
    for (int64_t d = 0; d < axis; ++d) e.outer *= s[d];
    for (int64_t d = axis + 1; d < s.size(); ++d) e.inner *= s[d];
    return e;
}

/// The shape after a reduction along `axis`.
Shape aux_reduced(const Shape &s, int axis, bool keepdims) {
    Shape r;
    for (int64_t d = 0; d < s.size(); ++d)
        if (d != axis)
            r.push_back(s[d]);
        else if (keepdims)
            r.push_back(1);
    return r;
}

/// Nested to the right: acc = f(x[len-1], init), then acc = f(x[l], acc) down to x[0]; per
/// (outer, inner) position, vectorized over the inner positions.
template <class T, class F>
void aux_fold_right(const T *x, const Extents &e, T init, T *out, F f) {
    for (int64_t o = 0; o < e.outer; ++o) {
        T *acc = out + o * e.inner;
        for (int64_t i = 0; i < e.inner; ++i) acc[i] = init;
        for (int64_t l = e.len - 1; l >= 0; --l) {
            const T *row = x + (o * e.len + l) * e.inner;
            for (int64_t i = 0; i < e.inner; ++i) acc[i] = f(row[i], acc[i]);
        }
    }
}

/// The best entry along the axis in order (first wins ties) and its index, by `better`.
template <class T, class B>
void aux_select(const T *x, const Extents &e, T *value, T *index, B better) {
    for (int64_t o = 0; o < e.outer; ++o)
        for (int64_t i = 0; i < e.inner; ++i) {
            T best = x[o * e.len * e.inner + i];
            int64_t at = 0;
            for (int64_t l = 1; l < e.len; ++l) {
                const T v = x[(o * e.len + l) * e.inner + i];
                if (!std::isnan(best) && (better(v, best) || std::isnan(v))) best = v, at = l;
            }
            if (value) value[o * e.inner + i] = best;
            if (index) index[o * e.inner + i] = T(at);
        }
}

// Linear Algebra ----------------------------------------------------------------------------------

/// The matrix `k` of a batch of (rows, cols) matrices.
template <class T>
MapC<T> aux_matrix(const T *p, int64_t k, int64_t rows, int64_t cols) {
    return MapC<T>(p + k * rows * cols, rows, cols);
}

/// The shape of a batch with its matrix of (rows, cols).
Shape aux_with_matrix(const Shape &batch, int64_t rows, int64_t cols) {
    Shape s = batch;
    s.push_back(rows);
    s.push_back(cols);
    return s;
}

/// The batch axes of a shape (all but the last two).
Shape aux_batch(const Shape &s) {
    Shape b;
    for (int64_t d = 0; d + 2 < s.size(); ++d) b.push_back(s[d]);
    return b;
}

/// The number of matrices of a tensor.
int64_t aux_matrices(const Shape &s) { return aux_batch(s).numel(); }

void aux_require_matrix(const Shape &s, const char *op, bool square) {
    if (s.size() < 2)
        CORA_FAIL(Error::dimension_mismatch, std::string("CoraTensor::") + op +
                                                 ": needs a matrix (..., n, m), got " +
                                                 aux_text(s));
    if (square && s[s.size() - 1] != s[s.size() - 2])
        CORA_FAIL(Error::dimension_mismatch,
                  std::string("CoraTensor::") + op + ": needs square matrices, got " + aux_text(s));
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Shapes ------------------------------------------------------------------------------------------

Shape strides_of(const Shape &shape) {
    Shape st;
    for (int64_t d = 0; d < shape.size(); ++d) st.push_back(0);
    int64_t step = 1;
    for (int64_t d = shape.size() - 1; d >= 0; --d) {
        st[d] = step;
        step *= shape[d];
    }
    return st;
}

int axis_of(int axis, int64_t rank, const char *op) {
    const int64_t a = axis < 0 ? axis + rank : axis;
    if (a < 0 || a >= rank)
        CORA_FAIL(Error::wrong_value, std::string("CoraTensor::") + op + ": no axis " +
                                          std::to_string(axis) + " in a tensor of " +
                                          std::to_string(rank) + " axes");
    return static_cast<int>(a);
}

// Numpy's rule: align the last axes; sizes agree or one of them is 1.
Shape broadcast_shapes(const Shape &a, const Shape &b) {
    const int64_t rank = std::max(a.size(), b.size());
    Shape r;
    for (int64_t d = 0; d < rank; ++d) {
        const int64_t da = d - (rank - a.size()), db = d - (rank - b.size());
        const int64_t x = da >= 0 ? a[da] : 1, y = db >= 0 ? b[db] : 1;
        if (x != y && x != 1 && y != 1)
            CORA_FAIL(Error::dimension_mismatch, "CoraTensor: the shapes " + aux_text(a) + " and " +
                                                     aux_text(b) + " do not broadcast");
        r.push_back(x == 1 ? y : x);
    }
    return r;
}

// The batch axes broadcast; each result matrix names its operands' matrices.
BatchPairs batch_pairs(const Shape &a, const Shape &b, const char *op) {
    aux_require_matrix(a, op, false);
    aux_require_matrix(b, op, false);
    if (a.size() == 2 && b.size() == 2) return {}; // two matrices: one pair, nothing to broadcast
    const Shape ba = aux_batch(a), bb = aux_batch(b);
    const Shape batch = broadcast_shapes(ba, bb);
    return {batch, aux_broadcast_strides(ba, batch), aux_broadcast_strides(bb, batch)};
}

// Construction ------------------------------------------------------------------------------------

template <class T>
HostTensor<T>::HostTensor(const Shape &shape, uint8_t backend)
    : Tensor::Impl(shape, aux_dtype<T>(), Device{}, backend) {}

template <class T>
std::shared_ptr<HostTensor<T>> HostTensor<T>::make(const Shape &shape, uint8_t backend) {
    auto t = std::make_shared<HostTensor>(shape, backend);
    const int64_t n = shape.numel();
    if (n <= kInline) {
        t->p = t->small;
    } else {
        t->own.reset(new T[n]);
        t->p = t->own.get();
    }
    return t;
}

template <class T>
std::shared_ptr<HostTensor<T>> HostTensor<T>::borrow(const T *data, const Shape &shape,
                                                     uint8_t backend,
                                                     std::shared_ptr<const void> owner) {
    auto t = std::make_shared<HostTensor>(shape, backend);
    t->p = const_cast<T *>(data); // never written: borrowed memory is not owned
    t->borrowed = std::move(owner);
    return t;
}

// The values as doubles, which every float32 and float64 is exactly.
template <class T>
std::vector<double> HostTensor<T>::data() const {
    return std::vector<double>(p, p + shape.numel());
}

template <class T>
Tensor::Ptr HostTensor<T>::to(Device target) const {
    if (target.type != Device::Type::cpu)
        CORA_FAIL(Error::wrong_value, "CoraTensor: the host backends run on the CPU only");
    return self();
}

// A copy in the other dtype; lean has float64 only.
template <class T>
Tensor::Ptr HostTensor<T>::to(DType target) const {
    if (target == dtype) return self();
    if (backend == backend_id::lean)
        CORA_FAIL(Error::not_supported, "CoraTensor: the lean backend computes in float64 only");
    const int64_t n = shape.numel();
    if (target == DType::float32) {
        auto r = HostTensor<float>::make(shape, backend);
        for (int64_t i = 0; i < n; ++i) r->p[i] = static_cast<float>(p[i]);
        return r;
    }
    auto r = HostTensor<double>::make(shape, backend);
    for (int64_t i = 0; i < n; ++i) r->p[i] = static_cast<double>(p[i]);
    return r;
}

// New tensors of this backend and dtype.
template <class T>
Tensor::Ptr HostTensor<T>::like(const double *values, const Shape &s) const {
    auto r = fresh(s);
    const int64_t n = s.numel();
    for (int64_t i = 0; i < n; ++i) r->p[i] = static_cast<T>(values[i]);
    return r;
}

template <class T>
Tensor::Ptr HostTensor<T>::full(const Shape &s, double value) const {
    auto r = fresh(s);
    std::fill(r->p, r->p + s.numel(), static_cast<T>(value));
    return r;
}

template <class T>
Tensor::Ptr HostTensor<T>::arange(int64_t n) const {
    auto r = fresh(Shape{n});
    for (int64_t i = 0; i < n; ++i) r->p[i] = T(i);
    return r;
}

// Elementwise -------------------------------------------------------------------------------------

// Into new memory of the broadcast shape.
template <class T>
Tensor::Ptr HostTensor<T>::binary(Binary op, const Tensor::Impl &o) const {
    const Shape out = shape == o.shape ? shape : broadcast_shapes(shape, o.shape);
    auto r = fresh(out);
    aux_with_binary<T>(op, [&](auto f) { aux_binary_into(*this, of(o), r->p, out, f); });
    return r;
}

template <class T>
Tensor::Ptr HostTensor<T>::scalar(Binary op, double s, bool scalar_first) const {
    auto r = fresh(shape);
    const T v = static_cast<T>(s);
    const int64_t n = shape.numel();
    aux_with_binary<T>(op, [&](auto f) {
        if (scalar_first)
            for (int64_t i = 0; i < n; ++i) r->p[i] = f(v, p[i]);
        else
            for (int64_t i = 0; i < n; ++i) r->p[i] = f(p[i], v);
    });
    return r;
}

// The C library's function of every entry.
template <class T>
Tensor::Ptr HostTensor<T>::unary(Unary op) const {
    auto r = fresh(shape);
    const int64_t n = shape.numel();
    aux_with_unary<T>(op, [&](auto f) {
        for (int64_t i = 0; i < n; ++i) r->p[i] = f(p[i]);
    });
    return r;
}

// a, b and the condition broadcast to one shape.
template <class T>
Tensor::Ptr HostTensor<T>::where(const Tensor::Impl &a, const Tensor::Impl &b) const {
    const Shape out = broadcast_shapes(broadcast_shapes(shape, a.shape), b.shape);
    auto r = fresh(out);
    const std::array<Shape, 4> st = {aux_broadcast_strides(shape, out),
                                     aux_broadcast_strides(a.shape, out),
                                     aux_broadcast_strides(b.shape, out), strides_of(out)};
    const T *pa = of(a).p, *pb = of(b).p;
    aux_walk<4>(out, st, [&](int64_t ic, int64_t ia, int64_t ib, int64_t io) {
        r->p[io] = p[ic] != T(0) ? pa[ia] : pb[ib];
    });
    return r;
}

// In place: only on memory of its own, and only when the result keeps this shape.
template <class T>
bool HostTensor<T>::binary_inplace(Binary op, const Tensor::Impl &o) {
    if (!owns() || broadcast_shapes(shape, o.shape) != shape) return false;
    aux_with_binary<T>(op, [&](auto f) { aux_binary_into(*this, of(o), p, shape, f); });
    return true;
}

template <class T>
bool HostTensor<T>::scalar_inplace(Binary op, double s, bool scalar_first) {
    if (!owns()) return false;
    const T v = static_cast<T>(s);
    const int64_t n = shape.numel();
    aux_with_binary<T>(op, [&](auto f) {
        if (scalar_first)
            for (int64_t i = 0; i < n; ++i) p[i] = f(v, p[i]);
        else
            for (int64_t i = 0; i < n; ++i) p[i] = f(p[i], v);
    });
    return true;
}

// Every entry replaced by its value.
template <class T>
bool HostTensor<T>::unary_inplace(Unary op) {
    if (!owns()) return false;
    const int64_t n = shape.numel();
    aux_with_unary<T>(op, [&](auto f) {
        for (int64_t i = 0; i < n; ++i) p[i] = f(p[i]);
    });
    return true;
}

// Reductions --------------------------------------------------------------------------------------

// Sums and products nest to the right from 0 and 1; max, min and their indices take the first.
template <class T>
Tensor::Ptr HostTensor<T>::reduce(Reduce op, int axis, bool keepdims) const {
    axis = axis_of(axis, shape.size(), "reduce");
    if (backend == backend_id::eigen && op == Reduce::sum && axis + 1 == shape.size())
        return eigen::sum_last(*this, keepdims);
    const Extents e = aux_extents(shape, axis);
    auto r = fresh(aux_reduced(shape, axis, keepdims));
    if (e.len == 0 && op != Reduce::sum && op != Reduce::prod)
        CORA_FAIL(Error::empty_set, "CoraTensor: the max, min or arg of an empty axis");
    T *none = nullptr;
    const auto above = [](T a, T b) { return a > b; };
    const auto below = [](T a, T b) { return a < b; };
    switch (op) {
    case Reduce::sum:
        aux_fold_right(p, e, T(0), r->p, [](T x, T acc) { return x + acc; });
        break;
    case Reduce::prod:
        aux_fold_right(p, e, T(1), r->p, [](T x, T acc) { return x * acc; });
        break;
    // the extrema and their indices
    case Reduce::max:
        aux_select(p, e, r->p, none, above);
        break;
    case Reduce::min:
        aux_select(p, e, r->p, none, below);
        break;
    case Reduce::argmax:
        aux_select(p, e, none, r->p, above);
        break;
    case Reduce::argmin:
        aux_select(p, e, none, r->p, below);
        break;
    }
    return r;
}

// The running sum from the first entry: c_0 = x_0, c_l = c_{l-1} + x_l.
template <class T>
Tensor::Ptr HostTensor<T>::cumsum(int axis) const {
    axis = axis_of(axis, shape.size(), "cumsum");
    const Extents e = aux_extents(shape, axis);
    auto r = fresh(shape);
    for (int64_t o = 0; o < e.outer; ++o)
        for (int64_t l = 0; l < e.len; ++l) {
            const int64_t at = (o * e.len + l) * e.inner;
            for (int64_t i = 0; i < e.inner; ++i)
                r->p[at + i] = l == 0 ? p[at + i] : r->p[at - e.inner + i] + p[at + i];
        }
    return r;
}

// Shape and Selection -----------------------------------------------------------------------------

// A view of the same memory: the new tensor keeps this one alive.
template <class T>
Tensor::Ptr HostTensor<T>::reshape(const Shape &s) const {
    if (s.numel() != shape.numel())
        CORA_FAIL(Error::dimension_mismatch, "CoraTensor::reshape: " + aux_text(shape) +
                                                 " has another number of entries than " +
                                                 aux_text(s));
    return borrow(p, s, backend, self());
}

// A copy that walks the source by permuted strides.
template <class T>
Tensor::Ptr HostTensor<T>::permute_dims(const std::vector<int> &axes) const {
    const Shape st = strides_of(shape);
    Shape out, strides;
    for (const int a : axes) {
        out.push_back(shape[a]);
        strides.push_back(st[a]);
    }
    auto r = fresh(out);
    aux_gather(p, 0, strides, out, r->p);
    return r;
}

// A copy that walks the source with stride zero along the broadcast axes.
template <class T>
Tensor::Ptr HostTensor<T>::broadcast_to(const Shape &s) const {
    if (broadcast_shapes(shape, s) != s)
        CORA_FAIL(Error::dimension_mismatch, "CoraTensor::broadcast_to: " + aux_text(shape) +
                                                 " does not broadcast to " + aux_text(s));
    if (s == shape) return self();
    auto r = fresh(s);
    aux_gather(p, 0, aux_broadcast_strides(shape, s), s, r->p);
    return r;
}

// A copy by strides: the axis's stride times the step, from the start.
template <class T>
Tensor::Ptr HostTensor<T>::slice(int axis, int64_t start, int64_t stop, int64_t step) const {
    const Shape st = strides_of(shape);
    Shape out = shape, strides = st;
    out[axis] = std::max<int64_t>(0, (stop - start + step - 1) / step);
    strides[axis] = st[axis] * step;
    auto r = fresh(out);
    aux_gather(p, start * st[axis], strides, out, r->p);
    return r;
}

// Whole blocks of the axes after `axis` are copied per index.
template <class T>
Tensor::Ptr HostTensor<T>::take(const Tensor::Impl &idx, int axis) const {
    const Extents e = aux_extents(shape, axis);
    const int64_t k = idx.shape.numel();
    const T *ix = of(idx).p;
    Shape out = shape;
    out[axis] = k;
    auto r = fresh(out);
    for (int64_t o = 0; o < e.outer; ++o)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t l = static_cast<int64_t>(ix[j]);
            if (l < 0 || l >= e.len)
                CORA_FAIL(Error::wrong_value, "CoraTensor::take: the index " + std::to_string(l) +
                                                  " is outside 0.." + std::to_string(e.len - 1));
            // a single entry per index (the last axis) is copied without a call
            if (e.inner == 1)
                r->p[o * k + j] = p[o * e.len + l];
            else
                std::copy_n(p + (o * e.len + l) * e.inner, e.inner, r->p + (o * k + j) * e.inner);
        }
    return r;
}

// out[pos] = x[pos with the axis at idx[pos]], idx broadcast over the other axes.
template <class T>
Tensor::Ptr HostTensor<T>::take_along_axis(const Tensor::Impl &idx, int axis) const {
    Shape base = shape;
    base[axis] = idx.shape[axis];
    const Shape out = broadcast_shapes(base, idx.shape);
    Shape xs = aux_broadcast_strides(shape, out);
    const int64_t axis_stride = strides_of(shape)[axis];
    xs[axis] = 0;
    const std::array<Shape, 3> st = {xs, aux_broadcast_strides(idx.shape, out), strides_of(out)};
    auto r = fresh(out);
    const T *ix = of(idx).p;
    const int64_t len = shape[axis];
    aux_walk<3>(out, st, [&](int64_t ixx, int64_t ii, int64_t io) {
        const int64_t l = static_cast<int64_t>(ix[ii]);
        if (l < 0 || l >= len)
            CORA_FAIL(Error::wrong_value, "CoraTensor::take_along_axis: the index " +
                                              std::to_string(l) + " is outside 0.." +
                                              std::to_string(len - 1));
        r->p[io] = p[ixx + l * axis_stride];
    });
    // the concatenated result
    return r;
}

// The parts' blocks follow each other within every outer position.
template <class T>
Tensor::Ptr HostTensor<T>::concat(std::span<const Tensor::Impl *const> parts, int axis) const {
    Shape out = shape;
    out[axis] = 0;
    for (const Tensor::Impl *q : parts) {
        Shape probe = q->shape;
        probe[axis] = shape[axis];
        if (probe != shape)
            CORA_FAIL(Error::dimension_mismatch,
                      "CoraTensor::concat: " + aux_text(shape) + " and " + aux_text(q->shape) +
                          " differ outside the axis " + std::to_string(axis));
        out[axis] += q->shape[axis];
    }
    const Extents e = aux_extents(out, axis);
    auto r = fresh(out);
    // per outer position, each part's block in turn
    for (int64_t o = 0; o < e.outer; ++o) {
        T *at = r->p + o * e.len * e.inner;
        for (const Tensor::Impl *q : parts) {
            const int64_t block = q->shape[axis] * e.inner;
            at = std::copy_n(of(*q).p + o * block, block, at);
        }
    }
    return r;
}

// A stable sort of each line's indices; descending keeps equal entries in index order too.
template <class T>
Tensor::Ptr HostTensor<T>::argsort(int axis, bool descending) const {
    const Extents e = aux_extents(shape, axis);
    auto r = fresh(shape);
    std::vector<int64_t> order(e.len);
    std::vector<T> line(e.len);
    for (int64_t o = 0; o < e.outer; ++o)
        for (int64_t i = 0; i < e.inner; ++i) {
            for (int64_t l = 0; l < e.len; ++l) line[l] = p[(o * e.len + l) * e.inner + i];
            std::iota(order.begin(), order.end(), 0);
            if (descending)
                std::stable_sort(order.begin(), order.end(),
                                 [&](int64_t a, int64_t b) { return line[a] > line[b]; });
            else
                std::stable_sort(order.begin(), order.end(),
                                 [&](int64_t a, int64_t b) { return line[a] < line[b]; });
            for (int64_t l = 0; l < e.len; ++l) r->p[(o * e.len + l) * e.inner + i] = T(order[l]);
        }
    return r;
}

// Linear Algebra ----------------------------------------------------------------------------------

// Each entry nests its sum to the right from zero; the loop over j keeps that order per entry.
template <class T>
Tensor::Ptr HostTensor<T>::matmul(const Tensor::Impl &o) const {
    if (backend == backend_id::eigen) return eigen::matmul(*this, of(o));
    const BatchPairs bp = batch_pairs(shape, o.shape, "matmul");
    const int64_t n = shape[shape.size() - 2], k = shape.back(), m = o.shape.back();
    if (o.shape[o.shape.size() - 2] != k)
        CORA_FAIL(Error::dimension_mismatch, "CoraTensor::matmul: " + aux_text(shape) + " times " +
                                                 aux_text(o.shape) + ": inner sizes differ");
    auto r = fresh(aux_with_matrix(bp.batch, n, m));
    const T *b = of(o).p;
    parallel_for(bp.batch.numel(), (long long)bp.batch.numel() * n * k * m, [&](int64_t z) {
        const auto [left, right] = bp.at(z);
        const T *A = p + left * n * k, *B = b + right * k * m;
        T *C = r->p + z * n * m;
        for (int64_t i = 0; i < n; ++i) {
            // row i of C, from the last term to the first: a_iq b_qj + (the later terms)
            T *row = C + i * m;
            std::fill(row, row + m, T(0));
            for (int64_t q = k - 1; q >= 0; --q) {
                const T a = A[i * k + q];
                const T *brow = B + q * m;
                for (int64_t j = 0; j < m; ++j) row[j] = a * brow[j] + row[j];
            }
        }
    });
    return r;
}

// LU with partial pivoting per matrix, as LAPACK's gesv and torch.
template <class T>
Tensor::Ptr HostTensor<T>::solve(const Tensor::Impl &rhs) const {
    aux_require_matrix(shape, "solve", true);
    const BatchPairs bp = batch_pairs(shape, rhs.shape, "solve");
    const int64_t n = shape.back(), m = rhs.shape.back();
    if (rhs.shape[rhs.shape.size() - 2] != n)
        CORA_FAIL(Error::dimension_mismatch, "CoraTensor::solve: A " + aux_text(shape) + " and B " +
                                                 aux_text(rhs.shape) + " differ in rows");
    auto r = fresh(aux_with_matrix(bp.batch, n, m));
    const T *b = of(rhs).p;
    parallel_for(bp.batch.numel(), (long long)bp.batch.numel() * n * n * (n + m), [&](int64_t z) {
        const auto [left, right] = bp.at(z);
        Map<T>(r->p + z * n * m, n, m) =
            aux_matrix(p, left, n, n).partialPivLu().solve(aux_matrix(b, right, n, m));
    });
    return r;
}

// The lower factor; a matrix that is not positive definite is an error.
template <class T>
Tensor::Ptr HostTensor<T>::cholesky() const {
    aux_require_matrix(shape, "cholesky", true);
    const int64_t n = shape.back(), count = aux_matrices(shape);
    auto r = fresh(shape);
    for (int64_t z = 0; z < count; ++z) {
        const Eigen::LLT<RowMat<T>> llt(aux_matrix(p, z, n, n));
        if (llt.info() != Eigen::Success)
            CORA_FAIL(Error::wrong_value,
                      "CoraTensor::cholesky: the matrix is not positive definite");
        Map<T>(r->p + z * n * n, n, n) = llt.matrixL();
    }
    return r;
}

// Householder QR, reduced: Q (n, k), R (k, m).
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr> HostTensor<T>::qr() const {
    aux_require_matrix(shape, "qr", false);
    const int64_t n = shape[shape.size() - 2], m = shape.back(), k = std::min(n, m);
    const int64_t count = aux_matrices(shape);
    const Shape batch = aux_batch(shape);
    auto Q = fresh(aux_with_matrix(batch, n, k)), R = fresh(aux_with_matrix(batch, k, m));
    for (int64_t z = 0; z < count; ++z) {
        const Eigen::HouseholderQR<RowMat<T>> qr(aux_matrix(p, z, n, m));
        Map<T>(Q->p + z * n * k, n, k) = qr.householderQ() * RowMat<T>::Identity(n, k);
        Map<T>(R->p + z * k * m, k, m) =
            qr.matrixQR().topRows(k).template triangularView<Eigen::Upper>();
    }
    return {Q, R};
}

// One-sided Jacobi, the most accurate of Eigen's; thin U and V.
template <class T>
std::array<Tensor::Ptr, 3> HostTensor<T>::svd() const {
    aux_require_matrix(shape, "svd", false);
    const int64_t n = shape[shape.size() - 2], m = shape.back(), k = std::min(n, m);
    const int64_t count = aux_matrices(shape);
    const Shape batch = aux_batch(shape);
    auto U = fresh(aux_with_matrix(batch, n, k)), S = fresh(aux_with_matrix(batch, k, 1)),
         V = fresh(aux_with_matrix(batch, m, k));
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    for (int64_t z = 0; z < count; ++z) {
        const Mat A = aux_matrix(p, z, n, m);
#if EIGEN_VERSION_AT_LEAST(3, 4, 90)
        const Eigen::JacobiSVD<Mat, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(A);
#else
        const Eigen::JacobiSVD<Mat> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
#endif
        Map<T>(U->p + z * n * k, n, k) = svd.matrixU();
        Map<T>(S->p + z * k, k, 1) = svd.singularValues();
        Map<T>(V->p + z * m * k, m, k) = svd.matrixV();
    }
    // the factors of the decomposition
    return {U, S, V};
}

// The symmetric eigensolver: ascending eigenvalues.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr> HostTensor<T>::eigh() const {
    aux_require_matrix(shape, "eigh", true);
    const int64_t n = shape.back(), count = aux_matrices(shape);
    auto W = fresh(aux_with_matrix(aux_batch(shape), n, 1)), V = fresh(shape);
    for (int64_t z = 0; z < count; ++z) {
        const Eigen::SelfAdjointEigenSolver<RowMat<T>> es(aux_matrix(p, z, n, n));
        Map<T>(W->p + z * n, n, 1) = es.eigenvalues();
        Map<T>(V->p + z * n * n, n, n) = es.eigenvectors();
    }
    return {W, V};
}

// The column's entries on the diagonal, zeros elsewhere.
template <class T>
Tensor::Ptr HostTensor<T>::diag() const {
    const int64_t n = shape[shape.size() - 2], count = aux_matrices(shape);
    auto r = fresh(aux_with_matrix(aux_batch(shape), n, n));
    std::fill(r->p, r->p + count * n * n, T(0));
    for (int64_t z = 0; z < count; ++z)
        for (int64_t i = 0; i < n; ++i) r->p[z * n * n + i * n + i] = p[z * n + i];
    return r;
}

// Derived and Fused ------------------------------------------------------------------------------

// CORALean's exponential is an interval Taylor series, which no point function reproduces.
template <class T>
Tensor::Ptr HostTensor<T>::expm() const {
    if (backend == backend_id::eigen) return eigen::expm(*this);
    CORA_FAIL(Error::special_error,
              "CoraTensor: the lean backend has no expm; CORALean's exponential is an interval "
              "Taylor series (1/i! as bounds), which no point function reproduces");
}

template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr>
HostTensor<T>::zonotope_linear_map(const Tensor::Impl &c, const Tensor::Impl &G) const {
    if (backend == backend_id::eigen) return eigen::zonotope_linear_map(*this, of(c), of(G));
    return Impl::zonotope_linear_map(c, G);
}

// The fused primitives: Eigen's kernels on eigen (one batch), the specification on lean.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr>
HostTensor<T>::zonotope_interval_map(const Tensor::Impl &sup, const Tensor::Impl &c,
                                     const Tensor::Impl &G) const {
    if (backend == backend_id::eigen && aux_batch(c.shape) == aux_batch(G.shape))
        return eigen::zonotope_interval_map(*this, of(sup), of(c), of(G));
    return Impl::zonotope_interval_map(sup, c, G);
}

template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr>
HostTensor<T>::zonotope_linear_combination(const Tensor::Impl &G1, const Tensor::Impl &c2,
                                           const Tensor::Impl &G2) const {
    if (backend == backend_id::eigen) // the front end gave the centers the generators' batch
        return eigen::zonotope_linear_combination(*this, of(G1), of(c2), of(G2));
    return Impl::zonotope_linear_combination(G1, c2, G2);
}

// Both interval products are one kernel, with the interval on the left or on the right.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr> HostTensor<T>::interval_matmul(const Tensor::Impl &sup,
                                                                   const Tensor::Impl &M) const {
    if (backend == backend_id::eigen) return eigen::interval_matmul(*this, of(sup), of(M), false);
    return Impl::interval_matmul(sup, M);
}

template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr> HostTensor<T>::matmul_interval(const Tensor::Impl &inf,
                                                                   const Tensor::Impl &sup) const {
    if (backend == backend_id::eigen) return eigen::interval_matmul(of(inf), of(sup), *this, true);
    return Impl::matmul_interval(inf, sup);
}

// Eigen's scores of all columns at once on eigen; the specification on lean.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr> HostTensor<T>::girard_split(int64_t keep) const {
    if (backend == backend_id::eigen) return eigen::girard_split(*this, keep);
    return Impl::girard_split(keep);
}

template <class T>
Tensor::Ptr HostTensor<T>::abs_sum() const {
    if (backend == backend_id::eigen) return eigen::abs_sum(*this);
    return Impl::abs_sum();
}

template struct HostTensor<double>;
template struct HostTensor<float>;

} // namespace cora::host

// ---------------------------------------  END OF CODE  ---------------------------------------- //
