// eigen - the Eigen backend of Tensor: float64 or float32 tensors of any shape on the CPU
//
// The tensors are HostTensors (host_tensor.h) of the eigen id; the products, the sums along the
// last axis, the matrix exponential and the fused primitives are Eigen's kernels here, matrix by
// matrix over the batch. Nothing else in the library includes Eigen for its own use.

#include "cora/global/backend/eigen.h"
#include "cora/global/backend/host.h"
#include "cora/global/error.h"
#include "global/backend/host_tensor.h"

#include <algorithm>
#include <numeric>

#include <unsupported/Eigen/MatrixFunctions>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

template <class T> using RowMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <class T> using MapC = Eigen::Map<const RowMat<T>>;
template <class T> using Map = Eigen::Map<RowMat<T>>;
template <class T> using H = host::HostTensor<T>;

/// The rows and columns of the matrices of a shape.
int64_t aux_rows(const Shape &s) { return s[s.size() - 2]; }

/// The shape of the batch `bp` with matrices (rows, cols).
Shape aux_matrix_shape(const Shape &batch, int64_t rows, int64_t cols) {
    Shape s = batch;
    s.push_back(rows);
    s.push_back(cols);
    return s;
}

/// out = a x b for every pair of the batch, each one Eigen product.
template <class T>
void aux_products(const T *a, const Shape &sa, const T *b, const Shape &sb, T *out,
                  const host::BatchPairs &bp) {
    const int64_t n = aux_rows(sa), k = sa.back(), m = sb.back();
    host::parallel_for(bp.batch.numel(), (long long)bp.batch.numel() * n * k * m, [&](int64_t z) {
        Map<T>(out + z * n * m, n, m).noalias() =
            MapC<T>(a + bp.left[z] * n * k, n, k) * MapC<T>(b + bp.right[z] * k * m, k, m);
    });
}

/// The batch pairs of a product, with its inner sizes checked.
host::BatchPairs aux_pairs(const Shape &a, const Shape &b, const char *op) {
    host::BatchPairs bp = host::batch_pairs(a, b, op);
    if (aux_rows(b) != a.back())
        CORA_FAIL(Error::dimension_mismatch, std::string("CoraTensor::") + op +
                                                 ": the inner sizes of the product differ");
    return bp;
}

/// Makes Eigen tensors from host numbers, on the CPU only.
struct EigenBackend : Tensor::Backend {
    std::string name() const override { return "eigen"; }
    Tensor::Ptr make(const std::vector<double> &data, const Shape &shape,
                     const std::string &device, DType dtype) const override {
        if (!device.empty() && device != "cpu")
            CORA_FAIL(Error::wrong_value, "CoraTensor: the eigen backend runs on the CPU only");
        if (dtype == DType::float32) {
            auto t = H<float>::make(shape, backend_id::eigen);
            std::copy(data.begin(), data.end(), t->p);
            return t;
        }
        auto t = H<double>::make(shape, backend_id::eigen);
        std::copy(data.begin(), data.end(), t->p);
        return t;
    }
};

} // namespace


// ===========================================  MAIN  =========================================== //

// Kernels -----------------------------------------------------------------------------------------

namespace host::eigen {

template <class T> Tensor::Ptr matmul(const HostTensor<T> &a, const HostTensor<T> &b) {
    const BatchPairs bp = aux_pairs(a.shape, b.shape, "matmul");
    auto r = a.fresh(aux_matrix_shape(bp.batch, aux_rows(a.shape), b.shape.back()));
    aux_products(a.p, a.shape, b.p, b.shape, r->p, bp);
    return r;
}

// All rows of all matrices at once: one Eigen row sum.
template <class T> Tensor::Ptr sum_last(const HostTensor<T> &a, bool keepdims) {
    const int64_t len = a.shape.back(), rows = a.shape.numel() / std::max<int64_t>(len, 1);
    Shape out = a.shape;
    if (keepdims) out.back() = 1;
    else out = Shape(std::vector<int64_t>(a.shape.begin(), a.shape.end() - 1));
    auto r = a.fresh(out);
    if (len == 0) std::fill(r->p, r->p + rows, T(0));
    else Map<T>(r->p, rows, 1) = MapC<T>(a.p, rows, len).rowwise().sum();
    return r;
}

template <class T> Tensor::Ptr expm(const HostTensor<T> &a) {
    const int64_t n = a.shape.back(), count = a.shape.numel() / std::max<int64_t>(n * n, 1);
    auto r = a.fresh(a.shape);
    for (int64_t z = 0; z < count; ++z) {
        const RowMat<T> m = MapC<T>(a.p + z * n * n, n, n);
        Map<T>(r->p + z * n * n, n, n) = m.exp();
    }
    return r;
}

// M c and M G, two products of one call.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr> zonotope_linear_map(const HostTensor<T> &M,
                                                        const HostTensor<T> &c,
                                                        const HostTensor<T> &G) {
    return {matmul(M, c), matmul(M, G)};
}

// With M+ = max(M, 0) and M- = min(M, 0): I M = [inf M+ + sup M-, sup M+ + inf M-], and
// M I = [M+ inf + M- sup, M+ sup + M- inf]: each term of CORA's sum is the same rounded product.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr> interval_matmul(const HostTensor<T> &inf,
                                                    const HostTensor<T> &sup,
                                                    const HostTensor<T> &M, bool point_left) {
    const int64_t count = M.shape.numel();
    const Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> m(M.p, count);
    const Eigen::Array<T, Eigen::Dynamic, 1> mp = m.max(T(0)), mn = m.min(T(0));
    auto Mp = M.fresh(M.shape), Mn = M.fresh(M.shape);
    std::copy_n(mp.data(), count, Mp->p);
    std::copy_n(mn.data(), count, Mn->p);
    const HostTensor<T> &first = point_left ? *Mp : inf;
    const BatchPairs bp = point_left ? aux_pairs(M.shape, inf.shape, "matmul_interval")
                                     : aux_pairs(inf.shape, M.shape, "interval_matmul");
    const Shape &left = point_left ? M.shape : inf.shape, &right = point_left ? inf.shape : M.shape;
    const Shape out = aux_matrix_shape(bp.batch, aux_rows(left), right.back());
    auto lo = first.fresh(out), hi = first.fresh(out), tmp = first.fresh(out);
    const int64_t n = out.numel();
    // lo = a1 b1 + a2 b2 and hi = a3 b3 + a4 b4 for the four products of each side
    const auto both = [&](const T *a1, const T *b1, const T *a2, const T *b2, T *dst) {
        aux_products(a1, left, b1, right, dst, bp);
        aux_products(a2, left, b2, right, tmp->p, bp);
        for (int64_t i = 0; i < n; ++i) dst[i] = dst[i] + tmp->p[i];
    };
    if (point_left) {
        both(Mp->p, inf.p, Mn->p, sup.p, lo->p);
        both(Mp->p, sup.p, Mn->p, inf.p, hi->p);
    } else {
        both(inf.p, Mp->p, sup.p, Mn->p, lo->p);
        both(sup.p, Mp->p, inf.p, Mn->p, hi->p);
    }
    return {lo, hi};
}

// The scores of all columns at once, then a stable sort per matrix.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr> girard_split(const HostTensor<T> &G, int64_t keep) {
    const int64_t n = aux_rows(G.shape), m = G.shape.back(), count = G.shape.numel() / (n * m);
    Shape batch(std::vector<int64_t>(G.shape.begin(), G.shape.end() - 2));
    auto kept = G.fresh(aux_matrix_shape(batch, 1, keep)),
         rest = G.fresh(aux_matrix_shape(batch, 1, m - keep));
    std::vector<int64_t> order(m);
    for (int64_t z = 0; z < count; ++z) {
        const auto A = MapC<T>(G.p + z * n * m, n, m).cwiseAbs();
        const Eigen::Matrix<T, 1, Eigen::Dynamic> score =
            A.colwise().sum() - A.colwise().maxCoeff();
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int64_t a, int64_t b) { return score(a) > score(b); });
        for (int64_t j = 0; j < keep; ++j) kept->p[z * keep + j] = T(order[j]);
        for (int64_t j = keep; j < m; ++j) rest->p[z * (m - keep) + j - keep] = T(order[j]);
    }
    return {kept, rest};
}

// All rows of all matrices at once: one Eigen row sum of |a|.
template <class T> Tensor::Ptr abs_sum(const HostTensor<T> &a) {
    const int64_t len = a.shape.back(), rows = a.shape.numel() / std::max<int64_t>(len, 1);
    Shape out = a.shape;
    out.back() = 1;
    auto r = a.fresh(out);
    if (len == 0) std::fill(r->p, r->p + rows, T(0));
    else Map<T>(r->p, rows, 1) = MapC<T>(a.p, rows, len).cwiseAbs().rowwise().sum();
    return r;
}

// The kernels of both dtypes.
#define CORA_EIGEN_KERNELS(T)                                                                      \
    template Tensor::Ptr matmul(const HostTensor<T> &, const HostTensor<T> &);                     \
    template Tensor::Ptr sum_last(const HostTensor<T> &, bool);                                    \
    template Tensor::Ptr expm(const HostTensor<T> &);                                              \
    template std::pair<Tensor::Ptr, Tensor::Ptr> zonotope_linear_map(                              \
        const HostTensor<T> &, const HostTensor<T> &, const HostTensor<T> &);                      \
    template std::pair<Tensor::Ptr, Tensor::Ptr> interval_matmul(                                  \
        const HostTensor<T> &, const HostTensor<T> &, const HostTensor<T> &, bool);                \
    template std::pair<Tensor::Ptr, Tensor::Ptr> girard_split(const HostTensor<T> &, int64_t);     \
    template Tensor::Ptr abs_sum(const HostTensor<T> &);
CORA_EIGEN_KERNELS(double)
CORA_EIGEN_KERNELS(float)
#undef CORA_EIGEN_KERNELS

} // namespace host::eigen

// Conversion --------------------------------------------------------------------------------------

std::shared_ptr<const Tensor::Backend> eigen_backend() { return std::make_shared<EigenBackend>(); }

Tensor from_eigen(const Eigen::MatrixXd &m) {
    auto t = H<double>::make(Shape{m.rows(), m.cols()}, backend_id::eigen);
    Map<double>(t->p, m.rows(), m.cols()) = m;
    return Tensor(t);
}

Eigen::MatrixXd to_eigen(const Tensor &t) {
    if (!is_host(t) || t.ndim() != 2 || t.dtype() != DType::float64)
        CORA_FAIL(Error::wrong_value, "to_eigen: takes a float64 matrix of a host backend");
    const auto &h = static_cast<const H<double> &>(t.impl());
    return MapC<double>(h.p, h.shape[0], h.shape[1]);
}

bool is_eigen(const Tensor &t) { return t.impl().backend == backend_id::eigen; }

// Host Memory -------------------------------------------------------------------------------------

bool is_host(const Tensor &t) {
    return t.impl().backend == backend_id::eigen || t.impl().backend == backend_id::lean;
}

Tensor host_borrow(const void *data, DType dtype, const Shape &shape,
                   std::shared_ptr<const void> owner, uint8_t backend) {
    if (backend == backend_id::lean && dtype != DType::float64)
        CORA_FAIL(Error::not_supported, "CoraTensor: the lean backend computes in float64 only");
    if (dtype == DType::float32)
        return Tensor(H<float>::borrow(static_cast<const float *>(data), shape, backend, owner));
    return Tensor(H<double>::borrow(static_cast<const double *>(data), shape, backend, owner));
}

// Unique: the memory is the tensor's own and the caller's tensor is its only holder.
HostView host_view(const Tensor &t) {
    if (!is_host(t)) CORA_FAIL(Error::wrong_value, "host_view: not a tensor of a host backend");
    const bool alone = t.ptr().use_count() == 1;
    if (t.dtype() == DType::float32) {
        const auto &h = static_cast<const H<float> &>(t.impl());
        return {h.p, h.dtype, h.shape, t.ptr(), alone && h.owns()};
    }
    const auto &h = static_cast<const H<double> &>(t.impl());
    return {h.p, h.dtype, h.shape, t.ptr(), alone && h.owns()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
