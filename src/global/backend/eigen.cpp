// eigen - the Eigen backend of Tensor: float64 or float32 tensors of any shape on the CPU
//
// The tensors are HostTensors (host_tensor.h) of the eigen id; the products, the sums along the
// last axis, the matrix exponential and the fused primitives are Eigen's kernels here, matrix by
// matrix over the batch. Nothing else in the library includes Eigen for its own use.

#include "cora/global/backend/eigen.h"
#include "cora/global/backend/host.h"
#include "cora/global/error.h"
#include "cora/global/threads.h"
#include "global/backend/host_tensor.h"

#include <algorithm>
#include <numeric>
#include <optional>

#include <unsupported/Eigen/MatrixFunctions>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

template <class T>
using RowMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <class T>
using MapC = Eigen::Map<const RowMat<T>>;
template <class T>
using Map = Eigen::Map<RowMat<T>>;
template <class T>
using H = host::HostTensor<T>;

/// The rows and columns of the matrices of a shape.
int64_t aux_rows(const Shape &s) { return s[s.size() - 2]; }

/// The shape of the batch `bp` with matrices (rows, cols).
Shape aux_matrix_shape(const Shape &batch, int64_t rows, int64_t cols) {
    Shape s = batch;
    s.push_back(rows);
    s.push_back(cols);
    return s;
}

/// Below this many multiplications a product is a plain loop: Eigen's setup costs more.
constexpr int64_t kSmallProduct = 512;
/// Below this much work (multiplications) Eigen keeps a product on one thread: 2 x its 50000 task.
constexpr long long kSerialWork = 100000;

/// C = A B of one pair, (n, k) times (k, m). A small product is a plain loop over the rows of B
/// weighted by a row of A: no setup cost.
template <class T>
void aux_product(const T *A, const T *B, T *C, int64_t n, int64_t k, int64_t m) {
    if (n * k * m > kSmallProduct) {
        Map<T>(C, n, m).noalias() = MapC<T>(A, n, k) * MapC<T>(B, k, m);
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        T *row = C + i * m;
        std::fill(row, row + m, T(0));
        for (int64_t q = 0; q < k; ++q) {
            const T aiq = A[i * k + q];
            for (int64_t j = 0; j < m; ++j) row[j] += aiq * B[q * m + j];
        }
    }
}

/// out = a x b for every pair of the batch, each one Eigen product. Eigen gets the threads the
/// work can keep busy: on row-major maps it would split even a (2, 200) x (200, 200) product.
template <class T>
void aux_products(const T *a, const Shape &sa, const T *b, const Shape &sb, T *out,
                  const host::BatchPairs &bp) {
    const int64_t n = aux_rows(sa), k = sa.back(), m = sb.back();
    const long long work = (long long)bp.batch.numel() * n * k * m;
    const auto product = [&](int64_t z) {
        const auto [left, right] = bp.at(z);
        aux_product(a + left * n * k, b + right * k * m, out + z * n * m, n, k, m);
    };
    // work Eigen would not split anyway runs here, without asking for threads (the first ask reads
    // the CPU topology)
    if (work < kSerialWork) {
        for (int64_t z = 0; z < bp.batch.numel(); ++z) product(z);
        return;
    }
    const EigenThreads narrow(work);
    host::parallel_for(bp.batch.numel(), work, product);
}

/// The batch pairs of a product, with its inner sizes checked.
host::BatchPairs aux_pairs(const Shape &a, const Shape &b, const char *op) {
    host::BatchPairs bp = host::batch_pairs(a, b, op);
    if (aux_rows(b) != a.back())
        CORA_FAIL(Error::dimension_mismatch,
                  std::string("CoraTensor::") + op + ": the inner sizes of the product differ");
    return bp;
}

/// Memory of this thread for the temporaries of a kernel, reused from call to call so that a
/// kernel allocates its results only.
template <class T>
T *aux_scratch(int64_t count) {
    thread_local std::vector<T> buffer;
    if (static_cast<int64_t>(buffer.size()) < count) buffer.resize(count);
    return buffer.data();
}

/// Makes Eigen tensors from host numbers, on the CPU only.
struct EigenBackend : Tensor::Backend {
    std::string name() const override { return "eigen"; }
    Tensor::Ptr make(const std::vector<double> &data, const Shape &shape, const std::string &device,
                     DType dtype) const override {
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

template <class T>
Tensor::Ptr matmul(const HostTensor<T> &a, const HostTensor<T> &b) {
    // two small matrices, the common case: no batch to pair, no threads to ask for
    if (a.shape.size() == 2 && b.shape.size() == 2 && a.shape[1] == b.shape[0] &&
        a.shape[0] * a.shape[1] * b.shape[1] < kSerialWork) {
        auto r = a.fresh({a.shape[0], b.shape[1]});
        aux_product(a.p, b.p, r->p, a.shape[0], a.shape[1], b.shape[1]);
        return r;
    }
    const BatchPairs bp = aux_pairs(a.shape, b.shape, "matmul");
    auto r = a.fresh(aux_matrix_shape(bp.batch, aux_rows(a.shape), b.shape.back()));
    aux_products(a.p, a.shape, b.p, b.shape, r->p, bp);
    return r;
}

// All rows of all matrices at once: one Eigen row sum.
template <class T>
Tensor::Ptr sum_last(const HostTensor<T> &a, bool keepdims) {
    const int64_t len = a.shape.back();
    Shape out = a.shape;
    if (keepdims)
        out.back() = 1;
    else
        out = Shape(std::vector<int64_t>(a.shape.begin(), a.shape.end() - 1));
    auto r = a.fresh(out);
    const int64_t rows = out.numel(); // an empty axis has no entries, so count the outputs
    if (len == 0)
        std::fill(r->p, r->p + rows, T(0));
    else
        Map<T>(r->p, rows, 1) = MapC<T>(a.p, rows, len).rowwise().sum();
    return r;
}

template <class T>
Tensor::Ptr expm(const HostTensor<T> &a) {
    const int64_t n = a.shape.back(), count = a.shape.numel() / std::max<int64_t>(n * n, 1);
    auto r = a.fresh(a.shape);
    // in column-major storage, where Eigen's Pade solve is quicker than on row-major matrices
    for (int64_t z = 0; z < count; ++z) {
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m = MapC<T>(a.p + z * n * n, n, n);
        Map<T>(r->p + z * n * n, n, n) = m.exp();
    }
    return r;
}

// Fused Kernels -----------------------------------------------------------------------------------

// M c and M G, two products of one call.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr>
zonotope_linear_map(const HostTensor<T> &M, const HostTensor<T> &c, const HostTensor<T> &G) {
    return {matmul(M, c), matmul(M, G)};
}

// With M+ = max(M, 0) and M- = min(M, 0): I M = [inf M+ + sup M-, sup M+ + inf M-], and
// M I = [M+ inf + M- sup, M+ sup + M- inf]: each term of CORA's sum is the same rounded product.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr> interval_matmul(const HostTensor<T> &inf,
                                                    const HostTensor<T> &sup,
                                                    const HostTensor<T> &M, bool point_left) {
    const int64_t count = M.shape.numel();
    const BatchPairs bp = point_left ? aux_pairs(M.shape, inf.shape, "matmul_interval")
                                     : aux_pairs(inf.shape, M.shape, "interval_matmul");
    const Shape &left = point_left ? M.shape : inf.shape, &right = point_left ? inf.shape : M.shape;
    const Shape out = aux_matrix_shape(bp.batch, aux_rows(left), right.back());
    const int64_t n = out.numel();
    // M+, M- and one product in this thread's scratch memory
    T *mp = aux_scratch<T>(2 * count + n), *mn = mp + count, *tmp = mn + count;
    for (int64_t i = 0; i < count; ++i) {
        mp[i] = std::max(M.p[i], T(0));
        mn[i] = std::min(M.p[i], T(0));
    }
    auto lo = M.fresh(out), hi = M.fresh(out);
    // dst = a1 b1 + a2 b2
    const auto both = [&](const T *a1, const T *b1, const T *a2, const T *b2, T *dst) {
        aux_products(a1, left, b1, right, dst, bp);
        aux_products(a2, left, b2, right, tmp, bp);
        for (int64_t i = 0; i < n; ++i) dst[i] = dst[i] + tmp[i];
    };
    if (point_left) {
        both(mp, inf.p, mn, sup.p, lo->p);
        both(mp, sup.p, mn, inf.p, hi->p);
    } else {
        both(inf.p, mp, sup.p, mn, lo->p);
        both(sup.p, mp, inf.p, mn, hi->p);
    }
    return {lo, hi};
}

// T = (inf + sup) / 2 and S = (sup - inf) / 2 in scratch memory, then T c, T G and the box of
// S times the row sums of |[c G]|, written straight into the results.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr>
zonotope_interval_map(const HostTensor<T> &inf, const HostTensor<T> &sup, const HostTensor<T> &c,
                      const HostTensor<T> &G) {
    const BatchPairs bp = aux_pairs(inf.shape, G.shape, "zonotope_interval_map");
    const int64_t n = aux_rows(inf.shape), k = inf.shape.back(), m = G.shape.back();
    const int64_t count = bp.batch.numel();
    auto Tc = c.fresh(aux_matrix_shape(bp.batch, n, 1));
    auto TG = G.fresh(aux_matrix_shape(bp.batch, n, m + n));
    T *t = aux_scratch<T>(2 * n * k + k + n), *s = t + n * k, *reach = s + n * k, *box = reach + k;
    const long long work = (long long)count * n * k * (m + 2);
    std::optional<EigenThreads> narrow;
    if (work >= kSerialWork) narrow.emplace(work);
    for (int64_t z = 0; z < count; ++z) {
        // the center T and radius S of the interval matrix of this member
        const auto [l, r] = bp.at(z);
        const T *lo = inf.p + l * n * k, *up = sup.p + l * n * k;
        for (int64_t i = 0; i < n * k; ++i) {
            t[i] = (lo[i] + up[i]) * T(0.5);
            s[i] = (up[i] - lo[i]) * T(0.5);
        }
        const auto Gz = MapC<T>(G.p + r * k * m, k, m);
        const auto cz = MapC<T>(c.p + r * k, k, 1);
        // the box S |[c G]| 1, and T c, T G
        Map<T>(reach, k, 1) = cz.cwiseAbs() + Gz.cwiseAbs().rowwise().sum();
        Map<T>(box, n, 1).noalias() = MapC<T>(s, n, k) * MapC<T>(reach, k, 1);
        Map<T>(Tc->p + z * n, n, 1).noalias() = MapC<T>(t, n, k) * cz;
        Eigen::Map<RowMat<T>, 0, Eigen::OuterStride<>> out(TG->p + z * n * (m + n), n, m + n,
                                                           Eigen::OuterStride<>(m + n));
        out.leftCols(m).noalias() = MapC<T>(t, n, k) * Gz;
        out.rightCols(n).setZero();
        for (int64_t i = 0; i < n; ++i) out(i, m + i) = box[i];
    }
    return {Tc, TG};
}

// The halves of the sum and the difference, written into their blocks of the result.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr>
zonotope_linear_combination(const HostTensor<T> &c1, const HostTensor<T> &G1,
                            const HostTensor<T> &c2, const HostTensor<T> &G2) {
    const int64_t m = G1.shape.back(), rows = G1.shape.numel() / std::max<int64_t>(m, 1);
    Shape shape = G1.shape;
    shape.back() = 2 * m + 1;
    auto c = c1.fresh(c1.shape), G = G1.fresh(shape);
    const int64_t nc = c1.shape.numel();
    for (int64_t i = 0; i < nc; ++i) c->p[i] = (c1.p[i] + c2.p[i]) * T(0.5);
    for (int64_t i = 0; i < rows; ++i) {
        const T *a = G1.p + i * m, *b = G2.p + i * m;
        T *row = G->p + i * (2 * m + 1);
        for (int64_t j = 0; j < m; ++j) {
            row[j] = (a[j] + b[j]) * T(0.5);
            row[m + j] = (a[j] - b[j]) * T(0.5);
        }
        row[2 * m] = (c1.p[i] - c2.p[i]) * T(0.5);
    }
    return {c, G};
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
template <class T>
Tensor::Ptr abs_sum(const HostTensor<T> &a) {
    const int64_t len = a.shape.back();
    Shape out = a.shape;
    out.back() = 1;
    auto r = a.fresh(out);
    const int64_t rows = out.numel(); // an empty axis has no entries, so count the outputs
    if (len == 0)
        std::fill(r->p, r->p + rows, T(0));
    else
        Map<T>(r->p, rows, 1) = MapC<T>(a.p, rows, len).cwiseAbs().rowwise().sum();
    return r;
}

// The kernels of both dtypes.
#define CORA_EIGEN_KERNELS(T)                                                                      \
    template Tensor::Ptr matmul(const HostTensor<T> &, const HostTensor<T> &);                     \
    template Tensor::Ptr sum_last(const HostTensor<T> &, bool);                                    \
    template Tensor::Ptr expm(const HostTensor<T> &);                                              \
    template std::pair<Tensor::Ptr, Tensor::Ptr> zonotope_linear_map(                              \
        const HostTensor<T> &, const HostTensor<T> &, const HostTensor<T> &);                      \
    template std::pair<Tensor::Ptr, Tensor::Ptr> zonotope_interval_map(                            \
        const HostTensor<T> &, const HostTensor<T> &, const HostTensor<T> &,                       \
        const HostTensor<T> &);                                                                    \
    template std::pair<Tensor::Ptr, Tensor::Ptr> zonotope_linear_combination(                      \
        const HostTensor<T> &, const HostTensor<T> &, const HostTensor<T> &,                       \
        const HostTensor<T> &);                                                                    \
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
