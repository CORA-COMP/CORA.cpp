// eigen - the Eigen backend of Tensor: float64 or float32 tensors of any shape on the CPU
//
// The tensors are HostTensors (host_tensor.h) of the eigen id; the products, the sums along the
// last axis, the matrix exponential and the fused primitives are Eigen's kernels here, matrix by
// matrix over the batch. Nothing else in the library includes Eigen for its own use.
//
// CORA.cpp only: the backends of Tensor replace MATLAB's matrices.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       01-October-2026
// Last update:   18-November-2026
// Reviewed:      ---

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

// Helpers -----------------------------------------------------------------------------------------

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

/// The shape of a center (batch..., n) as the row matrices (batch..., 1, n) it is in memory.
Shape aux_row_shape(const Shape &c) {
    Shape s;
    for (int64_t i = 0; i + 1 < c.size(); ++i) s.push_back(c[i]);
    s.push_back(1);
    s.push_back(c.back());
    return s;
}

/// Below this many multiplications a product is a plain loop: Eigen's setup costs more.
constexpr int64_t kSmallProduct = 512;
/// At most this inner size a product is a plain loop too: Eigen's kernels run along the inner axis,
/// slow on the tall (h, n) generator rows of a low-dimensional set.
constexpr int64_t kShortInner = 8;
/// Below this much work (multiplications) Eigen keeps a product on one thread: 2 x its 50000 task.
constexpr long long kSerialWork = 100000;

/// C = A B of one pair, (n, k) times (k, m). A small product is a plain loop over the rows of B
/// weighted by a row of A: no setup cost.
template <class T>
void aux_product(const T *A, const T *B, T *C, int64_t n, int64_t k, int64_t m) {
    if (n * k * m > kSmallProduct && k > kShortInner) {
        Map<T>(C, n, m).noalias() = MapC<T>(A, n, k) * MapC<T>(B, k, m);
        return;
    }
    if (m == 1) {     // a matrix times a vector: one dot product per row, summed in a register
        if (k == 2) { // a zonogon's rows, unrolled for the compiler: 0 + x is x, the same sums
            for (int64_t i = 0; i < n; ++i) C[i] = A[2 * i] * B[0] + A[2 * i + 1] * B[1];
            return;
        }
        for (int64_t i = 0; i < n; ++i) {
            T sum = T(0);
            for (int64_t q = 0; q < k; ++q) sum += A[i * k + q] * B[q];
            C[i] = sum;
        }
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
    // work Eigen would not split anyway runs here, without asking for threads
    if (work < kSerialWork) {
        for (int64_t z = 0; z < bp.batch.numel(); ++z) product(z);
        return;
    }
    const EigenThreads narrow(work);
    host::parallel_for(bp.batch.numel(), work, product);
}

/// C = A B' for an inner size K fixed at compile time: each entry 0 + a_0 b_0 + ... + a_{K-1}
/// b_{K-1} in order, as the loop of any K.
template <class T, int K>
void aux_dots_bt(const T *__restrict A, const T *__restrict B, T *__restrict C, int64_t n,
                 int64_t m) {
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j) {
            T sum = T(0);
            for (int q = 0; q < K; ++q) sum += A[i * K + q] * B[j * K + q];
            C[i * m + j] = sum;
        }
}

/// C = A B' of one pair, (n, k) times (m, k)': a small product is the dot products of the rows.
template <class T>
void aux_product_bt(const T *A, const T *B, T *C, int64_t n, int64_t k, int64_t m) {
    if (n * k * m > kSmallProduct && k > kShortInner) {
        Map<T>(C, n, m).noalias() = MapC<T>(A, n, k) * MapC<T>(B, m, k).transpose();
        return;
    }
    // the inner size known to the compiler: the same sums, unrolled
    switch (k) {
    case 2: // a zonogon
        return aux_dots_bt<T, 2>(A, B, C, n, m);
    case 3:
        return aux_dots_bt<T, 3>(A, B, C, n, m);
    case 4:
        return aux_dots_bt<T, 4>(A, B, C, n, m);
    case 5:
        return aux_dots_bt<T, 5>(A, B, C, n, m);
    case 6:
        return aux_dots_bt<T, 6>(A, B, C, n, m);
    case 7:
        return aux_dots_bt<T, 7>(A, B, C, n, m);
    case 8:
        return aux_dots_bt<T, 8>(A, B, C, n, m);
    default:
        break;
    }
    // any other inner size: the same loop at run time
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j) {
            T sum = T(0);
            for (int64_t q = 0; q < k; ++q) sum += A[i * k + q] * B[j * k + q];
            C[i * m + j] = sum;
        }
}

/// out = a x b' for every pair of the batch, as aux_products.
template <class T>
void aux_products_bt(const T *a, const Shape &sa, const T *b, const Shape &sb, T *out,
                     const host::BatchPairs &bp) {
    const int64_t n = aux_rows(sa), k = sa.back(), m = aux_rows(sb);
    const long long work = (long long)bp.batch.numel() * n * k * m;
    const auto product = [&](int64_t z) {
        const auto [left, right] = bp.at(z);
        aux_product_bt(a + left * n * k, b + right * k * m, out + z * n * m, n, k, m);
    };
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

/// sum = |a_0| + (|a_1| + (... + (|a_{m-1}| + 0))) over the m rows (m, n) of a, CORALean's order;
/// plain loops on unaliased pointers, as the rows are short and Eigen's per-row setup is not.
template <class T>
void aux_abs_rows(const T *__restrict a, int64_t m, int64_t n, T *__restrict sum) {
    std::fill(sum, sum + n, T(0));
    for (int64_t i = m; i-- > 0;)
        for (int64_t j = 0; j < n; ++j) sum[j] = std::abs(a[i * n + j]) + sum[j];
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
            std::transform(data.begin(), data.end(), t->p,
                           [](double v) { return static_cast<float>(v); });
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

// Sums --------------------------------------------------------------------------------------------

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

// The rows of every matrix summed at once: one Eigen column sum per matrix.
template <class T>
Tensor::Ptr sum_rows(const HostTensor<T> &a, bool keepdims) {
    const int64_t rows = aux_rows(a.shape), cols = a.shape.back();
    Shape out; // the batch axes, then the kept axis
    for (int64_t i = 0; i + 2 < a.shape.size(); ++i) out.push_back(a.shape[i]);
    if (keepdims) out.push_back(1);
    out.push_back(cols);
    auto r = a.fresh(out);
    const int64_t count = out.numel() / std::max<int64_t>(cols, 1);
    // row after row into the sums: one order of the terms wherever a matrix starts in memory
    for (int64_t z = 0; z < count; ++z) {
        // each entry its own sum, so the vector adds give the same bits at any alignment
        auto sum = Map<T>(r->p + z * cols, 1, cols);
        const T *m = a.p + z * rows * cols;
        sum.setZero();
        for (int64_t i = 0; i < rows; ++i) sum += MapC<T>(m + i * cols, 1, cols);
    }
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

// c M' and G M', two products of one call, on the rows of M as they are stored.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr>
zonotope_linear_map(const HostTensor<T> &M, const HostTensor<T> &c, const HostTensor<T> &G) {
    // a center (..., n) is the rows (..., 1, n) in memory; its image keeps the rank of c
    const auto product = [&](const HostTensor<T> &a, bool center) -> Tensor::Ptr {
        const Shape as = center ? aux_row_shape(a.shape) : a.shape;
        const BatchPairs bp = host::batch_pairs(as, M.shape, "zonotope_linear_map");
        Shape out = bp.batch;
        if (!center) out.push_back(aux_rows(as));
        out.push_back(aux_rows(M.shape));
        auto r = a.fresh(out);
        aux_products_bt(a.p, as, M.p, M.shape, r->p, bp);
        return r;
    };
    return {product(c, true), product(G, false)};
}

// With M+ = max(M, 0), M- = min(M, 0): I M = [inf M+ + sup M-, sup M+ + inf M-]; M I likewise.
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

// T = (inf + sup) / 2, S = max(|inf - T|, |sup - T|) in scratch memory, then c T', G T', the box.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr>
zonotope_interval_map(const HostTensor<T> &inf, const HostTensor<T> &sup, const HostTensor<T> &c,
                      const HostTensor<T> &G) {
    const BatchPairs bp = host::batch_pairs(inf.shape, G.shape, "zonotope_interval_map");
    const int64_t n = aux_rows(inf.shape), k = inf.shape.back(), m = aux_rows(G.shape);
    const int64_t count = bp.batch.numel();
    Shape tc = bp.batch; // the image of the center (..., n)
    tc.push_back(n);
    auto Tc = c.fresh(tc);
    auto TG = G.fresh(aux_matrix_shape(bp.batch, m + n, n));
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
            s[i] = std::max(std::abs(lo[i] - t[i]), std::abs(up[i] - t[i]));
        }
        const auto Gz = MapC<T>(G.p + r * m * k, m, k);
        const auto cz = MapC<T>(c.p + r * k, 1, k);
        // the box S |[c; G]|' 1, and c T', G T'; |c| + (|g_0| + (... + 0)), CORALean's order
        aux_abs_rows(G.p + r * m * k, m, k, reach);
        for (int64_t j = 0; j < k; ++j) reach[j] = std::abs(c.p[r * k + j]) + reach[j];
        Map<T>(box, n, 1).noalias() = MapC<T>(s, n, k) * MapC<T>(reach, k, 1);
        Map<T>(Tc->p + z * n, 1, n).noalias() = cz * MapC<T>(t, n, k).transpose();
        Map<T> out(TG->p + z * (m + n) * n, m + n, n);
        out.topRows(m).noalias() = Gz * MapC<T>(t, n, k).transpose();
        out.bottomRows(n).setZero();
        for (int64_t i = 0; i < n; ++i) out(m + i, i) = box[i];
    }
    return {Tc, TG};
}

// The halves of the sum and the difference, written into their blocks of the result.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr>
zonotope_linear_combination(const HostTensor<T> &c1, const HostTensor<T> &G1,
                            const HostTensor<T> &c2, const HostTensor<T> &G2) {
    // per member: the rows (G1 + G2)/2, then (G1 - G2)/2, then the shift (c1 - c2)/2
    const int64_t n = G1.shape.back(), m = aux_rows(G1.shape);
    const int64_t count = c1.shape.numel() / std::max<int64_t>(n, 1);
    Shape shape = G1.shape;
    shape[shape.size() - 2] = 2 * m + 1;
    auto c = c1.fresh(c1.shape), G = G1.fresh(shape);
    const int64_t nc = c1.shape.numel();
    for (int64_t i = 0; i < nc; ++i) c->p[i] = (c1.p[i] + c2.p[i]) * T(0.5);
    for (int64_t z = 0; z < count; ++z) {
        const T *a = G1.p + z * m * n, *b = G2.p + z * m * n;
        T *mean = G->p + z * (2 * m + 1) * n, *spread = mean + m * n, *shift = spread + m * n;
        for (int64_t i = 0; i < m * n; ++i) {
            mean[i] = (a[i] + b[i]) * T(0.5);
            spread[i] = (a[i] - b[i]) * T(0.5);
        }
        for (int64_t j = 0; j < n; ++j) shift[j] = (c1.p[z * n + j] - c2.p[z * n + j]) * T(0.5);
    }
    return {c, G};
}

// Girard's score |g|_1 - |g|_inf of each row of g (m, n), its terms in order at any alignment.
template <class T>
void aux_girard_scores(const T *g, int64_t m, int64_t n, T *score) {
    constexpr int64_t kBlock = 8;
    int64_t i = 0;
    // kBlock rows side by side: independent sums, each in the order of its row
    for (; i + kBlock <= m; i += kBlock) {
        T sum[kBlock] = {}, most[kBlock] = {};
        for (int64_t j = 0; j < n; ++j)
            for (int64_t r = 0; r < kBlock; ++r) {
                const T a = std::abs(g[(i + r) * n + j]);
                sum[r] += a;
                most[r] = a > most[r] ? a : most[r];
            }
        for (int64_t r = 0; r < kBlock; ++r) score[i + r] = sum[r] - most[r];
    }
    // the rows left over, one at a time
    for (; i < m; ++i) {
        T sum = T(0), most = T(0);
        for (int64_t j = 0; j < n; ++j) {
            const T a = std::abs(g[i * n + j]);
            sum += a;
            most = a > most ? a : most;
        }
        score[i] = sum - most;
    }
}

// The scores of all rows at once, then a stable sort per matrix.
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr> girard_split(const HostTensor<T> &G, int64_t keep) {
    const int64_t m = aux_rows(G.shape), n = G.shape.back();
    const int64_t count = G.shape.numel() / std::max<int64_t>(n * m, 1);
    Shape batch;
    for (int64_t i = 0; i + 2 < G.shape.size(); ++i) batch.push_back(G.shape[i]);
    auto kept = G.fresh(aux_matrix_shape(batch, keep, 1)),
         rest = G.fresh(aux_matrix_shape(batch, m - keep, 1));
    std::vector<int64_t> order(m);
    T *score = aux_scratch<T>(m);
    for (int64_t z = 0; z < count; ++z) {
        aux_girard_scores(G.p + z * m * n, m, n, score);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int64_t a, int64_t b) { return score[a] > score[b]; });
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

// |a_0| + (|a_1| + (... + 0)) over the rows of each matrix, as lean adds them, as a column.
template <class T>
Tensor::Ptr abs_sum_rows(const HostTensor<T> &a) {
    const int64_t m = aux_rows(a.shape), n = a.shape.back();
    Shape out(a.shape);
    out[out.size() - 2] = n;
    out.back() = 1;
    auto r = a.fresh(out);
    const int64_t count = out.numel() / std::max<int64_t>(n, 1);
    for (int64_t z = 0; z < count; ++z) aux_abs_rows(a.p + z * m * n, m, n, r->p + z * n);
    return r;
}

// The kernels of both dtypes.
#define CORA_EIGEN_KERNELS(T)                                                                      \
    template Tensor::Ptr matmul(const HostTensor<T> &, const HostTensor<T> &);                     \
    template Tensor::Ptr sum_last(const HostTensor<T> &, bool);                                    \
    template Tensor::Ptr sum_rows(const HostTensor<T> &, bool);                                    \
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
// the sum over the rows, of both dtypes
template Tensor::Ptr abs_sum_rows(const HostTensor<double> &);
template Tensor::Ptr abs_sum_rows(const HostTensor<float> &);

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
                   const std::shared_ptr<const void> &owner, uint8_t backend) {
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
