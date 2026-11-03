// host_tensor - the storage and the reference operations of the CPU backends (eigen and lean)
//
// A HostTensor holds row-major entries of any shape: its own memory (inline when small), or
// memory it borrows (a numpy array, the tensor a reshape views) with an owner kept alive. The
// operations here are the reference: elementwise in IEEE arithmetic, sums nested to the right
// from zero as CORALean's sumUp (so the lean backend is this class), batches looped in threads.
// The eigen backend derives from it and replaces the products, sums and the matrix exponential by
// Eigen's. The file is compiled with -ffp-contract=off (CMakeLists.txt): no product joins a sum.
//
// Syntax:   auto t = HostTensor<double>::make(shape, backend_id::lean);
// See also: host_tensor.cpp, eigen.cpp, lean.cpp, cora/global/backend/host.h

#pragma once

#include "cora/global/backend/host.h"
#include "cora/global/error.h"
#include "cora/global/tensor/tensor.h"
#include "cora/global/threads.h"

#include <memory>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::host {

// Shapes ------------------------------------------------------------------------------------------

/// Row-major strides of a shape.
Shape strides_of(const Shape &shape);

/// The axis `axis` of a tensor of `rank` axes (negative counts from the end), checked.
int axis_of(int axis, int64_t rank, const char *op);

/// The shape that a and b broadcast to, as numpy; an error that names both if none.
Shape broadcast_shapes(const Shape &a, const Shape &b);

// Tensor ------------------------------------------------------------------------------------------

template <class T>
struct HostTensor : Tensor::Impl {
    using Ptr = Tensor::Ptr;
    static constexpr int64_t kInline = 8; ///< up to this many entries live in the object

    /// A tensor of `shape` with uninitialized memory of its own.
    static std::shared_ptr<HostTensor> make(const Shape &shape, uint8_t backend);
    /// A tensor over `data` of `shape`, kept alive by `owner`; no copy.
    static std::shared_ptr<HostTensor> borrow(const T *data, const Shape &shape, uint8_t backend,
                                              std::shared_ptr<const void> owner);

    HostTensor(const Shape &shape, uint8_t backend);
    ~HostTensor() override = default;

    T *p = nullptr;                       ///< the entries, row-major
    std::unique_ptr<T[]> own;             ///< the memory it owns, unless inline or borrowed
    std::shared_ptr<const void> borrowed; ///< what keeps borrowed memory alive
    T small[kInline];

    /// Whether the memory is its own, so that an operation on a unique owner may reuse it.
    bool owns() const { return !borrowed; }
    static const HostTensor &of(const Tensor::Impl &o) {
        return static_cast<const HostTensor &>(o);
    }
    /// A new tensor of this backend.
    std::shared_ptr<HostTensor> fresh(const Shape &shape) const { return make(shape, backend); }

    // Construction and access
    std::vector<double> data() const override;
    Ptr to(Device device) const override;
    Ptr to(DType dtype) const override;
    Ptr like(const double *values, const Shape &shape) const override;
    Ptr full(const Shape &shape, double value) const override;
    Ptr arange(int64_t n) const override;

    // Elementwise
    Ptr binary(Tensor::Binary op, const Tensor::Impl &o) const override;
    Ptr scalar(Tensor::Binary op, double s, bool scalar_first) const override;
    Ptr unary(Tensor::Unary op) const override;
    Ptr where(const Tensor::Impl &a, const Tensor::Impl &b) const override;
    bool binary_inplace(Tensor::Binary op, const Tensor::Impl &o) override;
    bool scalar_inplace(Tensor::Binary op, double s, bool scalar_first) override;
    bool unary_inplace(Tensor::Unary op) override;

    // Reductions
    Ptr reduce(Tensor::Reduce op, int axis, bool keepdims) const override;
    Ptr cumsum(int axis) const override;

    // Shape and selection
    Ptr reshape(const Shape &shape) const override;
    Ptr permute_dims(const std::vector<int> &axes) const override;
    Ptr broadcast_to(const Shape &shape) const override;
    Ptr slice(int axis, int64_t start, int64_t stop, int64_t step) const override;
    Ptr take(const Tensor::Impl &idx, int axis) const override;
    Ptr take_along_axis(const Tensor::Impl &idx, int axis) const override;
    Ptr concat(std::span<const Tensor::Impl *const> parts, int axis) const override;
    Ptr argsort(int axis, bool descending) const override;

    // Linear algebra: the product nested to the right; the factorizations by Eigen, per matrix
    Ptr matmul(const Tensor::Impl &o) const override;
    Ptr solve(const Tensor::Impl &b) const override;
    Ptr cholesky() const override;
    std::pair<Ptr, Ptr> qr() const override;
    std::array<Ptr, 3> svd() const override;
    std::pair<Ptr, Ptr> eigh() const override;
    Ptr diag() const override;

    // Derived and fused: Eigen's kernels on the eigen backend, the defaults on lean
    Ptr expm() const override;
    std::pair<Ptr, Ptr> zonotope_linear_map(const Tensor::Impl &c,
                                            const Tensor::Impl &G) const override;
    std::pair<Ptr, Ptr> zonotope_interval_map(const Tensor::Impl &sup, const Tensor::Impl &c,
                                              const Tensor::Impl &G) const override;
    std::pair<Ptr, Ptr> zonotope_linear_combination(const Tensor::Impl &G1, const Tensor::Impl &c2,
                                                    const Tensor::Impl &G2) const override;
    std::pair<Ptr, Ptr> interval_matmul(const Tensor::Impl &sup,
                                        const Tensor::Impl &M) const override;
    std::pair<Ptr, Ptr> matmul_interval(const Tensor::Impl &inf,
                                        const Tensor::Impl &sup) const override;
    std::pair<Ptr, Ptr> girard_split(int64_t keep) const override;
    Ptr abs_sum() const override;
};

/// The kernels of the eigen backend (eigen.cpp), for HostTensor of that backend.
namespace eigen {
template <class T>
Tensor::Ptr matmul(const HostTensor<T> &a, const HostTensor<T> &b);
template <class T>
Tensor::Ptr sum_last(const HostTensor<T> &a, bool keepdims);
template <class T>
Tensor::Ptr expm(const HostTensor<T> &a);
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr>
zonotope_linear_map(const HostTensor<T> &M, const HostTensor<T> &c, const HostTensor<T> &G);
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr>
zonotope_interval_map(const HostTensor<T> &inf, const HostTensor<T> &sup, const HostTensor<T> &c,
                      const HostTensor<T> &G);
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr>
zonotope_linear_combination(const HostTensor<T> &c1, const HostTensor<T> &G1,
                            const HostTensor<T> &c2, const HostTensor<T> &G2);
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr> interval_matmul(const HostTensor<T> &inf,
                                                    const HostTensor<T> &sup,
                                                    const HostTensor<T> &M, bool point_left);
template <class T>
std::pair<Tensor::Ptr, Tensor::Ptr> girard_split(const HostTensor<T> &G, int64_t keep);
template <class T>
Tensor::Ptr abs_sum(const HostTensor<T> &a);
} // namespace eigen

// Batches -----------------------------------------------------------------------------------------

/// The matrices of a batched product (..., n, k) x (..., k, m), the batch axes broadcast: the
/// batch shape of the result and, per result matrix, the index of its left and right matrix.
struct BatchPairs {
    Shape batch, left_strides, right_strides; ///< strides in matrices, zero where broadcast
    /// The indices of the left and the right matrix of result matrix z.
    std::pair<int64_t, int64_t> at(int64_t z) const {
        int64_t l = 0, r = 0;
        for (int64_t d = batch.size() - 1; d >= 0; --d) {
            const int64_t i = z % batch[d];
            z /= batch[d];
            l += i * left_strides[d];
            r += i * right_strides[d];
        }
        return {l, r};
    }
};
BatchPairs batch_pairs(const Shape &a, const Shape &b, const char *op);

/// f(i) for i in [0, count), in threads when `work` (all of it, in scalar operations) pays.
template <class F>
void parallel_for(int64_t count, long long work, F f) {
    const int threads = threads_for(count, work);
    if (threads <= 1) {
        for (int64_t i = 0; i < count; ++i) f(i);
        return;
    }
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t i = 0; i < count; ++i) f(i);
}

} // namespace cora::host

// ---------------------------------------  END OF CODE  ---------------------------------------- //
