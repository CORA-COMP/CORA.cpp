// tensor - CoraTensor, the one array type that sets, dynamics and networks are written against
//
// A Tensor wraps a matrix library (Eigen, libtorch, lean) behind one operation set, named after
// the Python Array API standard, so an algorithm is written once and runs on all of them.
//
// The data decides. Every tensor carries its backend, device and dtype, and an operation runs
// where its operands live; operands of different backends, devices or dtypes are an error that
// names both. Library code makes new tensors only like an operand (zeros_like, eye_like, like);
// the current backend (BackendScope, set_default_backend, CORACPP_BACKEND) applies only where
// numbers become a tensor with no tensor to follow (Tensor({{1, 2}}), Tensor::zeros(shape)).
//
// Syntax:   cora::BackendScope scope("torch");   one of "eigen", "torch", "torch:cuda", "lean"
//           cora::Tensor A({{0, 1}, {-1, 0}});   built on that backend
//
// Layout: row-major, the last two axes are the matrix (..., rows, cols); a vector is a column
// (n, 1). Leading axes are batches on every backend and broadcast as in numpy.
//
// A backend implements the core operations of Tensor::Impl (ops.def, docs/tensor.md); the derived
// ones have portable defaults (src/global/tensor/) that a backend may override. Storage is shared
// and immutable to users; an operation on an rvalue tensor that alone owns its memory may reuse it.
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>


/// Marks a raw-number entry point: a function that makes tensors from host numbers on the current
/// backend (Tensor::zeros, from_data, Tensor({...})) because no tensor is there to follow, as a
/// random set or an oracle's answer. The conventions test allows that only in functions so marked.
#define CORA_FROM_NUMBERS

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// Shape, DType, Device ----------------------------------------------------------------------------

/// The sizes of the axes of a tensor, held inline (at most kMaxRank axes).
class Shape {
  public:
    static constexpr int kMaxRank = 8;

    Shape() = default;
    Shape(std::initializer_list<int64_t> sizes) {
        for (const int64_t d : sizes) push_back(d);
    }
    Shape(const std::vector<int64_t> &sizes) {
        for (const int64_t d : sizes) push_back(d);
    }

    int64_t size() const { return rank_; }
    bool empty() const { return rank_ == 0; }
    int64_t operator[](int64_t i) const { return d_[i]; }
    int64_t &operator[](int64_t i) { return d_[i]; }
    int64_t back() const { return d_[rank_ - 1]; }
    int64_t &back() { return d_[rank_ - 1]; }
    const int64_t *begin() const { return d_; }
    const int64_t *end() const { return d_ + rank_; }
    void push_back(int64_t v) {
        if (rank_ == kMaxRank) aux_too_many_axes();
        // append the size of the next axis
        d_[rank_++] = v;
    }
    /// The product of the sizes: the number of entries.
    int64_t numel() const {
        int64_t n = 1;
        for (int64_t i = 0; i < rank_; ++i) n *= d_[i];
        return n;
    }

    operator std::vector<int64_t>() const { return {begin(), end()}; }
    friend bool operator==(const Shape &a, const Shape &b) {
        if (a.rank_ != b.rank_) return false;
        for (int64_t i = 0; i < a.rank_; ++i)
            if (a.d_[i] != b.d_[i]) return false;
        return true;
    }
    friend bool operator!=(const Shape &a, const Shape &b) { return !(a == b); }
    friend bool operator==(const Shape &a, const std::vector<int64_t> &b) { return a == Shape(b); }
    friend bool operator!=(const Shape &a, const std::vector<int64_t> &b) { return !(a == b); }

  private:
    [[noreturn]] static void aux_too_many_axes();
    int64_t d_[kMaxRank] = {};
    int64_t rank_ = 0;
};

/// The number type of the entries.
enum class DType : uint8_t { float64, float32 };

/// "float64" or "float32".
const char *dtype_name(DType dtype);

/// Where the entries live: the CPU, or CUDA device `index`.
struct Device {
    enum class Type : uint8_t { cpu, cuda };
    Type type = Type::cpu;
    int8_t index = 0;
    std::string name() const; ///< "cpu", "cuda:0", ...
    friend bool operator==(Device a, Device b) { return a.type == b.type && a.index == b.index; }
    friend bool operator!=(Device a, Device b) { return !(a == b); }
};

// Operation Table ---------------------------------------------------------------------------------

/// Whether gradients pass an operation: everywhere, piecewise (the branch taken, held fixed), or
/// not at all (on torch the result is detached, with a warning; an error under GradientsScope).
enum class Gradient : uint8_t { differentiable, piecewise, not_differentiable };

/// core: every backend implements it; derived: a portable default; fused: a derived hot spot.
enum class OpKind : uint8_t { core, derived, fused };

/// How Eigen's and torch's results agree with the lean backend's (see ops.def).
enum class Agreement : uint8_t { exact, ulp, sum, norm };

/// The operations, in the order of ops.def.
enum class TensorOp : uint16_t {
#define CORA_TENSOR_OP(name, kind, gradient, agreement, k, summary) name,
#include "cora/global/tensor/ops.def"
#undef CORA_TENSOR_OP
};

/// The number of operations.
inline constexpr std::size_t kOpCount = 0
// NOLINTNEXTLINE(bugprone-macro-parentheses): +1 is a term of the sum over ops.def, not an operand
#define CORA_TENSOR_OP(name, kind, gradient, agreement, k, summary) +1
#include "cora/global/tensor/ops.def"
#undef CORA_TENSOR_OP
    ;

/// One row of ops.def.
struct OpInfo {
    const char *name;
    OpKind kind;
    Gradient gradient;
    Agreement agreement;
    int k; ///< the factor of the agreement bound
    const char *summary;
};

/// Every operation of Tensor, as ops.def lists them.
const std::vector<OpInfo> &tensor_ops();
const OpInfo &op_info(TensorOp op);

/// Makes a not_differentiable operation given tensors that need gradients an error rather than a
/// warning, on this thread until the scope ends (Python: `with cora.strict_gradients():`).
class GradientsScope {
  public:
    explicit GradientsScope(bool strict = true);
    ~GradientsScope();
    GradientsScope(const GradientsScope &) = delete;
    GradientsScope &operator=(const GradientsScope &) = delete;

  private:
    bool previous_;
};

// Tensor ------------------------------------------------------------------------------------------

class Tensor {
  public:
    /// The elementwise operations of two operands.
    enum class Binary : uint8_t {
        add,
        subtract,
        multiply,
        divide,
        pow,
        maximum,
        minimum,
        remainder,
        atan2,
        equal,
        not_equal,
        less,
        less_equal,
        greater,
        greater_equal
    };
    /// The elementwise functions.
    enum class Unary : uint8_t {
        negative,
        abs,
        sign,
        floor,
        ceil,
        round,
        sqrt,
        exp,
        log,
        // the trigonometric and hyperbolic functions
        sin,
        cos,
        tan,
        asin,
        acos,
        atan,
        sinh,
        cosh,
        tanh,
        asinh,
        acosh,
        atanh,
        isnan,
        isinf,
        isfinite
    };
    /// The reductions along an axis.
    enum class Reduce : uint8_t { sum, prod, max, min, argmax, argmin };
    /// The norms of matrix_norm and vector_norm.
    enum class Norm : uint8_t { one, two, inf, fro };

    struct Impl;
    using Ptr = std::shared_ptr<const Impl>;

    /// What a backend implements: the core operations (pure virtual) and, optionally, faster
    /// versions of the derived ones, which must agree with the defaults (see ops.def). Operands
    /// of a binary operation belong to one backend, device and dtype; Tensor checks that.
    struct Impl : std::enable_shared_from_this<Impl> {
        using Ptr = Tensor::Ptr;
        Impl(const Shape &shape, DType dtype, Device device, uint8_t backend)
            : shape(shape), dtype(dtype), device(device), backend(backend) {}
        virtual ~Impl() = default;

        Shape shape;
        DType dtype;
        Device device;
        uint8_t backend; ///< the backend's id (backend_id), for the operand checks

        Ptr self() const { return shared_from_this(); }

        // Core: construction and access
        virtual std::vector<double> data() const = 0;
        virtual Ptr to(Device device) const = 0;
        virtual Ptr to(DType dtype) const = 0;
        /// Host values (row-major) of `shape` on this backend, device and dtype.
        virtual Ptr like(const double *values, const Shape &shape) const = 0;
        virtual Ptr full(const Shape &shape, double value) const = 0;
        virtual Ptr arange(int64_t n) const = 0;
        virtual bool requires_grad() const { return false; }
        virtual Ptr detach() const { return self(); }

        // Core: elementwise, broadcast as in numpy
        virtual Ptr binary(Binary op, const Impl &o) const = 0;
        /// op(x, s), or op(s, x) when `scalar_first`.
        virtual Ptr scalar(Binary op, double s, bool scalar_first) const = 0;
        virtual Ptr unary(Unary op) const = 0;
        /// a where this (the condition) is nonzero, else b.
        virtual Ptr where(const Impl &a, const Impl &b) const = 0;
        /// The same in place, when this tensor's memory can be reused (it owns it alone, and the
        /// result has its shape and dtype); false otherwise. Called only for a unique owner.
        virtual bool binary_inplace(Binary, const Impl &) { return false; }
        virtual bool scalar_inplace(Binary, double, bool) { return false; }
        virtual bool unary_inplace(Unary) { return false; }

        // Core: reductions
        virtual Ptr reduce(Reduce op, int axis, bool keepdims) const = 0;
        virtual Ptr cumsum(int axis) const = 0;

        // Core: shape and selection
        virtual Ptr reshape(const Shape &shape) const = 0;
        virtual Ptr permute_dims(const std::vector<int> &axes) const = 0;
        virtual Ptr broadcast_to(const Shape &shape) const = 0;
        virtual Ptr slice(int axis, int64_t start, int64_t stop, int64_t step) const = 0;
        /// The entries at `idx` (a 1-D tensor of indices) along `axis`.
        virtual Ptr take(const Impl &idx, int axis) const = 0;
        virtual Ptr take_along_axis(const Impl &idx, int axis) const = 0;
        /// All the parts (this one first) joined along `axis`.
        virtual Ptr concat(std::span<const Impl *const> parts, int axis) const = 0;
        virtual Ptr argsort(int axis, bool descending) const = 0;

        // Core: linear algebra, batched
        virtual Ptr matmul(const Impl &o) const = 0;
        virtual Ptr solve(const Impl &b) const = 0;
        virtual Ptr cholesky() const = 0;
        virtual std::pair<Ptr, Ptr> qr() const = 0;
        virtual std::array<Ptr, 3> svd() const = 0;
        virtual std::pair<Ptr, Ptr> eigh() const = 0;

        // Derived: defaults made of core operations (src/global/tensor/derived.cpp)
        virtual Ptr expm() const;
        virtual Ptr sort(int axis, bool descending) const;
        virtual std::pair<Ptr, Ptr> topk(int64_t k, int axis, bool largest) const;
        virtual Ptr kron(const Impl &o) const;
        virtual Ptr matrix_norm(Norm ord) const;
        virtual Ptr vector_norm(Norm ord, int axis, bool keepdims) const;
        virtual Ptr diag() const;

        // Fused: one specification each, in the lean backend's order (src/global/tensor/fused.cpp)
        virtual std::pair<Ptr, Ptr> zonotope_linear_map(const Impl &c, const Impl &G) const;
        virtual std::pair<Ptr, Ptr> zonotope_interval_map(const Impl &sup, const Impl &c,
                                                          const Impl &G) const;
        virtual std::pair<Ptr, Ptr> zonotope_linear_combination(const Impl &G1, const Impl &c2,
                                                                const Impl &G2) const;
        virtual std::pair<Ptr, Ptr> interval_matmul(const Impl &sup, const Impl &M) const;
        virtual std::pair<Ptr, Ptr> matmul_interval(const Impl &inf, const Impl &sup) const;
        virtual std::pair<Ptr, Ptr> girard_split(int64_t keep) const;
        virtual Ptr abs_sum() const;
    };

    /// Makes the tensors of one backend from host numbers.
    struct Backend {
        virtual ~Backend() = default;
        virtual std::string name() const = 0;
        /// Row-major `data` of `shape` on `device` (empty: the backend's default device).
        virtual Ptr make(const std::vector<double> &data, const Shape &shape,
                         const std::string &device, DType dtype) const = 0;
    };

    /// One number of a column. It makes `{{1}, {2}}` read as two rows, not two columns of one.
    struct Entry {
        double value;
        Entry(double v) : value(v) {}
    };

    // Construction from numbers, on the current backend -------------------------------------------

    Tensor() = default;
    explicit Tensor(Ptr impl) : impl_(std::move(impl)) {}

    /// A column (n, 1); `device` is "cpu", "gpu", "cuda:1" (empty: the backend's default).
    Tensor(std::initializer_list<Entry> column, const std::string &device = "");

    /// A matrix from its rows.
    Tensor(std::initializer_list<std::initializer_list<double>> rows,
           const std::string &device = "");

    /// Row-major `data` of `shape`.
    static Tensor from_data(const std::vector<double> &data, const Shape &shape,
                            const std::string &device = "", DType dtype = DType::float64);
    static Tensor zeros(const Shape &shape, const std::string &device = "");
    static Tensor ones(const Shape &shape, const std::string &device = "");
    static Tensor eye(int64_t n, const std::string &device = "");

    // Construction like a tensor: its backend, device and dtype -----------------------------------

    /// Host values of `shape`.
    static Tensor like(const Tensor &like, const std::vector<double> &data, const Shape &shape);
    Tensor full_like(const Shape &shape, double value) const;
    Tensor zeros_like(const Shape &shape) const { return full_like(shape, 0.0); }
    Tensor ones_like(const Shape &shape) const { return full_like(shape, 1.0); }
    /// The identity (n, n).
    Tensor eye_like(int64_t n) const;
    /// 0, 1, ..., n-1 as a 1-D tensor.
    Tensor arange_like(int64_t n) const;
    /// Zeros, ones and the identity of this tensor's own shape ((..., n, n) for the identity).
    Tensor zeros_like() const { return full_like(shape(), 0.0); }
    Tensor ones_like() const { return full_like(shape(), 1.0); }
    Tensor eye_like() const;

    // Access --------------------------------------------------------------------------------------

    const Impl &impl() const { return *impl_; }
    const Ptr &ptr() const { return impl_; }
    bool defined() const { return impl_ != nullptr; }
    const Shape &shape() const { return impl_->shape; }
    int64_t ndim() const { return impl_->shape.size(); }
    /// The size of `axis` (negative counts from the end).
    int64_t size(int axis) const;
    int64_t numel() const { return impl_->shape.numel(); }
    DType dtype() const { return impl_->dtype; }
    std::string device() const { return impl_->device.name(); }
    Device device_id() const { return impl_->device; }
    /// The backend's name: "eigen", "lean" or "torch".
    const char *backend_name() const;
    std::vector<double> data() const { return impl_->data(); } ///< Host values, row-major.
    /// The entry of a tensor of one entry.
    double item() const;
    bool requires_grad() const { return impl_->requires_grad(); }
    Tensor detach() const { return Tensor(impl_->detach()); }

    /// The same values on `device` ("cpu", "gpu", "cuda:1"), or in another dtype.
    Tensor to(const std::string &device) const;
    Tensor to(DType dtype) const;

    // Elementwise ---------------------------------------------------------------------------------

    friend Tensor operator+(const Tensor &a, const Tensor &b) { return a.add(b); }
    friend Tensor operator+(Tensor &&a, const Tensor &b) { return std::move(a).add(b); }
    friend Tensor operator-(const Tensor &a, const Tensor &b) { return a.subtract(b); }
    friend Tensor operator-(Tensor &&a, const Tensor &b) { return std::move(a).subtract(b); }
    friend Tensor operator*(const Tensor &a, double s) { return a.scalar_op(Binary::multiply, s); }
    friend Tensor operator*(Tensor &&a, double s) {
        return std::move(a).scalar_op(Binary::multiply, s);
    }
    friend Tensor operator*(double s, const Tensor &a) { return a * s; }
    friend Tensor operator*(double s, Tensor &&a) { return std::move(a) * s; }
    // with a number, on either side
    friend Tensor operator+(const Tensor &a, double s) { return a.scalar_op(Binary::add, s); }
    friend Tensor operator+(double s, const Tensor &a) { return a.scalar_op(Binary::add, s); }
    friend Tensor operator-(const Tensor &a, double s) { return a.scalar_op(Binary::subtract, s); }
    friend Tensor operator-(double s, const Tensor &a) {
        return a.scalar_op(Binary::subtract, s, true);
    }
    friend Tensor operator/(const Tensor &a, double s) { return a.scalar_op(Binary::divide, s); }
    // an operand nobody else holds is written in place
    friend Tensor operator+(Tensor &&a, double s) { return std::move(a).scalar_op(Binary::add, s); }
    friend Tensor operator+(double s, Tensor &&a) { return std::move(a) + s; }
    friend Tensor operator-(Tensor &&a, double s) {
        return std::move(a).scalar_op(Binary::subtract, s);
    }
    friend Tensor operator-(double s, Tensor &&a) {
        return std::move(a).scalar_op(Binary::subtract, s, true);
    }
    friend Tensor operator/(Tensor &&a, double s) {
        return std::move(a).scalar_op(Binary::divide, s);
    }
    friend Tensor operator/(double s, const Tensor &a) {
        return a.scalar_op(Binary::divide, s, true);
    }
    friend Tensor operator-(const Tensor &a) { return a.negative(); }

    // the named forms; an rvalue tensor that alone owns its memory is reused
    Tensor add(const Tensor &o) const & { return binary(Binary::add, o); }
    Tensor add(const Tensor &o) && { return std::move(*this).binary(Binary::add, o); }
    Tensor subtract(const Tensor &o) const & { return binary(Binary::subtract, o); }
    Tensor subtract(const Tensor &o) && { return std::move(*this).binary(Binary::subtract, o); }
    Tensor multiply(const Tensor &o) const & { return binary(Binary::multiply, o); }
    Tensor multiply(const Tensor &o) && { return std::move(*this).binary(Binary::multiply, o); }
    Tensor divide(const Tensor &o) const & { return binary(Binary::divide, o); }
    Tensor divide(const Tensor &o) && { return std::move(*this).binary(Binary::divide, o); }
    Tensor pow(const Tensor &o) const { return binary(Binary::pow, o); }
    Tensor pow(double p) const { return scalar_op(Binary::pow, p); }
    Tensor maximum(const Tensor &o) const { return binary(Binary::maximum, o); }
    Tensor minimum(const Tensor &o) const { return binary(Binary::minimum, o); }
    Tensor maximum(double s) const & { return scalar_op(Binary::maximum, s); }
    Tensor maximum(double s) && { return std::move(*this).scalar_op(Binary::maximum, s); }
    Tensor minimum(double s) const & { return scalar_op(Binary::minimum, s); }
    Tensor minimum(double s) && { return std::move(*this).scalar_op(Binary::minimum, s); }
    Tensor remainder(const Tensor &o) const { return binary(Binary::remainder, o); }
    Tensor atan2(const Tensor &o) const { return binary(Binary::atan2, o); }
    Tensor equal(const Tensor &o) const { return binary(Binary::equal, o); }
    Tensor not_equal(const Tensor &o) const { return binary(Binary::not_equal, o); }
    // comparisons: 1 where they hold, else 0
    Tensor less(const Tensor &o) const { return binary(Binary::less, o); }
    Tensor less_equal(const Tensor &o) const { return binary(Binary::less_equal, o); }
    Tensor greater(const Tensor &o) const { return binary(Binary::greater, o); }
    Tensor greater_equal(const Tensor &o) const { return binary(Binary::greater_equal, o); }
    /// op(x, s) with a number s; scalar_first gives op(s, x).
    Tensor scalar_op(Binary op, double s, bool scalar_first = false) const &;
    Tensor scalar_op(Binary op, double s, bool scalar_first = false) &&;

    // the functions of one operand
    Tensor negative() const { return unary(Unary::negative); }
    Tensor abs() const & { return unary(Unary::abs); }
    Tensor abs() && { return std::move(*this).unary(Unary::abs); }
    Tensor sign() const { return unary(Unary::sign); }
    Tensor floor() const { return unary(Unary::floor); }
    Tensor ceil() const { return unary(Unary::ceil); }
    Tensor round() const { return unary(Unary::round); }
    Tensor sqrt() const { return unary(Unary::sqrt); }
    Tensor exp() const { return unary(Unary::exp); }
    Tensor log() const { return unary(Unary::log); }
    Tensor sin() const { return unary(Unary::sin); }
    Tensor cos() const { return unary(Unary::cos); }
    Tensor tan() const { return unary(Unary::tan); }
    Tensor asin() const { return unary(Unary::asin); }
    Tensor acos() const { return unary(Unary::acos); }
    Tensor atan() const { return unary(Unary::atan); }
    Tensor sinh() const { return unary(Unary::sinh); }
    Tensor cosh() const { return unary(Unary::cosh); }
    Tensor tanh() const { return unary(Unary::tanh); }
    Tensor asinh() const { return unary(Unary::asinh); }
    // the inverse hyperbolic functions
    Tensor acosh() const { return unary(Unary::acosh); }
    Tensor atanh() const { return unary(Unary::atanh); }
    // masks: 1 where it holds, else 0
    Tensor isnan() const { return unary(Unary::isnan); }
    Tensor isinf() const { return unary(Unary::isinf); }
    Tensor isfinite() const { return unary(Unary::isfinite); }
    Tensor square() const;
    Tensor sigmoid() const;
    Tensor clip(double low, double high) const;

    /// a where cond is nonzero, else b; the three broadcast.
    static Tensor where(const Tensor &cond, const Tensor &a, const Tensor &b);

    /// The same functions of a number or of a whole tensor, without `<cmath>`:
    /// `Tensor::cos(0.2)` is a double, so `Tensor A({{Tensor::cos(phi), -Tensor::sin(phi)}, ...})`
    /// builds a rotation matrix; `Tensor::cos(t)` is `t.cos()`.
    static double sin(double x) { return std::sin(x); }
    static double cos(double x) { return std::cos(x); }
    static double tan(double x) { return std::tan(x); }
    static double exp(double x) { return std::exp(x); }
    static double log(double x) { return std::log(x); }
    static double sqrt(double x) { return std::sqrt(x); }
    static Tensor sin(const Tensor &t) { return t.sin(); }
    static Tensor cos(const Tensor &t) { return t.cos(); }
    static Tensor tan(const Tensor &t) { return t.tan(); }
    static Tensor exp(const Tensor &t) { return t.exp(); }
    static Tensor log(const Tensor &t) { return t.log(); }
    static Tensor sqrt(const Tensor &t) { return t.sqrt(); }

    // Reductions along `axis` (negative counts from the end) --------------------------------------

    Tensor sum(int axis, bool keepdims = false) const {
        return reduce(Reduce::sum, axis, keepdims);
    }
    Tensor prod(int axis, bool keepdims = false) const {
        return reduce(Reduce::prod, axis, keepdims);
    }
    Tensor max(int axis, bool keepdims = false) const {
        return reduce(Reduce::max, axis, keepdims);
    }
    Tensor min(int axis, bool keepdims = false) const {
        return reduce(Reduce::min, axis, keepdims);
    }
    Tensor argmax(int axis, bool keepdims = false) const;
    Tensor argmin(int axis, bool keepdims = false) const;
    Tensor cumsum(int axis) const;
    Tensor mean(int axis, bool keepdims = false) const;
    Tensor any(int axis, bool keepdims = false) const;
    Tensor all(int axis, bool keepdims = false) const;
    Tensor count_nonzero(int axis, bool keepdims = false) const;
    Tensor softmax(int axis) const;
    /// Over all entries, as a (1, 1) tensor.
    Tensor sum() const;
    Tensor max() const;
    Tensor min() const;

    // Shape and selection -------------------------------------------------------------------------

    Tensor reshape(const Shape &shape) const;
    Tensor permute_dims(const std::vector<int> &axes) const;
    Tensor broadcast_to(const Shape &shape) const;
    /// x[..., start:stop:step, ...] along `axis`; stop is clamped to the size.
    Tensor slice(int axis, int64_t start, int64_t stop, int64_t step = 1) const;
    /// The entries at `idx` along `axis`: a 1-D tensor of indices like this one, or host indices.
    Tensor take(const Tensor &idx, int axis) const;
    Tensor take(const std::vector<int64_t> &idx, int axis) const;
    Tensor take(std::initializer_list<int64_t> idx, int axis) const {
        return take(std::vector<int64_t>(idx), axis);
    }
    Tensor take_along_axis(const Tensor &idx, int axis) const;
    static Tensor concat(const std::vector<Tensor> &parts, int axis);
    static Tensor concat(std::initializer_list<Tensor> parts, int axis);
    /// Tensors of one shape along a new axis (0: a new leading batch axis).
    static Tensor stack(const std::vector<Tensor> &parts, int axis = 0);
    Tensor matrix_transpose() const;
    Tensor expand_dims(int axis) const;
    Tensor squeeze(int axis) const;
    Tensor flip(int axis) const;
    Tensor tile(const std::vector<int64_t> &reps) const;
    /// A column (..., n, 1) as the diagonal matrix (..., n, n).
    Tensor diag() const;
    /// The diagonal of (..., n, m) as a column (..., min(n, m), 1).
    Tensor diagonal() const;
    Tensor tril(int64_t k = 0) const;
    Tensor triu(int64_t k = 0) const;
    static Tensor block_diag(const std::vector<Tensor> &blocks);

    // Sorting -------------------------------------------------------------------------------------

    Tensor argsort(int axis = -1, bool descending = false) const;
    Tensor sort(int axis = -1, bool descending = false) const;
    /// The k largest (or smallest) along `axis` in order, and their indices; ties by index.
    std::pair<Tensor, Tensor> topk(int64_t k, int axis = -1, bool largest = true) const;
    /// The flat (row-major) indices of the nonzero entries, on the host.
    std::vector<int64_t> nonzero() const;

    // Linear algebra, batched over the leading axes -----------------------------------------------

    Tensor matmul(const Tensor &o) const;
    /// x with A x = B.
    Tensor solve(const Tensor &B) const;
    Tensor inv() const;
    /// The least-squares x of A x = B, for A of full column rank.
    Tensor lstsq(const Tensor &B) const;
    Tensor pinv() const;
    /// The number of singular values above tol (negative: max(n, m) eps times the largest).
    int64_t matrix_rank(double tol = -1) const;
    Tensor cholesky() const;
    /// Q (..., n, k) and R (..., k, m), k = min(n, m), with diag(R) >= 0.
    std::pair<Tensor, Tensor> qr() const;
    /// U, S (a column, descending) and V with A = U diag(S) V^T; each column of U has its
    /// entry of largest magnitude positive.
    std::array<Tensor, 3> svd() const;
    /// The eigenvalues (an ascending column) and eigenvectors of a symmetric matrix; each vector
    /// has its entry of largest magnitude positive.
    std::pair<Tensor, Tensor> eigh() const;
    Tensor trace() const;
    Tensor kron(const Tensor &o) const;
    Tensor expm() const;
    /// The 1, 2, inf or Frobenius norm of (..., n, m), as (..., 1, 1).
    Tensor matrix_norm(Norm ord = Norm::fro) const;
    /// The 1, 2 or inf norm along `axis`.
    Tensor vector_norm(Norm ord, int axis, bool keepdims = false) const;

    // Fused primitives ----------------------------------------------------------------------------

    /// M c and M G, the linear map of the zonotope (c, G).
    static std::pair<Tensor, Tensor> zonotope_linear_map(const Tensor &M, const Tensor &c,
                                                         const Tensor &G);
    /// The map of the zonotope (c, G) by the interval matrix [inf, sup] = T +- S, CORA's mtimes:
    /// T c and [T G, diag(S sum|[c G]|)], the sum along the rows.
    static std::pair<Tensor, Tensor> zonotope_interval_map(const Tensor &inf, const Tensor &sup,
                                                           const Tensor &c, const Tensor &G);
    /// CORA's linComb of (c1, G1) and (c2, G2) of one shape: (c1 + c2) / 2 and
    /// [(G1 + G2) / 2, (G1 - G2) / 2, (c1 - c2) / 2].
    static std::pair<Tensor, Tensor> zonotope_linear_combination(const Tensor &c1, const Tensor &G1,
                                                                 const Tensor &c2,
                                                                 const Tensor &G2);
    /// The bounds of I M, I = [inf, sup] an interval matrix: CORA's mtimes, each entry the sum
    /// over k of the smaller and the larger of inf_ik m_kj and sup_ik m_kj.
    static std::pair<Tensor, Tensor> interval_matmul(const Tensor &inf, const Tensor &sup,
                                                     const Tensor &M);
    /// The bounds of M I: interval_matmul with the point matrix on the left.
    static std::pair<Tensor, Tensor> matmul_interval(const Tensor &M, const Tensor &inf,
                                                     const Tensor &sup);
    /// The columns of G (..., n, m) by Girard's score sum|g| - max|g|, largest first and stable:
    /// the indices (..., 1, keep) of the first `keep` and (..., 1, m - keep) of the rest.
    std::pair<Tensor, Tensor> girard_split(int64_t keep) const;
    /// sum(|x|) along the last axis, kept: (..., n, m) -> (..., n, 1).
    Tensor abs_sum() const;

  private:
    Tensor binary(Binary op, const Tensor &o) const &;
    Tensor binary(Binary op, const Tensor &o) &&;
    Tensor unary(Unary op) const &;
    Tensor unary(Unary op) &&;
    Tensor reduce(Reduce op, int axis, bool keepdims) const;
    /// The Impl to change in place, if this tensor is its only owner.
    Impl *unique();

    Ptr impl_;
};

/// The backend ids of Impl::backend.
namespace backend_id {
inline constexpr uint8_t eigen = 0, lean = 1, torch = 2;
}

// Backend Selection -------------------------------------------------------------------------------

/// The backend of `spec`: "eigen", "lean", "torch" (CPU), "torch:cuda" or "torch:cuda:1", the
/// torch ones with an optional ",custom_backward" (the hand-written backward pass of the matrix
/// exponential). The same spec gives the same backend object.
const Tensor::Backend &backend(const std::string &spec);

/// This thread's backend for new tensors: its innermost BackendScope, else the process default.
const Tensor::Backend &backend();

/// The process default for threads without a BackendScope: initially $CORACPP_BACKEND, else
/// Eigen in every build (libtorch only when chosen).
void set_default_backend(const std::string &spec);

/// Makes new tensors on the backend of `spec` on this thread until the scope ends; existing
/// tensors keep their own.
class BackendScope {
  public:
    explicit BackendScope(const std::string &spec);
    ~BackendScope();
    BackendScope(const BackendScope &) = delete;
    BackendScope &operator=(const BackendScope &) = delete;

  private:
    const Tensor::Backend *previous_;
};

std::ostream &operator<<(std::ostream &out, const Tensor &t);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
