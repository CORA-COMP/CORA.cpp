// tensor - CoraTensor, the one array type that sets and dynamics are written against
//
// A Tensor wraps a matrix library (Eigen, libtorch, ...) behind the few operations the
// algorithms need, so an algorithm is written once and runs on any of them. The backend is
// chosen once for the program, or from outside with CORACPP_BACKEND:
//
//     cora::setBackend("torch");                 // "eigen", "torch", "torch:cuda"
//     cora::Tensor A({{0, 1}, {-1, 0}});         // built on that backend
//
// Layout: the last two dimensions are the matrix, (..., rows, cols); a vector is a column
// (n, 1). Leading dimensions are batch dimensions where the backend has them (libtorch does,
// Eigen holds one matrix): a batch of sets or systems is the same call as one.
//
// A backend implements Tensor::Impl (the operations) and Tensor::Backend (making tensors from
// numbers); see eigen.cpp and torch.cpp.

#pragma once

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Tensor {
  public:
    // Backend Interface ---------------------------------------------------------------------------

    /// The elementwise functions of a tensor.
    enum class Unary { Sin, Cos, Tan, Exp, Log, Sqrt };

    /// What a backend implements. Both operands of a binary operation belong to the backend;
    /// Tensor checks that before calling in.
    struct Impl {
        using Ptr = std::shared_ptr<const Impl>;
        virtual ~Impl() = default;

        virtual std::vector<int64_t> shape() const = 0;
        /// The values on the host, row-major.
        virtual std::vector<double> data() const = 0;
        /// Where the values live: "cpu", "cuda:0", ...
        virtual std::string device() const = 0;
        /// The same values on `device` ("cpu", "gpu", "cuda", "cuda:1").
        virtual Ptr to(const std::string &device) const = 0;
        /// A tensor of this backend and device from host values (row-major) of `shape`.
        virtual Ptr like(const std::vector<double> &data,
                         const std::vector<int64_t> &shape) const = 0;

        /// Sum and difference; a column on either side is broadcast over the columns.
        virtual Ptr add(const Impl &o) const = 0;
        virtual Ptr sub(const Impl &o) const = 0;
        virtual Ptr scale(double s) const = 0;

        /// Elementwise product and quotient, broadcast like add and sub.
        virtual Ptr mul(const Impl &o) const = 0;
        virtual Ptr div(const Impl &o) const = 0;

        /// Matrix product, batched over the leading dimensions.
        virtual Ptr matmul(const Impl &o) const = 0;

        /// Swaps the last two dimensions.
        virtual Ptr transpose() const = 0;
        virtual Ptr abs() const = 0;

        /// max(x, 0) and min(x, 0), elementwise.
        virtual Ptr pos() const = 0;
        virtual Ptr neg() const = 0;

        /// The matrix exponential of (..., n, n).
        virtual Ptr expm() const = 0;

        /// A function applied to every element.
        virtual Ptr unary(Unary op) const = 0;

        /// Sums, or takes the largest of, the last dimension, keeping it: (.., n, m) -> (.., n, 1).
        virtual Ptr sumLast() const = 0;
        virtual Ptr maxLast() const = 0;

        /// The columns `idx` of (.., n, m), in that order: (.., n, idx.size()).
        virtual Ptr selectCols(const std::vector<int64_t> &idx) const = 0;

        /// A column (.., n, 1) as a diagonal matrix (.., n, n).
        virtual Ptr diag() const = 0;

        /// The identity, and zeros, in the shape of this (.., n, n) tensor.
        virtual Ptr eyeLike() const = 0;
        virtual Ptr zerosLike() const = 0;

        /// This tensor followed by `rest`, joined along the last dimension.
        virtual Ptr catLast(const std::vector<const Impl *> &rest) const = 0;

        /// This tensor followed by `rest`, joined along the rows (the second to last dimension).
        virtual Ptr catRows(const std::vector<const Impl *> &rest) const = 0;

        /// This tensor and `rest` (all of one shape) as one tensor with a new leading dimension.
        virtual Ptr stack(const std::vector<const Impl *> &rest) const = 0;
    };

    /// Makes the tensors of one backend from host numbers.
    struct Backend {
        virtual ~Backend() = default;
        virtual std::string name() const = 0;
        /// Row-major `data` of `shape` on `device` (empty: the backend's default device).
        virtual Impl::Ptr make(const std::vector<double> &data, const std::vector<int64_t> &shape,
                               const std::string &device) const = 0;
    };

    /// One number of a column. It makes `{{1}, {2}}` read as two rows, not two columns of one.
    struct Entry {
        double value;
        Entry(double v) : value(v) {}
    };

    // Construction --------------------------------------------------------------------------------

    Tensor() = default;
    explicit Tensor(Impl::Ptr impl) : impl_(std::move(impl)) {}

    /// A column (n, 1) on the current backend; `device` is "cpu", "gpu", "cuda:1" (empty: default).
    Tensor(std::initializer_list<Entry> column, const std::string &device = "");

    /// A matrix from its rows, on the current backend.
    Tensor(std::initializer_list<std::initializer_list<double>> rows,
           const std::string &device = "");

    /// Row-major `data` of `shape` on the current backend.
    static Tensor fromData(const std::vector<double> &data, const std::vector<int64_t> &shape,
                           const std::string &device = "");

    /// Like fromData, on the backend and device of `like` (library code meeting a given tensor).
    static Tensor like(const Tensor &like, const std::vector<double> &data,
                       const std::vector<int64_t> &shape);

    static Tensor zeros(const std::vector<int64_t> &shape, const std::string &device = "");
    static Tensor ones(const std::vector<int64_t> &shape, const std::string &device = "");
    static Tensor eye(int64_t n, const std::string &device = "");

    /// Joins tensors along the last dimension.
    static Tensor catLast(const std::vector<Tensor> &parts);

    /// Joins tensors along the rows (the second to last dimension).
    static Tensor catRows(const std::vector<Tensor> &parts);

    /// Stacks tensors of one shape along a new leading (batch) dimension.
    static Tensor stack(const std::vector<Tensor> &parts);

    // Access --------------------------------------------------------------------------------------

    const Impl &impl() const { return *impl_; }
    bool defined() const { return impl_ != nullptr; }
    std::vector<int64_t> shape() const { return impl_->shape(); }
    std::vector<double> data() const { return impl_->data(); }  ///< Host values, row-major.
    std::string device() const { return impl_->device(); }

    /// The same values on `device` ("cpu", "gpu", "cuda:1").
    Tensor to(const std::string &device) const { return Tensor(impl_->to(device)); }

    // Operations ----------------------------------------------------------------------------------

    friend Tensor operator+(const Tensor &a, const Tensor &b) { return a.binary(b, &Impl::add); }
    friend Tensor operator-(const Tensor &a, const Tensor &b) { return a.binary(b, &Impl::sub); }
    friend Tensor operator*(const Tensor &a, double s) { return Tensor(a.impl_->scale(s)); }
    friend Tensor operator*(double s, const Tensor &a) { return a * s; }

    /// Elementwise product and quotient; shapes broadcast as in +.
    Tensor mul(const Tensor &o) const { return binary(o, &Impl::mul); }
    Tensor div(const Tensor &o) const { return binary(o, &Impl::div); }
    Tensor matmul(const Tensor &o) const { return binary(o, &Impl::matmul); }
    Tensor transpose() const { return Tensor(impl_->transpose()); }
    Tensor abs() const { return Tensor(impl_->abs()); }
    Tensor pos() const { return Tensor(impl_->pos()); }
    Tensor neg() const { return Tensor(impl_->neg()); }
    Tensor expm() const { return Tensor(impl_->expm()); }
    Tensor sin() const { return Tensor(impl_->unary(Unary::Sin)); }  ///< Elementwise.
    Tensor cos() const { return Tensor(impl_->unary(Unary::Cos)); }
    Tensor tan() const { return Tensor(impl_->unary(Unary::Tan)); }
    Tensor exp() const { return Tensor(impl_->unary(Unary::Exp)); }
    Tensor log() const { return Tensor(impl_->unary(Unary::Log)); }
    Tensor sqrt() const { return Tensor(impl_->unary(Unary::Sqrt)); }

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
    Tensor sumLast() const { return Tensor(impl_->sumLast()); }
    Tensor maxLast() const { return Tensor(impl_->maxLast()); }

    /// The columns `idx`, in that order.
    Tensor selectCols(const std::vector<int64_t> &idx) const {
        return Tensor(impl_->selectCols(idx));
    }

    /// The elementwise larger and smaller of a and b.
    static Tensor maximum(const Tensor &a, const Tensor &b) { return b + (a - b).pos(); }
    static Tensor minimum(const Tensor &a, const Tensor &b) { return a - (a - b).pos(); }
    Tensor diag() const { return Tensor(impl_->diag()); }
    Tensor eyeLike() const { return Tensor(impl_->eyeLike()); }
    Tensor zerosLike() const { return Tensor(impl_->zerosLike()); }

  private:
    Tensor binary(const Tensor &o, Impl::Ptr (Impl::*op)(const Impl &) const) const;

    Impl::Ptr impl_;
};

// Backend Selection -------------------------------------------------------------------------------

/// Chooses the backend that new tensors are made on: "eigen", "torch" (CPU), "torch:cuda" or
/// "torch:cuda:1"; append ",customBackward" for the hand-written backward pass of the matrix
/// exponential. Existing tensors keep their backend.
void setBackend(const std::string &spec);

/// The current backend: what setBackend chose, else $CORACPP_BACKEND, else libtorch if built,
/// else Eigen.
const Tensor::Backend &backend();

std::ostream &operator<<(std::ostream &out, const Tensor &t);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
