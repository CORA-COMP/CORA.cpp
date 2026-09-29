// CoraTensor: the one array type every set and every dynamics class is written against.
//
// A `Tensor` wraps whatever library holds the numbers — Eigen matrices, libtorch tensors,
// a future backend — behind the few operations the algorithms need. Which library is set
// once, for the whole program:
//
//     ct::setBackend("eigen");             // or "torch", "torch:cuda"
//     ct::Tensor A({{0, 1}, {-1, 0}});      // built on that backend
//
// or from outside, with `CORACPP_BACKEND=eigen`, so the same binary reruns on another
// backend. Everything built from a tensor stays on its backend, so an algorithm is written
// once and runs on any of them.
//
// Layout: the last two dimensions are the matrix, `(..., rows, cols)`, and a vector is a
// `(n, 1)` column. Leading dimensions are batch dimensions where the library has them
// (libtorch broadcasts them, Eigen has none), which is what makes a batch of sets or of
// systems the same call as one.
//
// A backend is a `Backend` that makes tensors from numbers and a subclass of `Tensor::Impl`
// that implements the operations below; see eigen.cpp and torch.cpp.

#pragma once

#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace cora::ct {

class Tensor {
  public:
    /// What a backend implements. Operands of a binary operation are of the same backend;
    /// `Tensor` checks that before calling in.
    struct Impl {
        using Ptr = std::shared_ptr<const Impl>;
        virtual ~Impl() = default;

        virtual std::vector<int64_t> shape() const = 0;
        /// The values on the host, row-major.
        virtual std::vector<double> data() const = 0;
        /// Where the values live: "cpu", or "cuda:0" and the like.
        virtual std::string device() const = 0;
        /// The same values on `device`: "cpu", "gpu" (or "cuda", "cuda:1").
        virtual Ptr to(const std::string &device) const = 0;
        /// A tensor of this backend and device with the given values (row-major, host).
        virtual Ptr like(const std::vector<double> &data, const std::vector<int64_t> &shape) const = 0;

        virtual Ptr add(const Impl &o) const = 0;
        virtual Ptr sub(const Impl &o) const = 0;
        virtual Ptr scale(double s) const = 0;
        virtual Ptr matmul(const Impl &o) const = 0;
        virtual Ptr transpose() const = 0;
        virtual Ptr abs() const = 0;
        /// `max(x, 0)` and `min(x, 0)` elementwise.
        virtual Ptr pos() const = 0;
        virtual Ptr neg() const = 0;
        virtual Ptr expm() const = 0;
        /// Sum over the last dimension, kept: `(..., n, m)` to `(..., n, 1)`.
        virtual Ptr sumLast() const = 0;
        /// `(..., n, 1)` to the diagonal matrix `(..., n, n)`.
        virtual Ptr diag() const = 0;
        /// The identity and zeros in the shape of a square `(..., n, n)` tensor.
        virtual Ptr eyeLike() const = 0;
        virtual Ptr zerosLike() const = 0;
        /// This tensor followed by `rest`, joined along the last dimension.
        virtual Ptr catLast(const std::vector<const Impl *> &rest) const = 0;
    };

    /// Makes tensors of one backend from numbers on the host.
    struct Backend {
        virtual ~Backend() = default;
        virtual std::string name() const = 0;
        /// `data` row-major with the given shape, on `device` (empty: the backend default).
        virtual Impl::Ptr make(const std::vector<double> &data, const std::vector<int64_t> &shape,
                               const std::string &device) const = 0;
    };

    Tensor() = default;
    explicit Tensor(Impl::Ptr impl) : impl_(std::move(impl)) {}

    /// One number of a column. A wrapper so that `{{1}, {2}}` reads as two rows rather than
    /// two one-element columns: a braced list converts to a list of doubles as readily as to
    /// a list of lists, and the matrix constructor must be the better match.
    struct Entry {
        double value;
        Entry(double v) : value(v) {}
    };

    /// A column vector `(n, 1)` on the current backend. `device` is "cpu", "gpu" (or "cuda",
    /// "cuda:1"); empty is the backend default. Every constructor below takes it the same
    /// way, so a tensor is allocated where it is meant to live.
    Tensor(std::initializer_list<Entry> column, const std::string &device = "");
    /// A matrix from its rows, on the current backend.
    Tensor(std::initializer_list<std::initializer_list<double>> rows,
           const std::string &device = "");

    /// `data` row-major with `shape`, on the current backend.
    static Tensor fromData(const std::vector<double> &data, const std::vector<int64_t> &shape,
                            const std::string &device = "");
    static Tensor zeros(const std::vector<int64_t> &shape, const std::string &device = "");
    /// `data` with `shape` on the backend and device of `like`, whatever backend is current:
    /// what library code uses to make a tensor that meets one it was given.
    static Tensor like(const Tensor &like, const std::vector<double> &data,
                       const std::vector<int64_t> &shape);
    static Tensor eye(int64_t n, const std::string &device = "");

    const Impl &impl() const { return *impl_; }
    bool defined() const { return impl_ != nullptr; }
    std::vector<int64_t> shape() const { return impl_->shape(); }
    std::vector<double> data() const { return impl_->data(); }
    std::string device() const { return impl_->device(); }
    /// The same values on `device`, as "gpu" or "cpu".
    Tensor to(const std::string &device) const { return Tensor(impl_->to(device)); }

    friend Tensor operator+(const Tensor &a, const Tensor &b) { return a.binary(b, &Impl::add); }
    friend Tensor operator-(const Tensor &a, const Tensor &b) { return a.binary(b, &Impl::sub); }
    friend Tensor operator*(const Tensor &a, double s) { return Tensor(a.impl_->scale(s)); }
    friend Tensor operator*(double s, const Tensor &a) { return a * s; }

    Tensor matmul(const Tensor &o) const { return binary(o, &Impl::matmul); }
    Tensor transpose() const { return Tensor(impl_->transpose()); }
    Tensor abs() const { return Tensor(impl_->abs()); }
    Tensor pos() const { return Tensor(impl_->pos()); }
    Tensor neg() const { return Tensor(impl_->neg()); }
    Tensor expm() const { return Tensor(impl_->expm()); }
    Tensor sumLast() const { return Tensor(impl_->sumLast()); }
    Tensor diag() const { return Tensor(impl_->diag()); }
    Tensor eyeLike() const { return Tensor(impl_->eyeLike()); }
    Tensor zerosLike() const { return Tensor(impl_->zerosLike()); }

    /// The tensors joined along the last dimension.
    static Tensor catLast(const std::vector<Tensor> &parts);

  private:
    Tensor binary(const Tensor &o, Impl::Ptr (Impl::*op)(const Impl &) const) const;

    Impl::Ptr impl_;
};

/// Chooses the backend that new tensors are made on: `"eigen"`, `"torch"` (CPU),
/// `"torch:cuda"` or `"torch:cuda:1"` for a device, optionally followed by
/// `",customBackward"` to differentiate the matrix exponential with the hand-written
/// backward pass. Tensors that already exist keep their backend.
void setBackend(const std::string &spec);

/// The backend new tensors are made on. Unless `setBackend` chose one, it is
/// `$CORACPP_BACKEND`, else libtorch when it is built in, else Eigen.
const Tensor::Backend &backend();

std::ostream &operator<<(std::ostream &out, const Tensor &t);

} // namespace cora::ct
