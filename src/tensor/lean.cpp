// lean - the lean backend of Tensor: double matrices whose operations round and order as CORALean's
//
// The backend runs an algorithm of CORA.cpp with the arithmetic that the CORALean oracle uses at
// dtype "nearest": IEEE binary64, round-to-nearest, no fused multiply-add, and the sums of
// CORALean's `sumUp`, which nests to the right, `f 0 + (f 1 + (... + (f (n-1) + 0)))`. The result
// of an operation then equals the oracle's bit for bit. It is a reference, not a fast backend.
//
// The file is compiled with -ffp-contract=off (CMakeLists.txt) so that no product joins a sum.

#include "tensor/lean.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Tensor ------------------------------------------------------------------------------------------

/// A row-major matrix; every operation returns a new tensor.
struct LeanTensor : Tensor::Impl {
    int64_t rows, cols;
    std::vector<double> v;

    LeanTensor(int64_t rows, int64_t cols, std::vector<double> v)
        : rows(rows), cols(cols), v(std::move(v)) {}

    static Ptr wrap(int64_t rows, int64_t cols, std::vector<double> v) {
        return std::make_shared<LeanTensor>(rows, cols, std::move(v));
    }
    static const LeanTensor &of(const Impl &o) { return static_cast<const LeanTensor &>(o); }
    double at(int64_t i, int64_t j) const { return v[i * cols + j]; }

    std::vector<int64_t> shape() const override { return {rows, cols}; }
    std::vector<double> data() const override { return v; }
    std::string device() const override { return "cpu"; }
    Ptr to(const std::string &device) const override {
        requireCpu(device);
        return wrap(rows, cols, v);
    }
    Ptr like(const std::vector<double> &data, const std::vector<int64_t> &shape) const override {
        return wrap(shape[0], shape[1], data);
    }

    /// The lean backend runs on the CPU: any other device is an error.
    static void requireCpu(const std::string &device) {
        if (!device.empty() && device != "cpu")
            throw std::invalid_argument("CoraTensor: the lean backend runs on the CPU only");
    }

    /// `op(a, b)` elementwise, where a column on either side is combined with every column of the
    /// other, as the other backends broadcast it.
    template <class Op> Ptr combine(const LeanTensor &b, Op op) const {
        const int64_t n = std::max(cols, b.cols);
        if (rows != b.rows || (cols != b.cols && cols != 1 && b.cols != 1))
            throw std::invalid_argument("CoraTensor: shapes do not broadcast");
        std::vector<double> out(rows * n);
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < n; ++j)
                out[i * n + j] = op(at(i, cols == 1 ? 0 : j), b.at(i, b.cols == 1 ? 0 : j));
        return wrap(rows, n, std::move(out));
    }

    template <class Op> Ptr map(Op op) const {
        std::vector<double> out(v.size());
        std::transform(v.begin(), v.end(), out.begin(), op);
        return wrap(rows, cols, std::move(out));
    }

    // Elementwise: the machine's own IEEE operation, rounded to nearest.
    Ptr add(const Impl &o) const override {
        return combine(of(o), [](double a, double b) { return a + b; });
    }
    Ptr sub(const Impl &o) const override {
        return combine(of(o), [](double a, double b) { return a - b; });
    }
    Ptr mul(const Impl &o) const override {
        return combine(of(o), [](double a, double b) { return a * b; });
    }
    Ptr div(const Impl &o) const override {
        return combine(of(o), [](double a, double b) { return a / b; });
    }
    Ptr scale(double s) const override {
        return map([s](double a) { return a * s; });
    }

    // Each entry is CORALean's sum over the inner index: nested to the right, from zero.
    Ptr matmul(const Impl &o) const override {
        const LeanTensor &b = of(o);
        if (cols != b.rows) throw std::invalid_argument("CoraTensor: inner dimensions differ");
        std::vector<double> out(rows * b.cols);
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < b.cols; ++j) {
                double sum = 0.0;
                for (int64_t k = cols - 1; k >= 0; --k) sum = at(i, k) * b.at(k, j) + sum;
                out[i * b.cols + j] = sum;
            }
        return wrap(rows, b.cols, std::move(out));
    }

    Ptr transpose() const override {
        std::vector<double> out(v.size());
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < cols; ++j) out[j * rows + i] = at(i, j);
        return wrap(cols, rows, std::move(out));
    }
    // abs, max(x, 0) and min(x, 0) are exact.
    Ptr abs() const override {
        return map([](double a) { return std::fabs(a); });
    }
    Ptr pos() const override {
        return map([](double a) { return std::max(a, 0.0); });
    }
    Ptr neg() const override {
        return map([](double a) { return std::min(a, 0.0); });
    }

    Ptr expm() const override {
        throw std::runtime_error(
            "CoraTensor: the lean backend has no expm; CORALean's exponential is an interval "
            "Taylor series (1/i! as bounds), which no point function reproduces");
    }

    // sqrt is exact in IEEE; the others come from the C library, which CORALean does not have.
    Ptr unary(Tensor::Unary op) const override {
        switch (op) {
        case Tensor::Unary::Sin: return map([](double a) { return std::sin(a); });
        case Tensor::Unary::Cos: return map([](double a) { return std::cos(a); });
        case Tensor::Unary::Tan: return map([](double a) { return std::tan(a); });
        case Tensor::Unary::Exp: return map([](double a) { return std::exp(a); });
        case Tensor::Unary::Log: return map([](double a) { return std::log(a); });
        case Tensor::Unary::Sqrt: return map([](double a) { return std::sqrt(a); });
        }
        throw std::invalid_argument("CoraTensor: unknown elementwise function");
    }

    // The row sum is CORALean's `sumUp`: nested to the right, from zero.
    Ptr sumLast() const override {
        std::vector<double> out(rows);
        for (int64_t i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int64_t j = cols - 1; j >= 0; --j) sum = at(i, j) + sum;
            out[i] = sum;
        }
        return wrap(rows, 1, std::move(out));
    }
    // The largest entry of each row, taken in order.
    Ptr maxLast() const override {
        std::vector<double> out(rows);
        for (int64_t i = 0; i < rows; ++i) {
            double best = cols > 0 ? at(i, 0) : 0.0;
            for (int64_t j = 1; j < cols; ++j) best = std::max(best, at(i, j));
            out[i] = best;
        }
        return wrap(rows, 1, std::move(out));
    }

    // Rearrangements copy the entries, so no rounding is involved.
    Ptr selectCols(const std::vector<int64_t> &idx) const override {
        std::vector<double> out(rows * idx.size());
        for (int64_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < idx.size(); ++j) out[i * idx.size() + j] = at(i, idx[j]);
        return wrap(rows, static_cast<int64_t>(idx.size()), std::move(out));
    }
    Ptr diag() const override {
        std::vector<double> out(rows * rows, 0.0);
        for (int64_t i = 0; i < rows; ++i) out[i * rows + i] = at(i, 0);
        return wrap(rows, rows, std::move(out));
    }
    Ptr eyeLike() const override {
        std::vector<double> out(rows * cols, 0.0);
        for (int64_t i = 0; i < std::min(rows, cols); ++i) out[i * cols + i] = 1.0;
        return wrap(rows, cols, std::move(out));
    }
    Ptr zerosLike() const override { return wrap(rows, cols, std::vector<double>(v.size(), 0.0)); }

    // The tensors side by side: this one, then each of `rest`.
    Ptr catLast(const std::vector<const Impl *> &rest) const override {
        int64_t total = cols;
        for (const Impl *r : rest) total += of(*r).cols;
        std::vector<double> out(rows * total);
        for (int64_t i = 0; i < rows; ++i) {
            int64_t at_ = 0;
            for (int64_t j = 0; j < cols; ++j) out[i * total + at_++] = at(i, j);
            for (const Impl *r : rest)
                for (int64_t j = 0; j < of(*r).cols; ++j) out[i * total + at_++] = of(*r).at(i, j);
        }
        return wrap(rows, total, std::move(out));
    }

    // The tensors one below the other: this one, then each of `rest`.
    Ptr catRows(const std::vector<const Impl *> &rest) const override {
        std::vector<double> out = v;
        int64_t total = rows;
        for (const Impl *r : rest) {
            out.insert(out.end(), of(*r).v.begin(), of(*r).v.end());
            total += of(*r).rows;
        }
        return wrap(total, cols, std::move(out));
    }

    // Lean tensors are single matrices, like the eigen ones.
    Ptr stack(const std::vector<const Impl *> &rest) const override {
        if (!rest.empty())
            throw std::invalid_argument("CoraTensor::stack: the lean backend holds one set, not a "
                                        "batch; use the torch backend");
        return wrap(rows, cols, v);
    }
};

// Backend -----------------------------------------------------------------------------------------

/// Makes lean tensors; matrices only (no batch dimensions) and CPU only.
struct LeanBackend : Tensor::Backend {
    std::string name() const override { return "lean"; }
    Tensor::Impl::Ptr make(const std::vector<double> &data, const std::vector<int64_t> &shape,
                           const std::string &device) const override {
        LeanTensor::requireCpu(device);
        if (shape.size() != 2)
            throw std::invalid_argument("CoraTensor: the lean backend holds matrices only");
        return LeanTensor::wrap(shape[0], shape[1], data);
    }
};

} // namespace


// ===========================================  MAIN  =========================================== //

// Backend -----------------------------------------------------------------------------------------

std::shared_ptr<const Tensor::Backend> leanBackend() { return std::make_shared<LeanBackend>(); }

bool isLean(const Tensor &t) { return dynamic_cast<const LeanTensor *>(&t.impl()) != nullptr; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
