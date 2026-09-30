// eigen - the Eigen backend of Tensor: double matrices on the CPU, one matrix per tensor
//
// EigenTensor implements the operations of Tensor::Impl; EigenBackend makes tensors from host
// numbers. Nothing else in the library includes Eigen for its own use.

#include "tensor/eigen.h"

#include <stdexcept>

#include <unsupported/Eigen/MatrixFunctions>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Mat = Eigen::MatrixXd;
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Tensor ------------------------------------------------------------------------------------------

/// A matrix held as an Eigen::MatrixXd; every operation returns a new tensor.
struct EigenTensor : Tensor::Impl {
    Mat m;
    explicit EigenTensor(Mat m) : m(std::move(m)) {}

    static Ptr wrap(Mat m) { return std::make_shared<EigenTensor>(std::move(m)); }
    static const Mat &of(const Impl &o) { return static_cast<const EigenTensor &>(o).m; }

    std::vector<int64_t> shape() const override { return {m.rows(), m.cols()}; }
    std::vector<double> data() const override {
        const RowMajor rows = m;
        return std::vector<double>(rows.data(), rows.data() + rows.size());
    }
    std::string device() const override { return "cpu"; }
    Ptr to(const std::string &device) const override {
        requireCpu(device);
        return std::make_shared<EigenTensor>(m);
    }

    Ptr like(const std::vector<double> &data, const std::vector<int64_t> &shape) const override {
        return wrap(Eigen::Map<const RowMajor>(data.data(), shape[0], shape[1]));
    }

    /// `op(a, b)` elementwise, where a column on either side is combined with every column of the
    /// other, as libtorch broadcasts it.
    template <class Op> static Mat combine(const Mat &a, const Mat &b, Op op) {
        if (a.cols() == b.cols()) return a.binaryExpr(b, op);
        if (b.cols() == 1) return a.binaryExpr(b.col(0).replicate(1, a.cols()), op);
        if (a.cols() == 1) return a.col(0).replicate(1, b.cols()).binaryExpr(b, op);
        throw std::invalid_argument("CoraTensor: shapes do not broadcast");
    }

    /// Eigen runs on the CPU: any other device is an error.
    static void requireCpu(const std::string &device) {
        if (!device.empty() && device != "cpu")
            throw std::invalid_argument("CoraTensor: the eigen backend runs on the CPU only");
    }

    Ptr add(const Impl &o) const override {
        return wrap(combine(m, of(o), [](double a, double b) { return a + b; }));
    }
    Ptr sub(const Impl &o) const override {
        return wrap(combine(m, of(o), [](double a, double b) { return a - b; }));
    }
    Ptr scale(double s) const override { return wrap(m * s); }
    Ptr mul(const Impl &o) const override {
        return wrap(combine(m, of(o), [](double a, double b) { return a * b; }));
    }
    Ptr div(const Impl &o) const override {
        return wrap(combine(m, of(o), [](double a, double b) { return a / b; }));
    }
    Ptr matmul(const Impl &o) const override { return wrap(m * of(o)); }
    Ptr transpose() const override { return wrap(m.transpose()); }
    Ptr abs() const override { return wrap(m.cwiseAbs()); }
    Ptr pos() const override { return wrap(m.cwiseMax(0.0)); }
    Ptr neg() const override { return wrap(m.cwiseMin(0.0)); }
    Ptr expm() const override { return wrap(m.exp()); }
    // One case per function of Tensor::Unary.
    Ptr unary(Tensor::Unary op) const override {
        switch (op) {
        case Tensor::Unary::Sin: return wrap(m.array().sin().matrix());
        case Tensor::Unary::Cos: return wrap(m.array().cos().matrix());
        case Tensor::Unary::Tan: return wrap(m.array().tan().matrix());
        case Tensor::Unary::Exp: return wrap(m.array().exp().matrix());
        case Tensor::Unary::Log: return wrap(m.array().log().matrix());
        case Tensor::Unary::Sqrt: return wrap(m.array().sqrt().matrix());
        }
        throw std::invalid_argument("CoraTensor: unknown elementwise function");
    }
    Ptr sumLast() const override { return wrap(m.rowwise().sum()); }
    Ptr maxLast() const override { return wrap(m.rowwise().maxCoeff()); }
    Ptr selectCols(const std::vector<int64_t> &idx) const override {
        Mat out(m.rows(), idx.size());
        for (std::size_t j = 0; j < idx.size(); ++j) out.col(j) = m.col(idx[j]);
        return wrap(std::move(out));
    }
    Ptr diag() const override { return wrap(Mat(m.col(0).asDiagonal())); }
    Ptr eyeLike() const override { return wrap(Mat::Identity(m.rows(), m.cols())); }
    Ptr zerosLike() const override { return wrap(Mat::Zero(m.rows(), m.cols())); }

    // The tensors side by side: this one, then each of `rest`.
    Ptr catLast(const std::vector<const Impl *> &rest) const override {
        Eigen::Index cols = m.cols();
        for (const Impl *r : rest) cols += of(*r).cols();
        Mat out(m.rows(), cols);
        out.leftCols(m.cols()) = m;
        Eigen::Index at = m.cols();
        for (const Impl *r : rest) {
            out.middleCols(at, of(*r).cols()) = of(*r);
            at += of(*r).cols();
        }
        return wrap(std::move(out));
    }

    // The tensors one below the other: this one, then each of `rest`.
    Ptr catRows(const std::vector<const Impl *> &rest) const override {
        Eigen::Index rows = m.rows();
        for (const Impl *r : rest) rows += of(*r).rows();
        Mat out(rows, m.cols());
        out.topRows(m.rows()) = m;
        Eigen::Index at = m.rows();
        for (const Impl *r : rest) {
            out.middleRows(at, of(*r).rows()) = of(*r);
            at += of(*r).rows();
        }
        return wrap(std::move(out));
    }

    // Eigen tensors are single matrices: a stack of one is the tensor itself.
    Ptr stack(const std::vector<const Impl *> &rest) const override {
        if (!rest.empty())
            throw std::invalid_argument("CoraTensor::stack: the eigen backend holds one set, not a "
                                        "batch; use the torch backend");
        return wrap(Mat(m));
    }
};

// Backend -----------------------------------------------------------------------------------------

/// Makes Eigen tensors; matrices only (no batch dimensions) and CPU only.
struct EigenBackend : Tensor::Backend {
    std::string name() const override { return "eigen"; }
    Tensor::Impl::Ptr make(const std::vector<double> &data, const std::vector<int64_t> &shape,
                           const std::string &device) const override {
        EigenTensor::requireCpu(device);
        if (shape.size() != 2)
            throw std::invalid_argument("CoraTensor: the eigen backend holds matrices only");
        return EigenTensor::wrap(Eigen::Map<const RowMajor>(data.data(), shape[0], shape[1]));
    }
};

} // namespace


// ===========================================  MAIN  =========================================== //

// Conversion --------------------------------------------------------------------------------------

std::shared_ptr<const Tensor::Backend> eigenBackend() { return std::make_shared<EigenBackend>(); }

Tensor fromEigen(const Eigen::MatrixXd &m) { return Tensor(EigenTensor::wrap(m)); }

Eigen::MatrixXd toEigen(const Tensor &t) { return static_cast<const EigenTensor &>(t.impl()).m; }

bool isEigen(const Tensor &t) { return dynamic_cast<const EigenTensor *>(&t.impl()) != nullptr; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
