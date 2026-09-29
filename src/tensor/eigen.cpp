#include "tensor/eigen.h"

#include <stdexcept>

#include <unsupported/Eigen/MatrixFunctions>

namespace cora::ct {
namespace {

using Mat = Eigen::MatrixXd;
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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

    Ptr add(const Impl &o) const override { return wrap(m + of(o)); }
    Ptr sub(const Impl &o) const override { return wrap(m - of(o)); }
    Ptr scale(double s) const override { return wrap(m * s); }
    Ptr matmul(const Impl &o) const override { return wrap(m * of(o)); }
    Ptr transpose() const override { return wrap(m.transpose()); }
    Ptr abs() const override { return wrap(m.cwiseAbs()); }
    Ptr pos() const override { return wrap(m.cwiseMax(0.0)); }
    Ptr neg() const override { return wrap(m.cwiseMin(0.0)); }
    Ptr expm() const override { return wrap(m.exp()); }
    Ptr sum_last() const override { return wrap(m.rowwise().sum()); }
    Ptr diag() const override { return wrap(Mat(m.col(0).asDiagonal())); }
    Ptr eye_like() const override { return wrap(Mat::Identity(m.rows(), m.cols())); }
    Ptr zeros_like() const override { return wrap(Mat::Zero(m.rows(), m.cols())); }

    Ptr cat_last(const std::vector<const Impl *> &rest) const override {
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
};

struct EigenBackend : Tensor::Backend {
    std::string name() const override { return "eigen"; }
    Tensor::Impl::Ptr make(const std::vector<double> &data,
                           const std::vector<int64_t> &shape) const override {
        if (shape.size() != 2)
            throw std::invalid_argument("CoraTensor: the eigen backend holds matrices only");
        return EigenTensor::wrap(Eigen::Map<const RowMajor>(data.data(), shape[0], shape[1]));
    }
};

} // namespace

std::shared_ptr<const Tensor::Backend> eigen_backend() { return std::make_shared<EigenBackend>(); }

Tensor from_eigen(const Eigen::MatrixXd &m) { return Tensor(EigenTensor::wrap(m)); }

Eigen::MatrixXd to_eigen(const Tensor &t) { return static_cast<const EigenTensor &>(t.impl()).m; }

} // namespace cora::ct
