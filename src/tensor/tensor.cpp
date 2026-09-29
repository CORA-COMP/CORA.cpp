// tensor - the Tensor front end: construction, the current backend, and backend selection

#include "tensor/tensor.h"

#include "tensor/eigen.h"
#ifdef CORACPP_TORCH
#include "tensor/torch.h"
#endif

#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <typeinfo>

namespace cora::ct {
namespace {

/// The backend new tensors are made on; empty until first used.
std::shared_ptr<const Tensor::Backend> &current() {
    static std::shared_ptr<const Tensor::Backend> backend;
    return backend;
}

/// The backend named by spec: "eigen", "torch", "torch:cuda", "torch:cuda:1", each with an optional
/// ",customBackward" (libtorch only).
std::shared_ptr<const Tensor::Backend> makeBackend(const std::string &spec) {
    // spec is name[:device][,customBackward]
    const std::size_t comma = spec.find(',');
    const std::string head = spec.substr(0, comma);
    const std::string option = comma == std::string::npos ? "" : spec.substr(comma + 1);
    const std::size_t colon = head.find(':');
    const std::string name = head.substr(0, colon);
    const std::string device = colon == std::string::npos ? "cpu" : head.substr(colon + 1);

    if (!option.empty() && option != "customBackward")
        throw std::invalid_argument("CoraTensor: unknown backend option: " + option);
    if (name == "eigen") {
        if (colon != std::string::npos || !option.empty())
            throw std::invalid_argument("CoraTensor: the eigen backend has no options");
        return eigenBackend();
    }
#ifdef CORACPP_TORCH
    if (name == "torch") return torchBackend(torch::Device(device), option == "customBackward");
#endif
    throw std::invalid_argument("CoraTensor: no backend " + name + " in this build");
}

/// The number of elements of a tensor of this shape.
int64_t countOf(const std::vector<int64_t> &shape) {
    int64_t count = 1;
    for (const int64_t d : shape) count *= d;
    return count;
}

} // namespace

void setBackend(const std::string &spec) { current() = makeBackend(spec); }

const Tensor::Backend &backend() {
    if (!current()) {
        const char *env = std::getenv("CORACPP_BACKEND");
#ifdef CORACPP_TORCH
        current() = makeBackend(env && *env ? env : "torch");
#else
        current() = makeBackend(env && *env ? env : "eigen");
#endif
    }
    return *current();
}

Tensor::Tensor(std::initializer_list<Entry> column, const std::string &device) {
    std::vector<double> data;
    for (const Entry &e : column) data.push_back(e.value);
    *this = fromData(data, {static_cast<int64_t>(data.size()), 1}, device);
}

Tensor::Tensor(std::initializer_list<std::initializer_list<double>> rows,
               const std::string &device) {
    std::vector<double> data;
    for (const auto &row : rows) {
        if (row.size() != rows.begin()->size())
            throw std::invalid_argument("CoraTensor: the rows differ in length");
        data.insert(data.end(), row.begin(), row.end());
    }
    *this = fromData(
        data, {static_cast<int64_t>(rows.size()), static_cast<int64_t>(rows.begin()->size())},
        device);
}

Tensor Tensor::fromData(const std::vector<double> &data, const std::vector<int64_t> &shape,
                         const std::string &device) {
    if (static_cast<int64_t>(data.size()) != countOf(shape))
        throw std::invalid_argument("CoraTensor: the data does not fill the shape");
    return Tensor(backend().make(data, shape, device));
}

Tensor Tensor::like(const Tensor &like, const std::vector<double> &data,
                    const std::vector<int64_t> &shape) {
    if (static_cast<int64_t>(data.size()) != countOf(shape))
        throw std::invalid_argument("CoraTensor: the data does not fill the shape");
    return Tensor(like.impl().like(data, shape));
}

Tensor Tensor::zeros(const std::vector<int64_t> &shape, const std::string &device) {
    return fromData(std::vector<double>(static_cast<std::size_t>(countOf(shape)), 0.0), shape,
                     device);
}

Tensor Tensor::eye(int64_t n, const std::string &device) {
    return zeros({n, n}, device).eyeLike();
}

Tensor Tensor::binary(const Tensor &o, Impl::Ptr (Impl::*op)(const Impl &) const) const {
    if (!impl_ || !o.impl_) throw std::invalid_argument("CoraTensor: an undefined tensor");
    if (typeid(*impl_) != typeid(*o.impl_))
        throw std::invalid_argument("CoraTensor: operands are on different backends");
    return Tensor(((*impl_).*op)(*o.impl_));
}

Tensor Tensor::catLast(const std::vector<Tensor> &parts) {
    if (parts.empty()) throw std::invalid_argument("CoraTensor: nothing to concatenate");
    std::vector<const Impl *> rest;
    for (std::size_t i = 1; i < parts.size(); ++i) {
        if (typeid(parts[i].impl()) != typeid(parts[0].impl()))
            throw std::invalid_argument("CoraTensor: operands are on different backends");
        rest.push_back(&parts[i].impl());
    }
    return Tensor(parts[0].impl().catLast(rest));
}

std::ostream &operator<<(std::ostream &out, const Tensor &t) {
    const std::vector<int64_t> shape = t.shape();
    const std::vector<double> data = t.data();
    const int64_t cols = shape.back();
    out << "Tensor(shape [";
    for (std::size_t i = 0; i < shape.size(); ++i) out << (i ? ", " : "") << shape[i];
    out << "])";
    for (std::size_t i = 0; i < data.size(); ++i) out << (i % cols == 0 ? "\n  " : " ") << data[i];
    return out;
}

} // namespace cora::ct
