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

std::shared_ptr<const Tensor::Backend> &current() {
    static std::shared_ptr<const Tensor::Backend> backend;
    return backend;
}

std::shared_ptr<const Tensor::Backend> make_backend(const std::string &spec) {
    // name[:device][,custom_backward]
    const std::size_t comma = spec.find(',');
    const std::string head = spec.substr(0, comma);
    const std::string option = comma == std::string::npos ? "" : spec.substr(comma + 1);
    const std::size_t colon = head.find(':');
    const std::string name = head.substr(0, colon);
    const std::string device = colon == std::string::npos ? "cpu" : head.substr(colon + 1);

    if (!option.empty() && option != "custom_backward")
        throw std::invalid_argument("CoraTensor: unknown backend option: " + option);
    if (name == "eigen") {
        if (colon != std::string::npos || !option.empty())
            throw std::invalid_argument("CoraTensor: the eigen backend has no options");
        return eigen_backend();
    }
#ifdef CORACPP_TORCH
    if (name == "torch") return torch_backend(torch::Device(device), option == "custom_backward");
#endif
    throw std::invalid_argument("CoraTensor: no backend " + name + " in this build");
}

int64_t count_of(const std::vector<int64_t> &shape) {
    int64_t count = 1;
    for (const int64_t d : shape) count *= d;
    return count;
}

} // namespace

void set_backend(const std::string &spec) { current() = make_backend(spec); }

const Tensor::Backend &backend() {
    if (!current()) {
        const char *env = std::getenv("CORACPP_BACKEND");
#ifdef CORACPP_TORCH
        current() = make_backend(env && *env ? env : "torch");
#else
        current() = make_backend(env && *env ? env : "eigen");
#endif
    }
    return *current();
}

Tensor::Tensor(std::initializer_list<Entry> column, const std::string &device) {
    std::vector<double> data;
    for (const Entry &e : column) data.push_back(e.value);
    *this = from_data(data, {static_cast<int64_t>(data.size()), 1}, device);
}

Tensor::Tensor(std::initializer_list<std::initializer_list<double>> rows,
               const std::string &device) {
    std::vector<double> data;
    for (const auto &row : rows) {
        if (row.size() != rows.begin()->size())
            throw std::invalid_argument("CoraTensor: the rows differ in length");
        data.insert(data.end(), row.begin(), row.end());
    }
    *this = from_data(
        data, {static_cast<int64_t>(rows.size()), static_cast<int64_t>(rows.begin()->size())},
        device);
}

Tensor Tensor::from_data(const std::vector<double> &data, const std::vector<int64_t> &shape,
                         const std::string &device) {
    if (static_cast<int64_t>(data.size()) != count_of(shape))
        throw std::invalid_argument("CoraTensor: the data does not fill the shape");
    return Tensor(backend().make(data, shape, device));
}

Tensor Tensor::like(const Tensor &like, const std::vector<double> &data,
                    const std::vector<int64_t> &shape) {
    if (static_cast<int64_t>(data.size()) != count_of(shape))
        throw std::invalid_argument("CoraTensor: the data does not fill the shape");
    return Tensor(like.impl().like(data, shape));
}

Tensor Tensor::zeros(const std::vector<int64_t> &shape, const std::string &device) {
    return from_data(std::vector<double>(static_cast<std::size_t>(count_of(shape)), 0.0), shape,
                     device);
}

Tensor Tensor::eye(int64_t n, const std::string &device) {
    return zeros({n, n}, device).eye_like();
}

Tensor Tensor::binary(const Tensor &o, Impl::Ptr (Impl::*op)(const Impl &) const) const {
    if (!impl_ || !o.impl_) throw std::invalid_argument("CoraTensor: an undefined tensor");
    if (typeid(*impl_) != typeid(*o.impl_))
        throw std::invalid_argument("CoraTensor: operands are on different backends");
    return Tensor(((*impl_).*op)(*o.impl_));
}

Tensor Tensor::cat_last(const std::vector<Tensor> &parts) {
    if (parts.empty()) throw std::invalid_argument("CoraTensor: nothing to concatenate");
    std::vector<const Impl *> rest;
    for (std::size_t i = 1; i < parts.size(); ++i) {
        if (typeid(parts[i].impl()) != typeid(parts[0].impl()))
            throw std::invalid_argument("CoraTensor: operands are on different backends");
        rest.push_back(&parts[i].impl());
    }
    return Tensor(parts[0].impl().cat_last(rest));
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
