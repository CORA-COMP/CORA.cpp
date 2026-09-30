// tensor - the Tensor front end: construction, the current backend, and backend selection
//
// Syntax:   setBackend("torch");   Tensor t({{1, 2}, {3, 4}});   Tensor z = Tensor::zeros({2, 2});
// See also: tensor.h, eigen.cpp, torch.cpp

#include "tensor/tensor.h"

#include "tensor/eigen.h"
#include "tensor/lean.h"
#ifdef CORACPP_TORCH
#include "tensor/torch.h"
#endif

#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <typeinfo>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The backend new tensors are made on; empty until first used.
std::shared_ptr<const Tensor::Backend> &aux_current() {
    static std::shared_ptr<const Tensor::Backend> backend;
    return backend;
}

/// The backend named by spec: "eigen", "torch", "torch:cuda", "torch:cuda:1", each with an optional
/// ",customBackward" (libtorch only).
std::shared_ptr<const Tensor::Backend> aux_makeBackend(const std::string &spec) {
    // spec is name[:device][,customBackward]
    const std::size_t comma = spec.find(',');
    const std::string head = spec.substr(0, comma);
    const std::string option = comma == std::string::npos ? "" : spec.substr(comma + 1);
    const std::size_t colon = head.find(':');
    const std::string name = head.substr(0, colon);
    const std::string device = colon == std::string::npos ? "cpu" : head.substr(colon + 1);

    if (!option.empty() && option != "customBackward")
        throw std::invalid_argument("CoraTensor: unknown backend option '" + option +
                                    "'; the only option is customBackward");
    // Eigen has neither devices nor options.
    if (name == "eigen") {
        if (colon != std::string::npos || !option.empty())
            throw std::invalid_argument("CoraTensor: the eigen backend has no device or options");
        return eigenBackend();
    }
    // The reference backend of the CORALean oracle.
    if (name == "lean") {
        if (colon != std::string::npos || !option.empty())
            throw std::invalid_argument("CoraTensor: the lean backend has no device or options");
        return leanBackend();
    }
    // libtorch, where it is built in.
#ifdef CORACPP_TORCH
    if (name == "torch") return torchBackend(torchDevice(device), option == "customBackward");
#endif
    throw std::invalid_argument("CoraTensor: no backend '" + name + "' in this build; use eigen, " +
#ifdef CORACPP_TORCH
                                "lean or torch"
#else
                                "or lean (this build has no libtorch)"
#endif
    );
}

/// The number of elements of a tensor of this shape.
int64_t aux_countOf(const std::vector<int64_t> &shape) {
    int64_t count = 1;
    for (const int64_t d : shape) count *= d;
    return count;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Backend Selection -------------------------------------------------------------------------------

// Chooses the backend of new tensors; the spec is read by aux_makeBackend.
void setBackend(const std::string &spec) { aux_current() = aux_makeBackend(spec); }

// The current backend: what setBackend chose, else $CORACPP_BACKEND, else the best one built in.
const Tensor::Backend &backend() {
    if (!aux_current()) {
        const char *env = std::getenv("CORACPP_BACKEND");
#ifdef CORACPP_TORCH
        aux_current() = aux_makeBackend(env && *env ? env : "torch");
#else
        aux_current() = aux_makeBackend(env && *env ? env : "eigen");
#endif
    }
    return *aux_current();
}

// Construction ------------------------------------------------------------------------------------

// A column from a list of numbers.
Tensor::Tensor(std::initializer_list<Entry> column, const std::string &device) {
    std::vector<double> data;
    for (const Entry &e : column) data.push_back(e.value);
    *this = fromData(data, {static_cast<int64_t>(data.size()), 1}, device);
}

// A matrix from a list of rows, which must all be as long as the first.
Tensor::Tensor(std::initializer_list<std::initializer_list<double>> rows,
               const std::string &device) {
    std::vector<double> data;
    for (const auto &row : rows) {
        if (row.size() != rows.begin()->size())
            throw std::invalid_argument("CoraTensor: the rows of a matrix differ in length");
        data.insert(data.end(), row.begin(), row.end());
    }
    *this = fromData(
        data, {static_cast<int64_t>(rows.size()), static_cast<int64_t>(rows.begin()->size())},
        device);
}

// Host data of a shape, on the current backend.
Tensor Tensor::fromData(const std::vector<double> &data, const std::vector<int64_t> &shape,
                        const std::string &device) {
    if (static_cast<int64_t>(data.size()) != aux_countOf(shape))
        throw std::invalid_argument("CoraTensor: the data does not fill the shape");
    return Tensor(backend().make(data, shape, device));
}

// Host data of a shape, on the backend and device of an existing tensor.
Tensor Tensor::like(const Tensor &like, const std::vector<double> &data,
                    const std::vector<int64_t> &shape) {
    if (static_cast<int64_t>(data.size()) != aux_countOf(shape))
        throw std::invalid_argument("CoraTensor: the data does not fill the shape");
    return Tensor(like.impl().like(data, shape));
}

Tensor Tensor::zeros(const std::vector<int64_t> &shape, const std::string &device) {
    const std::vector<double> zeros(static_cast<std::size_t>(aux_countOf(shape)), 0.0);
    return fromData(zeros, shape, device);
}

Tensor Tensor::ones(const std::vector<int64_t> &shape, const std::string &device) {
    const std::vector<double> ones(static_cast<std::size_t>(aux_countOf(shape)), 1.0);
    return fromData(ones, shape, device);
}

Tensor Tensor::eye(int64_t n, const std::string &device) {
    return zeros({n, n}, device).eyeLike();
}

// Combining Tensors -------------------------------------------------------------------------------

// Applies a binary operation of the backend; the operands must belong to the same one.
Tensor Tensor::binary(const Tensor &o, Impl::Ptr (Impl::*op)(const Impl &) const) const {
    if (!impl_ || !o.impl_) throw std::invalid_argument("CoraTensor: an undefined tensor");
    if (typeid(*impl_) != typeid(*o.impl_))
        throw std::invalid_argument("CoraTensor: operands are on different backends");
    return Tensor(((*impl_).*op)(*o.impl_));
}

// Joins the parts along the last dimension; all of them on one backend.
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

// Joins the parts along the rows; all of them on one backend.
Tensor Tensor::catRows(const std::vector<Tensor> &parts) {
    if (parts.empty()) throw std::invalid_argument("CoraTensor: nothing to concatenate");
    std::vector<const Impl *> rest;
    for (std::size_t i = 1; i < parts.size(); ++i) {
        if (typeid(parts[i].impl()) != typeid(parts[0].impl()))
            throw std::invalid_argument("CoraTensor: operands are on different backends");
        rest.push_back(&parts[i].impl());
    }
    return Tensor(parts[0].impl().catRows(rest));
}

// Stacks the parts along a new leading dimension; all of them on one backend.
Tensor Tensor::stack(const std::vector<Tensor> &parts) {
    if (parts.empty()) throw std::invalid_argument("CoraTensor: nothing to stack");
    std::vector<const Impl *> rest;
    for (std::size_t i = 1; i < parts.size(); ++i) {
        if (typeid(parts[i].impl()) != typeid(parts[0].impl()))
            throw std::invalid_argument("CoraTensor: operands are on different backends");
        if (parts[i].shape() != parts[0].shape())
            throw std::invalid_argument("CoraTensor::stack: the tensors differ in shape");
        rest.push_back(&parts[i].impl());
    }
    return Tensor(parts[0].impl().stack(rest));
}

// Printing ----------------------------------------------------------------------------------------

// The shape, then the values row by row.
std::ostream &operator<<(std::ostream &out, const Tensor &t) {
    const std::vector<int64_t> shape = t.shape();
    const std::vector<double> data = t.data();
    const int64_t cols = shape.back();
    out << "Tensor(shape [";
    for (std::size_t i = 0; i < shape.size(); ++i) out << (i ? ", " : "") << shape[i];
    out << "])";
    for (std::size_t i = 0; i < data.size(); ++i) out << (i % cols == 0 ? "\n  " : " ") << data[i];
    return out << "\n";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
