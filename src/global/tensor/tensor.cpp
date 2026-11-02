// tensor - the Tensor front end: construction, the current backend, and backend selection
//
// Syntax:   BackendScope scope("torch");   Tensor t({{1, 2}, {3, 4}});   Tensor z =
// Tensor::zeros({2, 2}); See also: tensor.h, eigen.cpp, torch.cpp

#include "cora/global/tensor/tensor.h"
#include "cora/global/error.h"

#include "cora/global/backend/eigen.h"
#include "cora/global/backend/lean.h"
#ifdef CORACPP_TORCH
#include "cora/global/backend/torch.h"
#endif

#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <typeinfo>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The backend named by spec: "eigen", "torch", "torch:cuda", "torch:cuda:1", each with an optional
/// ",custom_backward" (libtorch only).
std::shared_ptr<const Tensor::Backend> aux_make_backend(const std::string &spec) {
    // spec is name[:device][,custom_backward]
    const std::size_t comma = spec.find(',');
    const std::string head = spec.substr(0, comma);
    const std::string option = comma == std::string::npos ? "" : spec.substr(comma + 1);
    const std::size_t colon = head.find(':');
    const std::string name = head.substr(0, colon);
    const std::string device = colon == std::string::npos ? "cpu" : head.substr(colon + 1);

    if (!option.empty() && option != "custom_backward")
        CORA_FAIL(Error::wrong_value,
                  "unknown backend option '" + option + "'; the only option is custom_backward");
    // Eigen has neither devices nor options.
    if (name == "eigen") {
        if (colon != std::string::npos || !option.empty())
            CORA_FAIL(Error::wrong_value,
                      "the eigen backend has no device or options: use 'eigen'");
        return eigen_backend();
    }
    // The reference backend of the CORALean oracle.
    if (name == "lean") {
        if (colon != std::string::npos || !option.empty())
            CORA_FAIL(Error::wrong_value, "the lean backend has no device or options: use 'lean'");
        return lean_backend();
    }
    // libtorch, where it is built in.
#ifdef CORACPP_TORCH
    if (name == "torch") return torch_backend(torch_device(device), option == "custom_backward");
#endif
    CORA_FAIL(Error::wrong_value, "no backend '" + name + "' in this build; use eigen, " +
#ifdef CORACPP_TORCH
                                      "lean or torch"
#else
                                      "or lean (this build has no libtorch)"
#endif
    );
}

/// The process default, made on first use, and this thread's scope (none: the default).
std::atomic<const Tensor::Backend *> &aux_default() {
    static std::atomic<const Tensor::Backend *> current(nullptr);
    return current;
}
thread_local const Tensor::Backend *aux_scoped = nullptr;

/// The number of elements of a tensor of this shape.
int64_t aux_count_of(const std::vector<int64_t> &shape) {
    int64_t count = 1;
    for (const int64_t d : shape) count *= d;
    return count;
}

/// The parts after the first of a concatenation, all on the backend of the first one.
std::vector<const Tensor::Impl *> aux_rest_of(const std::vector<Tensor> &parts) {
    if (parts.empty()) CORA_FAIL(Error::wrong_value, "CoraTensor: nothing to concatenate");
    std::vector<const Tensor::Impl *> rest;
    for (std::size_t i = 1; i < parts.size(); ++i) {
        if (typeid(parts[i].impl()) != typeid(parts[0].impl()))
            CORA_FAIL(Error::wrong_value, "CoraTensor: operands are on different backends");
        rest.push_back(&parts[i].impl());
    }
    return rest;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Backend Selection -------------------------------------------------------------------------------

// The backend of a spec, made once and kept for the program: a scope then holds a plain pointer.
const Tensor::Backend &backend(const std::string &spec) {
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<const Tensor::Backend>> made;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = made.find(spec);
    if (it == made.end()) it = made.emplace(spec, aux_make_backend(spec)).first;
    return *it->second;
}

// This thread's backend: a scope's, else the process default ($CORACPP_BACKEND or the best built
// in).
const Tensor::Backend &backend() {
    if (aux_scoped) return *aux_scoped;
    const Tensor::Backend *current = aux_default().load(std::memory_order_acquire);
    if (current) return *current;
    const char *env = std::getenv("CORACPP_BACKEND");
#ifdef CORACPP_TORCH
    const Tensor::Backend &chosen = backend(env && *env ? env : "torch");
#else
    const Tensor::Backend &chosen = backend(env && *env ? env : "eigen");
#endif
    // the first thread to get here sets the default; another may have chosen the same meanwhile
    const Tensor::Backend *expected = nullptr;
    aux_default().compare_exchange_strong(expected, &chosen, std::memory_order_acq_rel);
    return *aux_default().load(std::memory_order_acquire);
}

void set_default_backend(const std::string &spec) {
    aux_default().store(&backend(spec), std::memory_order_release);
}

BackendScope::BackendScope(const std::string &spec) : previous_(aux_scoped) {
    aux_scoped = &backend(spec);
}

BackendScope::~BackendScope() { aux_scoped = previous_; }

// Construction ------------------------------------------------------------------------------------

// A column from a list of numbers.
Tensor::Tensor(std::initializer_list<Entry> column, const std::string &device) {
    std::vector<double> data;
    for (const Entry &e : column) data.push_back(e.value);
    *this = from_data(data, {static_cast<int64_t>(data.size()), 1}, device);
}

// A matrix from a list of rows, which must all be as long as the first.
Tensor::Tensor(std::initializer_list<std::initializer_list<double>> rows,
               const std::string &device) {
    std::vector<double> data;
    for (const auto &row : rows) {
        if (row.size() != rows.begin()->size())
            CORA_FAIL(Error::wrong_value, "CoraTensor: the rows of a matrix differ in length");
        data.insert(data.end(), row.begin(), row.end());
    }
    *this = from_data(
        data, {static_cast<int64_t>(rows.size()), static_cast<int64_t>(rows.begin()->size())},
        device);
}

// Host data of a shape, on the current backend.
Tensor Tensor::from_data(const std::vector<double> &data, const std::vector<int64_t> &shape,
                         const std::string &device) {
    if (static_cast<int64_t>(data.size()) != aux_count_of(shape))
        CORA_FAIL(Error::wrong_value, "CoraTensor: the data does not fill the shape");
    return Tensor(backend().make(data, shape, device));
}

// Host data of a shape, on the backend and device of an existing tensor.
Tensor Tensor::like(const Tensor &like, const std::vector<double> &data,
                    const std::vector<int64_t> &shape) {
    if (static_cast<int64_t>(data.size()) != aux_count_of(shape))
        CORA_FAIL(Error::wrong_value, "CoraTensor: the data does not fill the shape");
    return Tensor(like.impl().like(data, shape));
}

Tensor Tensor::zeros(const std::vector<int64_t> &shape, const std::string &device) {
    const std::vector<double> zeros(static_cast<std::size_t>(aux_count_of(shape)), 0.0);
    return from_data(zeros, shape, device);
}

Tensor Tensor::ones(const std::vector<int64_t> &shape, const std::string &device) {
    const std::vector<double> ones(static_cast<std::size_t>(aux_count_of(shape)), 1.0);
    return from_data(ones, shape, device);
}

Tensor Tensor::eye(int64_t n, const std::string &device) {
    return zeros({n, n}, device).eye_like();
}

// Combining Tensors -------------------------------------------------------------------------------

// Applies a binary operation of the backend; the operands must belong to the same one.
Tensor Tensor::binary(const Tensor &o, Impl::Ptr (Impl::*op)(const Impl &) const) const {
    if (!impl_ || !o.impl_) CORA_FAIL(Error::wrong_value, "CoraTensor: an undefined tensor");
    if (typeid(*impl_) != typeid(*o.impl_))
        CORA_FAIL(Error::wrong_value, "CoraTensor: operands are on different backends");
    return Tensor(((*impl_).*op)(*o.impl_));
}

// Joins the parts along the last dimension; all of them on one backend.
Tensor Tensor::cat_last(const std::vector<Tensor> &parts) {
    return Tensor(parts[0].impl().cat_last(aux_rest_of(parts)));
}

// Joins the parts along the rows; all of them on one backend.
Tensor Tensor::cat_rows(const std::vector<Tensor> &parts) {
    return Tensor(parts[0].impl().cat_rows(aux_rest_of(parts)));
}

// Stacks the parts along a new leading dimension; all of them on one backend.
Tensor Tensor::stack(const std::vector<Tensor> &parts) {
    if (parts.empty()) CORA_FAIL(Error::wrong_value, "CoraTensor: nothing to stack");
    std::vector<const Impl *> rest;
    for (std::size_t i = 1; i < parts.size(); ++i) {
        if (typeid(parts[i].impl()) != typeid(parts[0].impl()))
            CORA_FAIL(Error::wrong_value, "CoraTensor: operands are on different backends");
        if (parts[i].shape() != parts[0].shape())
            CORA_FAIL(Error::wrong_value, "CoraTensor::stack: the tensors differ in shape");
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
