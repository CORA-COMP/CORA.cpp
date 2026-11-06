// tensor - the Tensor front end: construction, operand checks, copy-on-write, backend selection
//
// Every operation checks that its operands share backend, device and dtype and then calls the
// backend. An rvalue operand that alone owns its memory is reused in place (copy-on-write). The
// derived operations are in derived.cpp, the fused ones in fused.cpp, the table in ops.cpp.
//
// Syntax:   BackendScope scope("torch");   Tensor t({{1, 2}, {3, 4}});   Tensor z = t.zeros_like();
// See also: tensor.h, derived.cpp, eigen.cpp, torch.cpp
//
// Authors:       Claude Sonnet 5.5
// Written:       01-October-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/tensor/tensor.h"
#include "cora/global/error.h"
#include "global/tensor/derived.h"

#include "cora/global/backend/eigen.h"
#include "cora/global/backend/lean.h"
#ifdef CORACPP_TORCH
#include "cora/global/backend/torch.h"
#endif

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <ostream>

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
    if (name == "torch") {
        const std::string device = colon == std::string::npos ? "cpu" : head.substr(colon + 1);
        return torch_backend(torch_device(device), option == "custom_backward");
    }
#endif
#ifdef CORACPP_TORCH
    const std::string others = "lean or torch";
#else
    const std::string others = "or lean (this build has no libtorch)";
#endif
    CORA_FAIL(Error::wrong_value, "no backend '" + name + "' in this build; use eigen, " + others);
}

/// The process default, made on first use, and this thread's scope (none: the default).
std::atomic<const Tensor::Backend *> &aux_default() {
    static std::atomic<const Tensor::Backend *> current(nullptr);
    return current;
}
thread_local const Tensor::Backend *aux_scoped = nullptr;

/// The device of a name: "cpu", "gpu" or "cuda" (device 0), "cuda:<index>".
Device aux_device(const std::string &name) {
    if (name.empty() || name == "cpu") return {};
    if (name == "gpu" || name == "cuda") return {Device::Type::cuda, 0};
    if (name.rfind("cuda:", 0) == 0)
        return {Device::Type::cuda, static_cast<int8_t>(std::atoi(name.c_str() + 5))};
    CORA_FAIL(Error::wrong_value,
              "CoraTensor: unknown device '" + name + "'; use cpu, gpu, cuda or cuda:<index>");
}

/// The parts joined along `axis`: checked to share one place and rank, passed on without a heap
/// copy of the list when it is short.
template <class It>
Tensor aux_concat(It first, It last, int axis) {
    const std::size_t count = static_cast<std::size_t>(last - first);
    if (count == 0) CORA_FAIL(Error::wrong_value, "CoraTensor::concat: no tensors given");
    const Tensor &head = *first;
    const int a = detail::axis_of(axis, head.ndim(), "concat");
    if (count == 1) return head;
    constexpr std::size_t kShort = 16;
    const Tensor::Impl *small[kShort];
    std::vector<const Tensor::Impl *> large;
    if (count > kShort) large.resize(count);
    const Tensor::Impl **impls = count > kShort ? large.data() : small;
    std::size_t i = 0;
    for (It it = first; it != last; ++it, ++i) {
        detail::check_pair(head, *it, "concat");
        if (it->ndim() != head.ndim())
            CORA_FAIL(Error::dimension_mismatch, "CoraTensor::concat: the parts differ in rank");
        impls[i] = &it->impl();
    }
    // the parts concatenated along the axis
    return Tensor(head.impl().concat({impls, count}, a));
}

/// The table's operation of a comparison or a mask function: ops.def lists them in the order of
/// Tensor::Binary and Tensor::Unary.
static_assert(int(TensorOp::greater_equal) - int(TensorOp::equal) ==
              int(Tensor::Binary::greater_equal) - int(Tensor::Binary::equal));
static_assert(int(TensorOp::isfinite) - int(TensorOp::isnan) ==
              int(Tensor::Unary::isfinite) - int(Tensor::Unary::isnan));
TensorOp aux_op_of(Tensor::Binary op) {
    return static_cast<TensorOp>(static_cast<int>(TensorOp::equal) + static_cast<int>(op) -
                                 static_cast<int>(Tensor::Binary::equal));
}
TensorOp aux_op_of(Tensor::Unary op) {
    return static_cast<TensorOp>(static_cast<int>(TensorOp::isnan) + static_cast<int>(op) -
                                 static_cast<int>(Tensor::Unary::isnan));
}

/// The names of the backends by id.
const char *aux_backend_name(uint8_t id) {
    if (id == backend_id::eigen) return "eigen";
    if (id == backend_id::lean) return "lean";
    return "torch";
}

/// "the eigen backend", "cuda:0", "float32": where a tensor differs from another.
std::string aux_place(const Tensor::Impl &a, const Tensor::Impl &b) {
    if (a.backend != b.backend)
        return std::string("the ") + aux_backend_name(a.backend) + " backend";
    if (a.device != b.device) return a.device.name();
    return dtype_name(a.dtype);
}

std::string aux_text(const Shape &s) {
    std::string text = "(";
    for (int64_t i = 0; i < s.size(); ++i) text += (i ? ", " : "") + std::to_string(s[i]);
    return text + ")";
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Shape, DType, Device ----------------------------------------------------------------------------

void Shape::aux_too_many_axes() {
    CORA_FAIL(Error::not_supported,
              "CoraTensor: at most " + std::to_string(kMaxRank) + " axes are supported");
}


// The same rank and sizes.

const char *dtype_name(DType dtype) { return dtype == DType::float32 ? "float32" : "float64"; }

std::string Device::name() const {
    return type == Type::cpu ? "cpu" : "cuda:" + std::to_string(index);
}

// Operand Checks ----------------------------------------------------------------------------------

// Always on, checks or not: operands of two backends would be read as the wrong type.
void detail::check_pair(const Tensor &a, const Tensor &b, const char *op) {
    if (!a.defined() || !b.defined())
        CORA_FAIL(Error::wrong_value, std::string("CoraTensor::") + op + ": an undefined tensor");
    const Tensor::Impl &x = a.impl(), &y = b.impl();
    if (x.backend == y.backend && x.device == y.device && x.dtype == y.dtype) return;
    CORA_FAIL(Error::wrong_value, std::string("CoraTensor::") + op +
                                      ": expected all tensors on one backend, device and dtype, "
                                      "but the first is on " +
                                      aux_place(x, y) + " and the second on " + aux_place(y, x));
}

int detail::axis_of(int axis, int64_t rank, const char *op) {
    const int64_t a = axis < 0 ? axis + rank : axis;
    if (a < 0 || a >= rank)
        CORA_FAIL(Error::wrong_value, std::string("CoraTensor::") + op + ": no axis " +
                                          std::to_string(axis) + " in a tensor of " +
                                          std::to_string(rank) + " axes");
    return static_cast<int>(a);
}

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

namespace detail {
void set_backend_threads([[maybe_unused]] int n) {
#ifdef CORACPP_TORCH
    torch_set_threads(n);
#endif
}
} // namespace detail

// This thread's backend: a scope's, else the process default ($CORACPP_BACKEND, else the best).
const Tensor::Backend &backend() {
    if (aux_scoped) return *aux_scoped;
    const Tensor::Backend *current = aux_default().load(std::memory_order_acquire);
    if (current) return *current;
    const char *env = std::getenv("CORACPP_BACKEND");
#ifdef CORACPP_TORCH
    const std::string name = env && *env ? env : "torch";
#else
    const std::string name = env && *env ? env : "eigen";
#endif
    const Tensor::Backend &chosen = backend(name);
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

// Construction from Numbers -----------------------------------------------------------------------

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
Tensor Tensor::from_data(const std::vector<double> &data, const Shape &shape,
                         const std::string &device, DType dtype) {
    if (static_cast<int64_t>(data.size()) != shape.numel())
        CORA_FAIL(Error::wrong_value, "CoraTensor: the data does not fill the shape");
    return Tensor(backend().make(data, shape, device, dtype));
}

Tensor Tensor::zeros(const Shape &shape, const std::string &device) {
    return from_data(std::vector<double>(static_cast<std::size_t>(shape.numel()), 0.0), shape,
                     device);
}

Tensor Tensor::ones(const Shape &shape, const std::string &device) {
    return from_data(std::vector<double>(static_cast<std::size_t>(shape.numel()), 1.0), shape,
                     device);
}

Tensor Tensor::eye(int64_t n, const std::string &device) { return zeros({1}, device).eye_like(n); }

// Construction Like a Tensor ----------------------------------------------------------------------

Tensor Tensor::like(const Tensor &like, const std::vector<double> &data, const Shape &shape) {
    if (static_cast<int64_t>(data.size()) != shape.numel())
        CORA_FAIL(Error::wrong_value, "CoraTensor: the data does not fill the shape");
    return Tensor(like.impl().like(data.data(), shape));
}

Tensor Tensor::full_like(const Shape &shape, double value) const {
    return Tensor(impl_->full(shape, value));
}

Tensor Tensor::arange_like(int64_t n) const { return Tensor(impl_->arange(n)); }

// One where the row index equals the column index.
Tensor Tensor::eye_like(int64_t n) const {
    const Tensor i = arange_like(n);
    return i.reshape({n, 1}).equal(i.reshape({1, n}));
}

// The identity of the last two axes, repeated over the batch.
Tensor Tensor::eye_like() const {
    const int64_t n = size(-2), m = size(-1);
    const Tensor eye = arange_like(n).reshape({n, 1}).equal(arange_like(m).reshape({1, m}));
    return eye.broadcast_to(shape());
}

// Access ------------------------------------------------------------------------------------------

int64_t Tensor::size(int axis) const { return impl_->shape[detail::axis_of(axis, ndim(), "size")]; }

const char *Tensor::backend_name() const { return aux_backend_name(impl_->backend); }

double Tensor::item() const {
    if (numel() != 1)
        CORA_FAIL(Error::wrong_value,
                  "CoraTensor::item: needs one entry, got " + aux_text(shape()));
    return data()[0];
}

Tensor Tensor::to(const std::string &device) const { return Tensor(impl_->to(aux_device(device))); }

Tensor Tensor::to(DType dtype) const { return Tensor(impl_->to(dtype)); }

// Elementwise -------------------------------------------------------------------------------------

Tensor::Impl *Tensor::unique() {
    return impl_ && impl_.use_count() == 1 ? const_cast<Impl *>(impl_.get()) : nullptr;
}

// Comparisons produce masks, which carry no gradient.
Tensor Tensor::binary(Binary op, const Tensor &o) const & {
    detail::check_pair(*this, o, "binary");
    if (op >= Binary::equal) {
        const Tensor a = detail::without_gradient(aux_op_of(op), *this);
        return Tensor(a.impl().binary(op, *detail::without_gradient(aux_op_of(op), o).impl_));
    }
    return Tensor(impl_->binary(op, *o.impl_));
}

// The left operand's memory is reused when nobody else holds it.
Tensor Tensor::binary(Binary op, const Tensor &o) && {
    detail::check_pair(*this, o, "binary");
    if (op < Binary::equal)
        if (Impl *mine = unique(); mine && mine->binary_inplace(op, *o.impl_))
            return std::move(*this);
    return static_cast<const Tensor &>(*this).binary(op, o);
}

Tensor Tensor::scalar_op(Binary op, double s, bool scalar_first) const & {
    if (op >= Binary::equal)
        return Tensor(
            detail::without_gradient(aux_op_of(op), *this).impl().scalar(op, s, scalar_first));
    return Tensor(impl_->scalar(op, s, scalar_first));
}

Tensor Tensor::scalar_op(Binary op, double s, bool scalar_first) && {
    if (op < Binary::equal)
        if (Impl *mine = unique(); mine && mine->scalar_inplace(op, s, scalar_first))
            return std::move(*this);
    return static_cast<const Tensor &>(*this).scalar_op(op, s, scalar_first);
}

// isnan, isinf and isfinite produce masks.
Tensor Tensor::unary(Unary op) const & {
    if (op >= Unary::isnan)
        return Tensor(detail::without_gradient(aux_op_of(op), *this).impl().unary(op));
    return Tensor(impl_->unary(op));
}

Tensor Tensor::unary(Unary op) && {
    if (op < Unary::isnan)
        if (Impl *mine = unique(); mine && mine->unary_inplace(op)) return std::move(*this);
    return static_cast<const Tensor &>(*this).unary(op);
}

Tensor Tensor::where(const Tensor &cond, const Tensor &a, const Tensor &b) {
    detail::check_pair(cond, a, "where");
    detail::check_pair(cond, b, "where");
    return Tensor(cond.detach().impl().where(*a.impl_, *b.impl_));
}

// Reductions --------------------------------------------------------------------------------------

Tensor Tensor::reduce(Reduce op, int axis, bool keepdims) const {
    const int a = detail::axis_of(axis, ndim(), "reduce");
    if (op == Reduce::argmax || op == Reduce::argmin) {
        const TensorOp which = op == Reduce::argmax ? TensorOp::argmax : TensorOp::argmin;
        return Tensor(detail::without_gradient(which, *this).impl().reduce(op, a, keepdims));
    }
    return Tensor(impl_->reduce(op, a, keepdims));
}

Tensor Tensor::argmax(int axis, bool keepdims) const {
    return reduce(Reduce::argmax, axis, keepdims);
}
Tensor Tensor::argmin(int axis, bool keepdims) const {
    return reduce(Reduce::argmin, axis, keepdims);
}

Tensor Tensor::cumsum(int axis) const {
    return Tensor(impl_->cumsum(detail::axis_of(axis, ndim(), "cumsum")));
}

// Shape and Selection -----------------------------------------------------------------------------

// A view where the backend can (the host backends always), else a copy.
Tensor Tensor::reshape(const Shape &s) const {
    if (s.numel() != numel())
        CORA_FAIL(Error::dimension_mismatch, "CoraTensor::reshape: " + aux_text(shape()) +
                                                 " cannot hold the shape " + aux_text(s));
    return Tensor(impl_->reshape(s));
}

Tensor Tensor::permute_dims(const std::vector<int> &axes) const {
    std::vector<int> normal;
    std::vector<bool> seen(ndim(), false);
    for (const int a : axes) {
        const int n = detail::axis_of(a, ndim(), "permute_dims");
        if (seen[n]) CORA_FAIL(Error::wrong_value, "CoraTensor::permute_dims: an axis twice");
        seen[n] = true;
        normal.push_back(n);
    }
    if (static_cast<int64_t>(normal.size()) != ndim())
        CORA_FAIL(Error::wrong_value, "CoraTensor::permute_dims: name every axis once");
    return Tensor(impl_->permute_dims(normal));
}

// The checks of the selections: axes in range, indices on the tensor's own backend.
Tensor Tensor::broadcast_to(const Shape &s) const { return Tensor(impl_->broadcast_to(s)); }

Tensor Tensor::slice(int axis, int64_t start, int64_t stop, int64_t step) const {
    const int a = detail::axis_of(axis, ndim(), "slice");
    const int64_t len = shape()[a];
    if (start < 0 || step < 1)
        CORA_FAIL(Error::wrong_value, "CoraTensor::slice: needs start >= 0 and step >= 1");
    start = std::min(start, len);
    return Tensor(impl_->slice(a, start, std::max(start, std::min(stop, len)), step));
}

Tensor Tensor::take(const Tensor &idx, int axis) const {
    detail::check_pair(*this, idx, "take");
    if (idx.ndim() != 1)
        CORA_FAIL(Error::wrong_value, "CoraTensor::take: the indices must be a 1-D tensor");
    return Tensor(impl_->take(*idx.detach().impl_, detail::axis_of(axis, ndim(), "take")));
}

// Host indices become a tensor like this one.
Tensor Tensor::take(const std::vector<int64_t> &idx, int axis) const {
    std::vector<double> values(idx.size());
    std::transform(idx.begin(), idx.end(), values.begin(),
                   [](int64_t i) { return static_cast<double>(i); });
    return take(like(*this, values, {static_cast<int64_t>(idx.size())}), axis);
}

Tensor Tensor::take_along_axis(const Tensor &idx, int axis) const {
    detail::check_pair(*this, idx, "take_along_axis");
    if (idx.ndim() != ndim())
        CORA_FAIL(Error::wrong_value,
                  "CoraTensor::take_along_axis: the indices need the rank of the tensor");
    const int a = detail::axis_of(axis, ndim(), "take_along_axis");
    return Tensor(impl_->take_along_axis(*idx.detach().impl_, a));
}

// One part is itself.
Tensor Tensor::concat(const std::vector<Tensor> &parts, int axis) {
    return aux_concat(parts.begin(), parts.end(), axis);
}

Tensor Tensor::concat(std::initializer_list<Tensor> parts, int axis) {
    return aux_concat(parts.begin(), parts.end(), axis);
}

// Linear Algebra ----------------------------------------------------------------------------------

Tensor Tensor::matmul(const Tensor &o) const {
    detail::check_pair(*this, o, "matmul");
    return Tensor(impl_->matmul(*o.impl_));
}

// The batches of A and B broadcast.
Tensor Tensor::solve(const Tensor &B) const {
    detail::check_pair(*this, B, "solve");
    return Tensor(impl_->solve(*B.impl_));
}

Tensor Tensor::cholesky() const { return Tensor(impl_->cholesky()); }

Tensor Tensor::kron(const Tensor &o) const {
    detail::check_pair(*this, o, "kron");
    return Tensor(impl_->kron(*o.impl_));
}

Tensor Tensor::expm() const { return Tensor(impl_->expm()); }

// The norms as ops.def defines them.
Tensor Tensor::matrix_norm(Norm ord) const { return Tensor(impl_->matrix_norm(ord)); }

Tensor Tensor::vector_norm(Norm ord, int axis, bool keepdims) const {
    return Tensor(impl_->vector_norm(ord, detail::axis_of(axis, ndim(), "vector_norm"), keepdims));
}

Tensor Tensor::diag() const {
    if (ndim() < 2 || shape().back() != 1)
        CORA_FAIL(Error::dimension_mismatch,
                  "CoraTensor::diag: needs a column (..., n, 1), got " + aux_text(shape()));
    return Tensor(impl_->diag());
}

// Sorting -------------------------------------------------------------------------------------

Tensor Tensor::argsort(int axis, bool descending) const {
    const int a = detail::axis_of(axis, ndim(), "argsort");
    return Tensor(detail::without_gradient(TensorOp::argsort, *this).impl().argsort(a, descending));
}

// sort and topk are stable: equal entries keep their order.
Tensor Tensor::sort(int axis, bool descending) const {
    return Tensor(impl_->sort(detail::axis_of(axis, ndim(), "sort"), descending));
}

std::pair<Tensor, Tensor> Tensor::topk(int64_t k, int axis, bool largest) const {
    const int a = detail::axis_of(axis, ndim(), "topk");
    if (k < 0 || k > shape()[a])
        CORA_FAIL(Error::wrong_value, "CoraTensor::topk: k = " + std::to_string(k) +
                                          " is outside 0.." + std::to_string(shape()[a]));
    const auto [values, indices] = impl_->topk(k, a, largest);
    return {Tensor(values), Tensor(indices)};
}

// Printing ----------------------------------------------------------------------------------------

// The shape, then the values row by row.
std::ostream &operator<<(std::ostream &out, const Tensor &t) {
    const Shape shape = t.shape();
    const std::vector<double> data = t.data();
    const int64_t cols = shape.empty() ? 1 : shape.back();
    out << "Tensor(shape " << aux_text(shape) << ")";
    for (std::size_t i = 0; i < data.size(); ++i) out << (i % cols == 0 ? "\n  " : " ") << data[i];
    return out << "\n";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
