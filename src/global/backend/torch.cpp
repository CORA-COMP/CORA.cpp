// torch - the libtorch backend of Tensor: batched float64 or float32 tensors on the CPU or a GPU
//
// TorchTensor implements the operations of Tensor::Impl on torch::Tensor, so autograd
// differentiates through them; TorchBackend makes tensors from host numbers. Masks and indices
// are tensors of the operand's dtype, as on the other backends.

#include "cora/global/backend/torch.h"
#include "cora/global/error.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Binary = Tensor::Binary;
using Unary = Tensor::Unary;
using Reduce = Tensor::Reduce;
using Norm = Tensor::Norm;

constexpr double kInf = std::numeric_limits<double>::infinity();

// Matrix Exponential ------------------------------------------------------------------------------

/// `e^A` with a hand-written backward pass: the gradient of `<G, e^A>` with respect to `A`
/// is the upper-right block of `exp([[Aᵀ, G], [0, Aᵀ]])`, so nothing is stored per Taylor
/// term or squaring.
struct ExpmFunction : torch::autograd::Function<ExpmFunction> {
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, const torch::Tensor &a) {
        ctx->save_for_backward({a});
        return torch::matrix_exp(a);
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::tensor_list grad) {
        const torch::Tensor a = ctx->get_saved_variables()[0];
        const int64_t n = a.size(-1);
        const torch::Tensor at = a.transpose(-2, -1);
        const torch::Tensor top = torch::cat({at, grad[0]}, -1);
        const torch::Tensor bottom = torch::cat({torch::zeros_like(at), at}, -1);
        const torch::Tensor big = torch::matrix_exp(torch::cat({top, bottom}, -2));
        return {big.narrow(-2, 0, n).narrow(-1, n, n)};
    }
};

// Tensor ------------------------------------------------------------------------------------------

DType aux_dtype(const torch::Tensor &t) {
    return t.scalar_type() == torch::kFloat ? DType::float32 : DType::float64;
}

torch::ScalarType aux_scalar(DType d) {
    return d == DType::float32 ? torch::kFloat : torch::kDouble;
}

Device aux_device(const torch::Device &d) {
    if (!d.is_cuda()) return {};
    return {Device::Type::cuda, static_cast<int8_t>(d.has_index() ? d.index() : 0)};
}

/// A floating tensor: float64 and float32 stay, anything else becomes float64.
torch::Tensor aux_floating(const torch::Tensor &t) {
    return t.scalar_type() == torch::kDouble || t.scalar_type() == torch::kFloat
               ? t
               : t.to(torch::kDouble);
}

/// A torch::Tensor; custom_backward selects the hand-written backward pass of expm and is passed on
/// to every tensor made from this one.
struct TorchTensor : Tensor::Impl {
    torch::Tensor t;
    bool custom_backward;
    TorchTensor(torch::Tensor x, bool custom)
        : Tensor::Impl(Shape(x.sizes().vec()), aux_dtype(x), aux_device(x.device()),
                       backend_id::torch),
          t(std::move(x)), custom_backward(custom) {}

    Ptr wrap(torch::Tensor r) const {
        return std::make_shared<TorchTensor>(aux_floating(r), custom_backward);
    }
    static const torch::Tensor &of(const Impl &o) { return static_cast<const TorchTensor &>(o).t; }
    /// A mask or index result in this tensor's dtype.
    Ptr values(const torch::Tensor &r) const { return wrap(r.to(t.scalar_type())); }

    // Construction and access
    std::vector<double> data() const override {
        const torch::Tensor host = t.detach().to(torch::kCPU, torch::kDouble).contiguous();
        return std::vector<double>(host.data_ptr<double>(), host.data_ptr<double>() + host.numel());
    }
    Ptr to(Device d) const override {
        return wrap(t.to(d.type == Device::Type::cpu ? torch::Device(torch::kCPU)
                                                      : torch::Device(torch::kCUDA, d.index)));
    }
    Ptr to(DType d) const override { return wrap(t.to(aux_scalar(d))); }
    Ptr like(const double *v, const Shape &s) const override {
        const torch::Tensor host = torch::from_blob(const_cast<double *>(v),
                                                    std::vector<int64_t>(s), torch::kDouble);
        return wrap(host.to(t.device(), t.scalar_type(), false, true));
    }
    Ptr full(const Shape &s, double value) const override {
        return wrap(torch::full(std::vector<int64_t>(s), value, t.options().requires_grad(false)));
    }
    Ptr arange(int64_t n) const override {
        return wrap(torch::arange(n, t.options().requires_grad(false)));
    }
    bool requires_grad() const override { return t.requires_grad(); }
    Ptr detach() const override { return t.requires_grad() ? wrap(t.detach()) : self(); }

    // Elementwise
    static torch::Tensor apply(Binary op, const torch::Tensor &a, const torch::Tensor &b) {
        switch (op) {
        case Binary::add: return a + b;
        case Binary::subtract: return a - b;
        case Binary::multiply: return a * b;
        case Binary::divide: return a / b;
        case Binary::pow: return torch::pow(a, b);
        case Binary::maximum: return torch::maximum(a, b);
        case Binary::minimum: return torch::minimum(a, b);
        case Binary::remainder: return torch::remainder(a, b);
        case Binary::atan2: return torch::atan2(a, b);
        case Binary::equal: return torch::eq(a, b).to(a.scalar_type());
        case Binary::not_equal: return torch::ne(a, b).to(a.scalar_type());
        case Binary::less: return torch::lt(a, b).to(a.scalar_type());
        case Binary::less_equal: return torch::le(a, b).to(a.scalar_type());
        case Binary::greater: return torch::gt(a, b).to(a.scalar_type());
        case Binary::greater_equal: return torch::ge(a, b).to(a.scalar_type());
        }
        CORA_FAIL(Error::wrong_value, "CoraTensor: unknown elementwise operation");
    }
    Ptr binary(Binary op, const Impl &o) const override { return wrap(apply(op, t, of(o))); }
    Ptr scalar(Binary op, double s, bool scalar_first) const override {
        const torch::Tensor v = torch::scalar_tensor(s, t.options().requires_grad(false));
        return wrap(scalar_first ? apply(op, v, t) : apply(op, t, v));
    }
    // round halves away from zero (MATLAB), torch.round to even; a zero is +0, as CORALean's
    static torch::Tensor round(const torch::Tensor &x) {
        const torch::Tensor whole = torch::trunc(x);
        const torch::Tensor tie = (x - whole).abs() == 0.5;
        return torch::where(tie, whole + torch::sign(x), torch::round(x)) + 0.0;
    }
    Ptr unary(Unary op) const override {
        switch (op) {
        case Unary::negative: return wrap(-t);
        case Unary::abs: return wrap(t.abs());
        case Unary::sign: return wrap(t.sign());
        case Unary::floor: return wrap(t.floor());
        case Unary::ceil: return wrap(t.ceil());
        case Unary::round: return wrap(round(t));
        case Unary::sqrt: return wrap(t.sqrt());
        case Unary::exp: return wrap(t.exp());
        case Unary::log: return wrap(t.log());
        // the functions of torch, within the ulp bounds of ops.def of the C library's
        case Unary::sin: return wrap(t.sin());
        case Unary::cos: return wrap(t.cos());
        case Unary::tan: return wrap(t.tan());
        case Unary::asin: return wrap(t.asin());
        case Unary::acos: return wrap(t.acos());
        case Unary::atan: return wrap(t.atan());
        case Unary::sinh: return wrap(t.sinh());
        case Unary::cosh: return wrap(t.cosh());
        case Unary::tanh: return wrap(t.tanh());
        case Unary::asinh: return wrap(t.asinh());
        case Unary::acosh: return wrap(t.acosh());
        case Unary::atanh: return wrap(t.atanh());
        case Unary::isnan: return values(t.isnan());
        case Unary::isinf: return values(t.isinf());
        case Unary::isfinite: return values(t.isfinite());
        }
        CORA_FAIL(Error::wrong_value, "CoraTensor: unknown elementwise function");
    }
    Ptr where(const Impl &a, const Impl &b) const override {
        return wrap(torch::where(t != 0, of(a), of(b)));
    }

    // Reductions
    Ptr reduce(Reduce op, int axis, bool keepdims) const override {
        switch (op) {
        case Reduce::sum: return wrap(t.sum(axis, keepdims));
        case Reduce::prod: return wrap(t.prod(axis, keepdims));
        case Reduce::max: return wrap(std::get<0>(t.max(axis, keepdims)));
        case Reduce::min: return wrap(std::get<0>(t.min(axis, keepdims)));
        case Reduce::argmax: return values(t.argmax(axis, keepdims));
        case Reduce::argmin: return values(t.argmin(axis, keepdims));
        }
        CORA_FAIL(Error::wrong_value, "CoraTensor: unknown reduction");
    }
    Ptr cumsum(int axis) const override { return wrap(t.cumsum(axis)); }

    // Shape and selection
    Ptr reshape(const Shape &s) const override { return wrap(t.reshape(std::vector<int64_t>(s))); }
    Ptr permute_dims(const std::vector<int> &axes) const override {
        return wrap(t.permute(std::vector<int64_t>(axes.begin(), axes.end())));
    }
    Ptr broadcast_to(const Shape &s) const override {
        return wrap(t.expand(std::vector<int64_t>(s)));
    }
    Ptr slice(int axis, int64_t start, int64_t stop, int64_t step) const override {
        return wrap(t.slice(axis, start, stop, step));
    }
    Ptr take(const Impl &idx, int axis) const override {
        return wrap(t.index_select(axis, of(idx).to(torch::kLong)));
    }
    Ptr take_along_axis(const Impl &idx, int axis) const override {
        return wrap(torch::take_along_dim(t, of(idx).to(torch::kLong), axis));
    }
    Ptr concat(std::span<const Impl *const> parts, int axis) const override {
        std::vector<torch::Tensor> all;
        for (const Impl *r : parts) all.push_back(of(*r));
        return wrap(torch::cat(all, axis));
    }
    Ptr argsort(int axis, bool descending) const override {
        return values(torch::argsort(t, /*stable=*/true, axis, descending));
    }

    // Linear algebra
    Ptr matmul(const Impl &o) const override { return wrap(t.matmul(of(o))); }
    Ptr solve(const Impl &b) const override { return wrap(torch::linalg_solve(t, of(b))); }
    Ptr cholesky() const override { return wrap(torch::linalg_cholesky(t)); }
    std::pair<Ptr, Ptr> qr() const override {
        const auto [Q, R] = torch::linalg_qr(t, "reduced");
        return {wrap(Q), wrap(R)};
    }
    std::array<Ptr, 3> svd() const override {
        const auto [U, S, Vh] = torch::linalg_svd(t, /*full_matrices=*/false);
        return {wrap(U), wrap(S.unsqueeze(-1)), wrap(Vh.transpose(-2, -1))};
    }
    std::pair<Ptr, Ptr> eigh() const override {
        const auto [w, V] = torch::linalg_eigh(t);
        return {wrap(w.unsqueeze(-1)), wrap(V)};
    }

    // Native versions of derived operations
    Ptr expm() const override {
        return wrap(custom_backward ? ExpmFunction::apply(t) : torch::matrix_exp(t));
    }
    Ptr sort(int axis, bool descending) const override {
        return wrap(std::get<0>(torch::sort(t, /*stable=*/true, axis, descending)));
    }
    Ptr diag() const override { return wrap(torch::diag_embed(t.squeeze(-1))); }
    Ptr matrix_norm(Norm ord) const override {
        const std::vector<int64_t> dims = {-2, -1};
        if (ord == Norm::fro) return wrap(torch::linalg_matrix_norm(t, "fro", dims, true));
        const double p = ord == Norm::one ? 1.0 : ord == Norm::two ? 2.0 : kInf;
        return wrap(torch::linalg_matrix_norm(t, p, dims, true));
    }
    Ptr vector_norm(Norm ord, int axis, bool keepdims) const override {
        const double p = ord == Norm::one ? 1.0 : ord == Norm::inf ? kInf : 2.0;
        return wrap(torch::linalg_vector_norm(t, p, std::vector<int64_t>{axis}, keepdims));
    }

    // Fused: CORA's sums of the smaller and larger products, by two products with M+ and M-
    std::pair<Ptr, Ptr> interval_matmul(const Impl &sup, const Impl &M) const override {
        const torch::Tensor mp = of(M).clamp_min(0), mn = of(M).clamp_max(0);
        return {wrap(t.matmul(mp) + of(sup).matmul(mn)), wrap(of(sup).matmul(mp) + t.matmul(mn))};
    }
    std::pair<Ptr, Ptr> matmul_interval(const Impl &inf, const Impl &sup) const override {
        const torch::Tensor mp = t.clamp_min(0), mn = t.clamp_max(0);
        return {wrap(mp.matmul(of(inf)) + mn.matmul(of(sup))),
                wrap(mp.matmul(of(sup)) + mn.matmul(of(inf)))};
    }
};

// Backend -----------------------------------------------------------------------------------------

/// Makes libtorch tensors on one default device.
struct TorchBackend : Tensor::Backend {
    torch::Device device;
    bool custom_backward;
    TorchBackend(torch::Device device, bool custom) : device(device), custom_backward(custom) {}

    std::string name() const override { return "torch"; }
    Tensor::Ptr make(const std::vector<double> &data, const Shape &shape, const std::string &where,
                     DType dtype) const override {
        const torch::Tensor host = torch::from_blob(const_cast<double *>(data.data()),
                                                    std::vector<int64_t>(shape), torch::kDouble);
        const torch::Device target = where.empty() ? device : torch_device(where);
        return std::make_shared<TorchTensor>(host.to(target, aux_scalar(dtype), false, true),
                                             custom_backward);
    }
};

} // namespace


// ===========================================  MAIN  =========================================== //

// Conversion --------------------------------------------------------------------------------------

std::shared_ptr<const Tensor::Backend> torch_backend(const torch::Device &device,
                                                     bool custom_backward) {
    return std::make_shared<TorchBackend>(device, custom_backward);
}

Tensor from_torch(const torch::Tensor &t, bool custom_backward) {
    return Tensor(std::make_shared<TorchTensor>(aux_floating(t), custom_backward));
}

torch::Tensor to_torch(const Tensor &t) {
    if (!is_torch(t)) CORA_FAIL(Error::wrong_value, "to_torch: not a tensor of the torch backend");
    return static_cast<const TorchTensor &>(t.impl()).t;
}

torch::Device torch_device(const std::string &name) {
    if (name == "gpu") return torch::Device("cuda");
    if (name == "cpu" || name == "cuda" || name.rfind("cuda:", 0) == 0) return torch::Device(name);
    CORA_FAIL(Error::wrong_value,
              "CoraTensor: unknown device '" + name + "'; use cpu, gpu, cuda or cuda:<index>");
}

bool is_torch(const Tensor &t) { return t.impl().backend == backend_id::torch; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
