#include "tensor/torch.h"

namespace cora::ct {
namespace {

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

struct TorchTensor : Tensor::Impl {
    torch::Tensor t;
    bool custom_backward;
    TorchTensor(torch::Tensor t, bool custom) : t(std::move(t)), custom_backward(custom) {}

    Ptr wrap(torch::Tensor r) const {
        return std::make_shared<TorchTensor>(std::move(r), custom_backward);
    }
    static const torch::Tensor &of(const Impl &o) { return static_cast<const TorchTensor &>(o).t; }

    std::vector<int64_t> shape() const override { return t.sizes().vec(); }
    std::vector<double> data() const override {
        const torch::Tensor host = t.detach().to(torch::kCPU, torch::kDouble).contiguous();
        return std::vector<double>(host.data_ptr<double>(), host.data_ptr<double>() + host.numel());
    }

    Ptr add(const Impl &o) const override { return wrap(t + of(o)); }
    Ptr sub(const Impl &o) const override { return wrap(t - of(o)); }
    Ptr scale(double s) const override { return wrap(t * s); }
    Ptr matmul(const Impl &o) const override { return wrap(t.matmul(of(o))); }
    Ptr transpose() const override { return wrap(t.transpose(-2, -1)); }
    Ptr abs() const override { return wrap(t.abs()); }
    Ptr pos() const override { return wrap(t.clamp_min(0)); }
    Ptr neg() const override { return wrap(t.clamp_max(0)); }
    Ptr expm() const override {
        return wrap(custom_backward ? ExpmFunction::apply(t) : torch::matrix_exp(t));
    }
    Ptr sum_last() const override { return wrap(t.sum(-1, /*keepdim=*/true)); }
    Ptr diag() const override { return wrap(torch::diag_embed(t.squeeze(-1))); }
    // Broadcast to the batch, so results of a batched system all have the same shape.
    Ptr eye_like() const override {
        return wrap(torch::eye(t.size(-1), t.options()).expand_as(t).contiguous());
    }
    Ptr zeros_like() const override { return wrap(torch::zeros_like(t)); }

    Ptr cat_last(const std::vector<const Impl *> &rest) const override {
        std::vector<torch::Tensor> all{t};
        for (const Impl *r : rest) all.push_back(of(*r));
        return wrap(torch::cat(all, -1));
    }
};

struct TorchBackend : Tensor::Backend {
    torch::Device device;
    bool custom_backward;
    TorchBackend(torch::Device device, bool custom) : device(device), custom_backward(custom) {}

    std::string name() const override { return "torch"; }
    Tensor::Impl::Ptr make(const std::vector<double> &data,
                           const std::vector<int64_t> &shape) const override {
        const torch::Tensor host =
            torch::from_blob(const_cast<double *>(data.data()), shape, torch::kDouble).clone();
        return std::make_shared<TorchTensor>(host.to(device), custom_backward);
    }
};

} // namespace

std::shared_ptr<const Tensor::Backend> torch_backend(const torch::Device &device,
                                                     bool custom_backward) {
    return std::make_shared<TorchBackend>(device, custom_backward);
}

Tensor from_torch(const torch::Tensor &t, bool custom_backward) {
    return Tensor(std::make_shared<TorchTensor>(t, custom_backward));
}

torch::Tensor to_torch(const Tensor &t) { return static_cast<const TorchTensor &>(t.impl()).t; }

} // namespace cora::ct
