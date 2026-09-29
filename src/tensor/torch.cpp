// torch - the libtorch backend of Tensor: batched tensors on the CPU or a CUDA device
//
// TorchTensor implements the operations of Tensor::Impl on torch::Tensor, so autograd
// differentiates through them; TorchBackend makes tensors from host numbers.

#include "tensor/torch.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

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

/// A torch::Tensor; customBackward selects the hand-written backward pass of expm and is passed on
/// to every tensor made from this one.
struct TorchTensor : Tensor::Impl {
    torch::Tensor t;
    bool customBackward;
    TorchTensor(torch::Tensor t, bool custom) : t(std::move(t)), customBackward(custom) {}

    Ptr wrap(torch::Tensor r) const {
        return std::make_shared<TorchTensor>(std::move(r), customBackward);
    }
    static const torch::Tensor &of(const Impl &o) { return static_cast<const TorchTensor &>(o).t; }

    std::vector<int64_t> shape() const override { return t.sizes().vec(); }
    std::vector<double> data() const override {
        const torch::Tensor host = t.detach().to(torch::kCPU, torch::kDouble).contiguous();
        return std::vector<double>(host.data_ptr<double>(), host.data_ptr<double>() + host.numel());
    }
    std::string device() const override { return t.device().str(); }
    Ptr to(const std::string &device) const override { return wrap(t.to(parseDevice(device))); }

    Ptr like(const std::vector<double> &data, const std::vector<int64_t> &shape) const override {
        const torch::Tensor host =
            torch::from_blob(const_cast<double *>(data.data()), shape, torch::kDouble).clone();
        return wrap(host.to(t.device()));
    }

    /// The device of a name, as torchDevice reads it.
    static torch::Device parseDevice(const std::string &name) { return torchDevice(name); }

    Ptr add(const Impl &o) const override { return wrap(t + of(o)); }
    Ptr sub(const Impl &o) const override { return wrap(t - of(o)); }
    Ptr scale(double s) const override { return wrap(t * s); }
    Ptr matmul(const Impl &o) const override { return wrap(t.matmul(of(o))); }
    Ptr transpose() const override { return wrap(t.transpose(-2, -1)); }
    Ptr abs() const override { return wrap(t.abs()); }
    Ptr pos() const override { return wrap(t.clamp_min(0)); }
    Ptr neg() const override { return wrap(t.clamp_max(0)); }
    Ptr expm() const override {
        return wrap(customBackward ? ExpmFunction::apply(t) : torch::matrix_exp(t));
    }
    Ptr sumLast() const override { return wrap(t.sum(-1, /*keepdim=*/true)); }
    Ptr diag() const override { return wrap(torch::diag_embed(t.squeeze(-1))); }
    // Broadcast to the batch, so results of a batched system all have the same shape.
    Ptr eyeLike() const override {
        return wrap(torch::eye(t.size(-1), t.options()).expand_as(t).contiguous());
    }
    Ptr zerosLike() const override { return wrap(torch::zeros_like(t)); }

    Ptr catLast(const std::vector<const Impl *> &rest) const override {
        std::vector<torch::Tensor> all{t};
        for (const Impl *r : rest) all.push_back(of(*r));
        return wrap(torch::cat(all, -1));
    }

    Ptr stack(const std::vector<const Impl *> &rest) const override {
        std::vector<torch::Tensor> all{t};
        for (const Impl *r : rest) all.push_back(of(*r));
        return wrap(torch::stack(all, 0));
    }
};

/// Makes libtorch tensors on one default device.
struct TorchBackend : Tensor::Backend {
    torch::Device device;
    bool customBackward;
    TorchBackend(torch::Device device, bool custom) : device(device), customBackward(custom) {}

    std::string name() const override { return "torch"; }
    Tensor::Impl::Ptr make(const std::vector<double> &data, const std::vector<int64_t> &shape,
                           const std::string &where) const override {
        const torch::Tensor host =
            torch::from_blob(const_cast<double *>(data.data()), shape, torch::kDouble).clone();
        const torch::Device target = where.empty() ? device : TorchTensor::parseDevice(where);
        return std::make_shared<TorchTensor>(host.to(target), customBackward);
    }
};

} // namespace

// ===========================================  MAIN  =========================================== //

std::shared_ptr<const Tensor::Backend> torchBackend(const torch::Device &device,
                                                     bool customBackward) {
    return std::make_shared<TorchBackend>(device, customBackward);
}

Tensor fromTorch(const torch::Tensor &t, bool customBackward) {
    return Tensor(std::make_shared<TorchTensor>(t, customBackward));
}

torch::Tensor toTorch(const Tensor &t) { return static_cast<const TorchTensor &>(t.impl()).t; }

torch::Device torchDevice(const std::string &name) {
    if (name == "gpu") return torch::Device("cuda");
    if (name == "cpu" || name == "cuda" || name.rfind("cuda:", 0) == 0) return torch::Device(name);
    throw std::invalid_argument("CoraTensor: unknown device '" + name +
                                "'; use cpu, gpu, cuda or cuda:<index>");
}

bool isTorch(const Tensor &t) { return dynamic_cast<const TorchTensor *>(&t.impl()) != nullptr; }

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
