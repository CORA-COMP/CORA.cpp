// The libtorch backend: one instance, on the CPU or a CUDA device.
//
// The same operations as the Eigen backend, through tensors instead. What it buys is the
// GPU and autograd; what it costs is a dispatch per call, which is why the CPU default
// stays with Eigen — see backend.h.
//
// Measurement runs under `NoGradGuard`: recording a graph is what makes the operations
// differentiable, not what the catalog asks them to time.

#include "backend.h"

#include "torch_sets.h"

#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace cora::torch_backend {
namespace {

using tb::Interval;
using tb::Zonotope;

/// The element type of every tensor. `double` to compare like for like with CORA's
/// MATLAB doubles; `CORACPP_DTYPE=float32` shows what a GPU can do instead.
torch::ScalarType dtype() {
    const char *want = std::getenv("CORACPP_DTYPE");
    if (want != nullptr && std::string(want) == "float32") return torch::kFloat;
    return torch::kDouble;
}

torch::Device device_of(const std::string &name) {
    return name == "gpu" ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
}

/// A set of either representation, so an operation is written once.
struct Set {
    Interval box;
    Zonotope zono;
    bool is_box = false;

    static Set random(const Params &in, const torch::TensorOptions &opts) {
        Set s;
        s.is_box = in.is_interval();
        const std::vector<int64_t> batch =
            in.batch > 1 ? std::vector<int64_t>{in.batch} : std::vector<int64_t>{};
        if (s.is_box) {
            s.box = Interval::generate_random(batch, in.n, opts);
        } else {
            s.zono = Zonotope::generate_random(batch, in.n, in.m, opts);
        }
        return s;
    }

    torch::Tensor rand_point(int64_t points) const {
        return is_box ? box.rand_point(points) : zono.rand_point(points);
    }

    torch::Tensor first() const { return is_box ? box.lo.flatten()[0] : zono.c.flatten()[0]; }
};

/// One instance's inputs, generated on the device, and the operation to repeat.
class TorchRunner : public Runner {
  public:
    TorchRunner(const Params &in, const torch::TensorOptions &opts)
        : in_(in), opts_(opts), device_(opts.device()) {
        if (in_.operation == "startup" || in_.operation == "generateRandom") return;
        s_ = Set::random(in_, opts_);
        const std::vector<int64_t> batch =
            in_.batch > 1 ? std::vector<int64_t>{in_.batch} : std::vector<int64_t>{};
        if (in_.operation == "minkSum") {
            other_ = Set::random(in_, opts_);
        } else if (in_.operation == "randPoint") {
            if (in_.kind != "standard") throw std::runtime_error("randPoint type " + in_.kind);
        } else if (in_.operation == "supportFunc") {
            if (in_.kind != "upper") throw std::runtime_error("supportFunc type " + in_.kind);
            std::vector<int64_t> shape = batch;
            shape.push_back(in_.n);
            const torch::Tensor d = torch::randn(shape, opts_);
            direction_ = d / d.norm(2, {-1}, true);
        } else if (in_.operation == "matMul") {
            matrix_ = torch::randn({in_.n, in_.n}, opts_);
        } else if (in_.operation == "contains") {
            points_ = s_.rand_point(in_.points);
        }
    }

    Mask run() override {
        const torch::NoGradGuard no_grad;
        const std::string &op = in_.operation;
        if (op == "startup") {
            sink(Zonotope::origin(in_.n, opts_).g.sum());
        } else if (op == "generateRandom") {
            sink(Set::random(in_, opts_).first());
        } else if (op == "randPoint") {
            sink(s_.rand_point(in_.points).flatten()[0]);
        } else if (op == "supportFunc") {
            sink(s_.is_box ? s_.box.support_func(direction_).flatten()[0]
                           : s_.zono.support_func(direction_).flatten()[0]);
        } else if (op == "matMul") {
            sink(s_.is_box ? s_.box.mat_mul(matrix_).lo.flatten()[0]
                           : s_.zono.mat_mul(matrix_).c.flatten()[0]);
        } else if (op == "minkSum") {
            sink(s_.is_box ? s_.box.mink_sum(other_.box).lo.flatten()[0]
                           : s_.zono.mink_sum(other_.zono).c.flatten()[0]);
        } else if (op == "contains") {
            answers_ = s_.is_box ? s_.box.contains(points_) : s_.zono.contains(points_);
            return read_back(answers_);
        }
        return {};
    }

    /// A CUDA call only queues work, so a measurement that did not wait would time the
    /// queueing. Reading a containment answer back already waits.
    void sync() override {
        if (device_.is_cuda()) torch::cuda::synchronize();
    }

  private:
    /// Keeps a result from being dropped: the operation is what is measured, and nothing
    /// downstream looks at what it produced. Kept on the device — copying it to the host
    /// would be a synchronization the catalog is not asking to time.
    void sink(const torch::Tensor &v) { kept_ = v; }

    static Mask read_back(const torch::Tensor &mask) {
        const torch::Tensor flat =
            mask.reshape({-1}).to(torch::kCPU).to(torch::kUInt8).contiguous();
        const auto *data = flat.data_ptr<std::uint8_t>();
        return Mask(data, data + flat.numel());
    }

    const Params &in_;
    torch::TensorOptions opts_;
    torch::Device device_;
    Set s_, other_;
    torch::Tensor matrix_, direction_, points_, answers_, kept_;
};

} // namespace

bool built() { return true; }

bool supports(const std::string &device) {
    if (device == "cpu") return true;
    return device == "gpu" && torch::cuda::is_available();
}

std::string describe() {
    std::ostringstream out;
    out << "libtorch " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "."
        << TORCH_VERSION_PATCH << ", element type: "
        << (dtype() == torch::kDouble ? "double" : "float");
    if (torch::cuda::is_available()) {
        out << ", cuda: " << static_cast<int>(torch::cuda::device_count()) << " device(s)";
    } else {
        out << ", cuda: none — gpu instances report unsupported";
    }
    return out.str();
}

std::unique_ptr<Runner> prepare(const Params &in) {
    const auto opts =
        torch::TensorOptions().dtype(dtype()).device(device_of(in.device));
    torch::manual_seed(kSeed);
    return std::make_unique<TorchRunner>(in, opts);
}

void warm_up() {
    std::vector<std::string> devices{"cpu"};
    if (torch::cuda::is_available()) devices.emplace_back("gpu");
    for (const std::string &device : devices) {
        for (const char *set : kRepresentations) {
            for (const char *op : kOperations) {
                for (int batch : {1, 2}) {
                    const std::string params =
                        std::string("{\"set\": \"") + set + "\", \"operation\": \"" + op
                        + "\", \"dim\": 3, \"generators\": 6, \"device\": \"" + device
                        + "\", \"repetition\": 1, \"points\": 2, \"batch_size\": "
                        + std::to_string(batch) + ", \"type\": \""
                        + (std::string(op) == "supportFunc" ? "upper" : "standard") + "\"}";
                    const Params in(params);
                    prepare(in)->run();
                }
            }
        }
        check_gradients(device);
    }
}

void check_gradients(const std::string &device) {
    const auto opts = torch::TensorOptions().dtype(torch::kDouble).device(device_of(device));
    // d/dc of `c'd + ||G'd||_1` is `d`, and d/dG is `d (sign(G'd))'`.
    const torch::Tensor d = torch::randn({4}, opts);
    Zonotope z{torch::randn({4}, opts.requires_grad(true)),
               torch::randn({4, 7}, opts.requires_grad(true))};
    z.support_func(d).backward();

    const torch::Tensor want_g =
        d.unsqueeze(-1) * d.unsqueeze(0).matmul(z.g).sign();
    const double err_c = (z.c.grad() - d).abs().max().item<double>();
    const double err_g = (z.g.grad() - want_g).abs().max().item<double>();
    if (!(err_c < 1e-12 && err_g < 1e-12))
        throw std::runtime_error("libtorch on " + device
                                 + ": the gradient of supportFunc is wrong");
}

} // namespace cora::torch_backend
