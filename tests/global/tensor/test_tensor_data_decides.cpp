// test_tensor_data_decides - an operation runs where its operands live; mixing places is an error
//
// A tensor keeps its backend, device and dtype whatever scope is current; tensors made like it
// follow it; operands of two backends or dtypes are an error that names both places.

#include "cora/global/tensor/tensor.h"
#include "testing.h"

#include <stdexcept>

using namespace cora;
using test::check;

namespace {

/// The message of the error that f raises, empty if none.
template <class F>
std::string aux_message(F f) {
    try {
        f();
    } catch (const std::exception &e) {
        return e.what();
    }
    return "";
}

void stays_where_it_lives() {
    Tensor a;
    {
        const BackendScope lean("lean");
        a = Tensor({{1.0, 2.0}, {3.0, 4.0}});
    }
    const BackendScope eigen("eigen");
    check(std::string(a.backend_name()) == "lean", "made on lean, it stays lean");
    check(std::string((a + a).backend_name()) == "lean",
          "an operation runs on its operands' backend");
    check(std::string(a.zeros_like({3, 1}).backend_name()) == "lean", "zeros like it are lean");
    check(std::string(a.eye_like().backend_name()) == "lean", "the identity like it is lean");
    check(std::string(Tensor::like(a, {1, 2}, {2, 1}).backend_name()) == "lean",
          "host values like it are lean");
    check(std::string(Tensor::zeros({2, 2}).backend_name()) == "eigen",
          "raw numbers take the current backend");
}

void mixing_is_an_error() {
    Tensor a;
    {
        const BackendScope lean("lean");
        a = Tensor({{1.0}});
    }
    const BackendScope eigen("eigen");
    const Tensor b({{1.0}});
    const std::string message = aux_message([&] { (void)(a + b); });
    check(message.find("lean") != std::string::npos && message.find("eigen") != std::string::npos,
          "the error names both backends: " + message);
    check(!aux_message([&] { (void)a.matmul(b); }).empty(), "matmul of two backends");
    check(!aux_message([&] { (void)Tensor::concat({a, b}, -1); }).empty(),
          "concat of two backends");
    const Tensor f = b.to(DType::float32);
    const std::string dtypes = aux_message([&] { (void)(f + b); });
    check(dtypes.find("float32") != std::string::npos &&
              dtypes.find("float64") != std::string::npos,
          "the error names both dtypes: " + dtypes);
    {
        const ChecksScope off(false);
        check(!aux_message([&] { (void)(a + b); }).empty(), "with the checks off as well");
    }
}

void float32_follows_the_data(const std::string &b) {
    if (b == "lean") return; // lean computes in float64 only
    const Tensor x = Tensor({{1.0, 2.0}, {3.0, 4.0}}).to(DType::float32);
    check(x.dtype() == DType::float32, b + ": to float32");
    const Tensor y = x.matmul(x).add(x.eye_like()).sum(-1, true);
    check(y.dtype() == DType::float32, b + ": operations stay float32");
    check(test::close(y, std::vector<double>{18.0, 38.0}, 1e-6), b + ": float32 values");
    check(x.zeros_like({2}).dtype() == DType::float32, b + ": zeros like it are float32");
    check(x.to(DType::float64).dtype() == DType::float64, b + ": back to float64");
}

} // namespace

int main() {
    stays_where_it_lives();
    mixing_is_an_error();
    std::vector<std::string> all = test::backends();
    all.push_back("lean");
    for (const std::string &b : all) {
        const BackendScope scope(b);
        float32_follows_the_data(b);
    }
    const BackendScope lean("lean");
    check(test::throws([] { (void)Tensor({{1.0}}).to(DType::float32); }), "lean has no float32");
    return test::finish("tensor data decides");
}
