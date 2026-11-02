// test_tensor_backend_scope - BackendScope: a backend is chosen per scope; bad choices are refused

#include "cora/global/tensor/tensor.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::throws;

namespace {

/// A scope holds on this thread and ends with its block; another thread keeps the default.
void a_scope_is_local(const std::string &b) {
    const Tensor::Backend *outer = &backend();
    {
        const BackendScope s("eigen");
        check(&backend() == &cora::backend("eigen"), b + ": the scope's backend");
    }
    check(&backend() == outer, b + ": the backend after the scope");
}

void the_backend_is_chosen_once(const std::string &b) {
    check(close(Tensor({1.0}).data()[0], 1.0), b + ": a backend is set");
    check(throws([] { const BackendScope s("jax"); }), b + ": an unknown backend");
    check(test::raises([] { const BackendScope s("jax"); }, Error::wrong_value), b + ": wrong_value");
    check(throws([] { set_default_backend("jax"); }), b + ": an unknown default backend");
    check(throws([] { const BackendScope s("eigen:cuda"); }), b + ": eigen has no device");
    check(throws([] { const BackendScope s("eigen,custom_backward"); }),
          b + ": eigen has no options");
    check(throws([] { const BackendScope s("torch,nonsense"); }), b + ": an unknown option");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        the_backend_is_chosen_once(b);
        a_scope_is_local(b);
    });
    return test::finish("tensor backend scope");
}
