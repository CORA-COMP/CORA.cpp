// test_tensor_setBackend - setBackend: a backend is chosen once; bad choices are refused

#include "tensor/tensor.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;
using test::throws;

namespace {

void the_backend_is_chosen_once(const std::string &b) {
    check(close(Tensor({1.0}).data()[0], 1.0), b + ": a backend is set");
    check(throws([] { cora::ct::setBackend("jax"); }), b + ": an unknown backend");
    check(throws([] { cora::ct::setBackend("eigen:cuda"); }), b + ": eigen has no device");
    check(throws([] { cora::ct::setBackend("eigen,customBackward"); }),
          b + ": eigen has no options");
    check(throws([] { cora::ct::setBackend("torch,nonsense"); }), b + ": an unknown option");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { the_backend_is_chosen_once(b); });
    return test::finish("tensor setBackend");
}
