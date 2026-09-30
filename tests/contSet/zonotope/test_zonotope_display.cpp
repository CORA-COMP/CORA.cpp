// test_zonotope_display - zonotope display: the text that describes a zonotope

#include "contSet/zonotope/zonotope.h"
#include "testing.h"

#include <sstream>

using namespace cora;
using test::check;

namespace {

void the_text_names_dimension_center_and_generators(const std::string &b) {
    const Zonotope Z(Tensor({-1.0, 0.0}), Tensor({{0.01, 0.0}, {0.0, 0.01}}));
    const std::string text = Z.display();
    check(text.find("zonotope") == 0, b + ": it starts with the class");
    check(text.find("- dimension: 2") != std::string::npos, b + ": the dimension");
    check(text.find("- center:") != std::string::npos && text.find("-1") != std::string::npos,
          b + ": the center");
    check(text.find("- generators (2):") != std::string::npos &&
              text.find("0.01") != std::string::npos,
          b + ": the generators");
    std::ostringstream out;
    out << Z;
    check(out.str() == text, b + ": the stream operator prints the same");
}

void a_batch_is_described_by_its_shape(const std::string &b) {
    if (b == "eigen") return;  // Eigen has no batches
    const Zonotope Z(Tensor({1.0, 0.0}), Tensor::eye(2));
    const Zonotope batch = Zonotope::stack({Z, Z, Z});
    check(batch.display().find("batch of shape (3, 2, 1)") != std::string::npos,
          b + ": the batch shape is shown");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        the_text_names_dimension_center_and_generators(b);
        a_batch_is_described_by_its_shape(b);
    });
    return test::finish("zonotope display");
}
