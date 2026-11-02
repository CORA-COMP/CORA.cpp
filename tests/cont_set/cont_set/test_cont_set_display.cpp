// test_cont_set_display - ContSet::display: the text of a set, as `std::cout << S` prints it

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// The text names the class; display takes no input to check.
void the_text_names_the_class() {
    const Zonotope Z = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    const ContSet &S = Z;
    check(S.display().find("zonotope") != std::string::npos ||
              S.display().find("Zonotope") != std::string::npos,
          "the class is named");
    ChecksScope off(false);
    check(S.display() == Z.display(), "the checks off");
}

} // namespace

int main() {
    the_text_names_the_class();
    return test::finish("cont_set display");
}
