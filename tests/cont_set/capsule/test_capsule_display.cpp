// test_capsule_display - Capsule display: the text of a capsule
//
// CORA: contSet/@capsule/display.m
//
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

#include <sstream>

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
        const std::string text = C.display();
        check(text.find("capsule") != std::string::npos &&
                  text.find("dimension: 2") != std::string::npos &&
                  text.find("radius") != std::string::npos && text.find("0.5") != std::string::npos,
              b + ": the text names the class, the dimension and the radius");
        std::ostringstream out;
        out << C;
        check(out.str() == text, b + ": operator<< prints the display");
        const Capsule batch(Tensor::zeros({2, 2, 1}), Tensor::zeros({2, 2, 1}), 0.1);
        check(batch.display().find("batch") != std::string::npos, b + ": a batch shows shapes");
    });
    // the text has no input to refuse; with the checks off it is the same
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    const std::string on = C.display();
    ChecksScope off(false);
    test::check(C.display() == on, "display: the same with the checks off");
    return test::finish("capsule display");
}


// ---------------------------------------  END OF CODE  ---------------------------------------- //
