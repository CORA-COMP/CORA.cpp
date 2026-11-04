// test_capsule_is_bounded - Capsule is_bounded: always true
//
// CORA: contSet/@capsule/isBounded.m

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::column;

int main() {
    test::for_each_backend([](const std::string &b) {
        check(Capsule(column({1.0, 2.0}), column({3.0, 4.0}), 0.5).is_bounded(),
              b + ": a capsule is bounded");
        check(Capsule::empty(2).is_bounded(), b + ": so is the empty one");
        const Set S = Capsule::origin(2);
        check(S->is_bounded(), b + ": through a Set");
    });
    // the question has no input to refuse; with the checks off it gives the same answer
    ChecksScope off(false);
    check(Capsule::origin(2).is_bounded(), "is_bounded: the same with the checks off");
    return test::finish("capsule is_bounded");
}
