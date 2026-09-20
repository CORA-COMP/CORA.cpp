// Commands the per-instance scripts call:
//
//     coracpp serve <server dir>           the warm daemon (see server.cpp)
//     coracpp run <params> <results file>  one instance in this process
//     coracpp env                          what the worker runs on
//     coracpp check                        every operation once, as an install smoke test

#include "instance.h"
#include "server.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    try {
        if (args.size() == 1 && args[0] == "env") {
            cora::print_env(std::cout);
            return 0;
        }
        if (args.size() == 1 && args[0] == "check") {
            cora::warm_up_backends();
            std::cout << "all operations ran\n";
            return 0;
        }
        if (args.size() == 2 && args[0] == "serve") {
            return cora::serve(args[1]);
        }
        if (args.size() == 3 && args[0] == "run") {
            const std::string verdict = cora::run_logged(args[1], args[2], std::cout);
            return verdict == "error" ? 1 : 0;
        }
    } catch (const std::exception &e) {
        std::cerr << "[coracpp] " << e.what() << "\n";
        return 1;
    }
    std::cerr << "usage: coracpp serve <dir> | run <params> <results file> | env | check\n";
    return 2;
}
