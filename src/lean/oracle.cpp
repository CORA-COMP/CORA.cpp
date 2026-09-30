// oracle - the dtype of new lean objects and the process that computes them (CORALean)
//
// See also: lean/tensor.h

#include "lean/oracle.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

std::string current = "binary64";

struct Process {
    pid_t pid = -1;
    FILE *in = nullptr;   // the oracle's stdout
    FILE *out = nullptr;  // the oracle's stdin
};

Process oracle;

void aux_start() {
    const char *command = std::getenv("CORACPP_ORACLE");
    if (!command || !*command)
        throw std::runtime_error("lean: set CORACPP_ORACLE to the command that starts the "
                                 "CORALean oracle, e.g. 'cd CORALean && lake exe oracle'");
    int toChild[2], fromChild[2];
    if (pipe(toChild) || pipe(fromChild)) throw std::runtime_error("lean: cannot create pipes");
    oracle.pid = fork();
    if (oracle.pid < 0) throw std::runtime_error("lean: cannot start the oracle process");
    if (oracle.pid == 0) {
        dup2(toChild[0], 0);
        dup2(fromChild[1], 1);
        for (const int fd : {toChild[0], toChild[1], fromChild[0], fromChild[1]}) close(fd);
        execl("/bin/sh", "sh", "-c", command, static_cast<char *>(nullptr));
        _exit(127);
    }
    // the parent keeps the other ends
    close(toChild[0]);
    close(fromChild[1]);
    oracle.out = fdopen(toChild[1], "w");
    oracle.in = fdopen(fromChild[0], "r");
}

std::string aux_readLine() {
    std::string line;
    int ch;
    // a response is one line
    while ((ch = fgetc(oracle.in)) != EOF && ch != '\n') line += static_cast<char>(ch);
    if (ch == EOF && line.empty())
        throw std::runtime_error("lean: the oracle process ended without answering");
    return line;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Dtype -----------------------------------------------------------------------------------

void setDType(const std::string &spec) {
    const std::string kind = spec.substr(0, spec.find(':'));
    const bool hasArg = spec.find(':') != std::string::npos;
    const bool known = (spec == "binary64") || (kind == "ieee" && hasArg) ||
                       (kind == "dyadic" && hasArg) || (kind == "fixedpoint" && hasArg);
    if (!known)
        throw std::invalid_argument("lean: unknown dtype '" + spec + "'; use 'binary64', "
                                    "'ieee:<format>', 'dyadic:<p>' or 'fixedpoint:<f>'");
    current = spec;
}

std::string dtype() { return current; }

// Calls ------------------------------------------------------------------------------------

// Failures of the oracle surface as exceptions with its message.
Json call(const Json &request) {
    if (oracle.pid < 0) aux_start();
    fputs((request.dump() + "\n").c_str(), oracle.out);
    fflush(oracle.out);
    // one line per request, one per response
    const Json response = Json::parse(aux_readLine());
    const Json *ok = response.find("ok");
    if (!ok || !ok->boolean) {
        const Json *error = response.find("error");
        const std::string message = error ? error->text : "no message";
        throw std::runtime_error("lean: the oracle failed: " + message);
    }
    return response;
}

void shutdown() {
    if (oracle.pid < 0) return;
    fclose(oracle.out);
    fclose(oracle.in);
    waitpid(oracle.pid, nullptr, 0);
    oracle = Process{};
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
