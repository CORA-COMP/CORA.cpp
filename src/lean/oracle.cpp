// oracle - the dtype of new lean objects and the process that computes them (CORALean)
//
// See also: lean/tensor.h

#include "lean/oracle.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

std::string current = "binary64";

struct Process {
    bool running = false;
    FILE *in = nullptr;   // the oracle's stdout
    FILE *out = nullptr;  // the oracle's stdin
#ifdef _WIN32
    HANDLE handle = nullptr;
#else
    pid_t pid = -1;
#endif
};

// [0] the sound oracle (CORACPP_ORACLE), [1] the bench oracle (CORACPP_ORACLE_BENCH)
Process oracles[2];

const char *aux_command(int which) {
    const char *name = which ? "CORACPP_ORACLE_BENCH" : "CORACPP_ORACLE";
    const char *command = std::getenv(name);
    if (!command || !*command)
        throw std::runtime_error(std::string("lean: set ") + name + " to the command that starts "
                                 "the CORALean " + (which ? "bench " : "") + "oracle, e.g. "
                                 "'cd CORALean && lake exe oracle'");
    return command;
}

// The bench oracle serves the dtypes without a soundness guarantee.
int aux_which(const std::string &dtype) { return dtype == "float" || dtype == "nearest"; }

#ifdef _WIN32

// Starts `cmd /c <command>` with its stdin and stdout on two pipes.
void aux_start(Process &oracle, int which) {
    const std::string command = std::string("cmd /s /c \"") + aux_command(which) + "\"";
    SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE childIn, parentOut, parentIn, childOut;
    if (!CreatePipe(&childIn, &parentOut, &inheritable, 0) ||
        !CreatePipe(&parentIn, &childOut, &inheritable, 0))
        throw std::runtime_error("lean: cannot create pipes");
    // the parent's ends must not be inherited by the child
    SetHandleInformation(parentOut, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(parentIn, HANDLE_FLAG_INHERIT, 0);
    // the child reads and writes the pipes instead of a console
    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = childIn;
    startup.hStdOutput = childOut;
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    // CreateProcess edits the command line in place, so it gets a mutable copy
    PROCESS_INFORMATION info{};
    std::string mutableCommand = command;
    if (!CreateProcessA(nullptr, mutableCommand.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr,
                        &startup, &info))
        throw std::runtime_error("lean: cannot start the oracle process: " + command);
    CloseHandle(childIn);
    CloseHandle(childOut);
    CloseHandle(info.hThread);
    // the parent talks to the child through C streams on its ends of the pipes
    oracle.handle = info.hProcess;
    oracle.out = _fdopen(_open_osfhandle(reinterpret_cast<intptr_t>(parentOut), _O_BINARY), "wb");
    oracle.in = _fdopen(_open_osfhandle(reinterpret_cast<intptr_t>(parentIn),
                                        _O_RDONLY | _O_BINARY), "rb");
    oracle.running = true;
}

void aux_wait(Process &oracle) {
    WaitForSingleObject(oracle.handle, INFINITE);
    CloseHandle(oracle.handle);
}

#else

// Starts `sh -c <command>` with its stdin and stdout on two pipes.
void aux_start(Process &oracle, int which) {
    const char *command = aux_command(which);
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
    oracle.running = true;
}

void aux_wait(Process &oracle) { waitpid(oracle.pid, nullptr, 0); }

#endif

std::string aux_readLine(Process &oracle) {
    std::string line;
    int ch;
    // a response is one line
    while ((ch = fgetc(oracle.in)) != EOF && ch != '\n') line += static_cast<char>(ch);
    if (ch == EOF && line.empty())
        throw std::runtime_error("lean: the oracle process ended without answering");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Dtype -----------------------------------------------------------------------------------

void setDType(const std::string &spec) {
    const std::string kind = spec.substr(0, spec.find(':'));
    const bool hasArg = spec.find(':') != std::string::npos;
    const bool known = (spec == "binary64") || spec == "float" || spec == "nearest" ||
                       (kind == "ieee" && hasArg) ||
                       (kind == "dyadic" && hasArg) || (kind == "fixedpoint" && hasArg);
    if (!known)
        throw std::invalid_argument("lean: unknown dtype '" + spec + "'; use 'binary64', "
                                    "'ieee:<format>', 'dyadic:<p>', 'fixedpoint:<f>', 'float' or "
                                    "'nearest'");
    current = spec;
}

std::string dtype() { return current; }

// Calls ------------------------------------------------------------------------------------

// Failures of the oracle surface as exceptions with its message.
Json call(const Json &request) {
    const Json *dtype = request.find("dtype");
    const int which = dtype ? aux_which(dtype->text) : 0;
    Process &oracle = oracles[which];
    if (!oracle.running) aux_start(oracle, which);
    fputs((request.dump() + "\n").c_str(), oracle.out);
    fflush(oracle.out);
    // one line per request, one per response
    const Json response = Json::parse(aux_readLine(oracle));
    const Json *ok = response.find("ok");
    if (!ok || !ok->boolean) {
        const Json *error = response.find("error");
        const std::string message = error ? error->text : "no message";
        throw std::runtime_error("lean: the oracle failed: " + message);
    }
    return response;
}

void shutdown() {
    for (Process &oracle : oracles) {
        if (!oracle.running) continue;
        fclose(oracle.out);
        fclose(oracle.in);
        aux_wait(oracle);
        oracle = Process{};
    }
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
