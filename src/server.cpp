// The warm CORA.cpp daemon.
//
// Process start is cheap here, but the thread pool, the first allocations and the page
// faults of a fresh process are not free, and the harness times all of it.
// `prepare_instance.sh` (untimed) therefore starts this daemon once, and
// `run_instance.sh` only sends it the instance over a localhost socket and waits for the
// verdict — bash's `/dev/tcp` needs no extra process for that.
//
// One request per connection, one line each way:
//
//     ping                                     -> pong
//     run <TAB> results file <TAB> params JSON -> the verdict
//
// Requests are served one at a time, so a daemon still busy with an instance the harness
// gave up on does not answer `ping`; `prepare_instance.sh` then replaces it.

#include "server.h"

#include "instance.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cora {
namespace {

std::string read_line(int fd) {
    std::string line;
    char c = 0;
    while (::read(fd, &c, 1) == 1 && c != '\n') line += c;
    return line;
}

} // namespace

int port() {
    if (const char *env = std::getenv("CORACPP_PORT")) return std::atoi(env);
    return 47916;
}

std::string run_logged(const std::string &params, const std::string &results_file,
                       std::ostream &log) {
    try {
        return run_instance(params, results_file, log);
    } catch (const std::exception &e) {
        log << "[coracpp] " << e.what() << "\n";
        write_error(results_file);
        return "error";
    } catch (...) {
        log << "[coracpp] the instance failed with an unknown error\n";
        write_error(results_file);
        return "error";
    }
}

int serve(const std::string &srv_dir) {
    {
        std::ofstream pid(srv_dir + "/server.pid", std::ios::trunc);
        pid << ::getpid() << "\n";
    }
    warm_up();

    const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) throw std::runtime_error("could not open a socket");
    int reuse = 1;
    ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof reuse);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
    address.sin_port = ::htons(static_cast<std::uint16_t>(port()));
    if (::bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof address) < 0)
        throw std::runtime_error("could not bind 127.0.0.1:" + std::to_string(port()));
    if (::listen(listener, 8) < 0) throw std::runtime_error("could not listen");

    std::cout << "[coracpp] serving on 127.0.0.1:" << port() << std::endl;
    const std::string log_path = srv_dir + "/job.log";

    while (true) {
        const int conn = ::accept(listener, nullptr, nullptr);
        if (conn < 0) continue;
        const std::string request = read_line(conn);
        std::string reply;
        if (request == "ping") {
            reply = "pong";
        } else if (request.rfind("run\t", 0) == 0) {
            const std::size_t split = request.find('\t', 4);
            if (split == std::string::npos) {
                reply = "unknown request \"run\"";
            } else {
                // The instance's own output belongs in the job log, not the daemon's; the
                // tool scripts print it when the verdict is not `finished`.
                std::ostringstream log;
                reply = run_logged(request.substr(split + 1), request.substr(4, split - 4), log);
                std::ofstream(log_path, std::ios::trunc) << log.str();
            }
        } else {
            reply = "unknown request \"" + request + "\"";
        }
        reply += "\n";
        const ssize_t written = ::write(conn, reply.data(), reply.size());
        (void)written;
        ::close(conn);
    }
}

} // namespace cora
