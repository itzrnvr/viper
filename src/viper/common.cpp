// Logging + status helpers.
#include "viper/common.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace viper {

namespace {
std::mutex g_log_mu;

std::string iso_now() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream os;
    os << std::put_time(&tm, "%H:%M:%S") << '.' << std::setw(3) << std::setfill('0')
       << ms;
    return os.str();
}

const char* level_str(LogLevel l) {
    switch (l) {
        case LogLevel::Debug: return "DBG";
        case LogLevel::Info:  return "INF";
        case LogLevel::Warn:  return "WRN";
        case LogLevel::Error: return "ERR";
    }
    return "???";
}
}  // namespace

void log_message(LogLevel level, std::string_view msg) {
    std::lock_guard<std::mutex> g(g_log_mu);
    auto& out = (level >= LogLevel::Warn) ? std::cerr : std::cout;
    out << '[' << iso_now() << "] [" << level_str(level) << "] " << msg << '\n';
    out.flush();
}

}  // namespace viper