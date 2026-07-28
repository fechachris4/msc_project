// Prints the effective configuration exactly as runtime_config.print_effective_config
// does, so the two can be diffed byte-for-byte. Contract clauses I1-I4.

#include <cstdio>
#include <exception>
#include <string>

#include "config/RuntimeConfig.h"

int main(int argc, char** argv) {
  try {
    const std::string path =
        argc > 1 ? argv[1] : srl::config::DefaultConfigPath();
    srl::config::PrintEffectiveConfig(srl::config::LoadConfig(path));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "config error: %s\n", error.what());
    return 1;
  }
  return 0;
}
