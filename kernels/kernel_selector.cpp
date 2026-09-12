#include "../include/adaptq/kernel.h"
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif
namespace adaptq {
IKernelBackend *create_scalar_backend();
IKernelBackend *create_avx2_backend();
static bool cpu_supports_avx2() {
#if defined(_MSC_VER)
    int regs[4] = {};
    __cpuid(regs, 0);
    if (regs[0] < 7)
        return false;
    __cpuidex(regs, 1, 0);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx)
        return false;
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
IKernelBackend *select_kernel_backend() {
    if (cpu_supports_avx2())
        return create_avx2_backend();
    return create_scalar_backend();
}
} /* namespace adaptq */
