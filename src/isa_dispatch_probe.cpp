/**
 * ISA Dispatch Probe for oneDAL
 * Tests three codepath tiers in one binary:
 *
 *   Tier 1 — SSE2   (baseline, all x86_64 CPUs since 2003)
 *   Tier 2 — SSE4.2 (Nehalem+, 2008)
 *   Tier 3 — AVX2   (Haswell+, 2013) + FMA + BMI
 *
 * Each tier: CPUID check → OS/XSAVE check → runtime execution test
 * Final output: clear per-tier PASS/FAIL/SKIP + dispatch recommendation
 *
 * Build:
 *   Linux/macOS:  g++ -O2 -msse4.2 -mavx2 -mfma -mbmi -mbmi2 -o isa_probe src/isa_dispatch_probe.cpp -lm
 *   MSVC:         cl /O2 /arch:AVX2 /EHsc src/isa_dispatch_probe.cpp
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// ─── Platform ────────────────────────────────────────────────────────────────
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__)
  #define ARCH_ARM 1
#else
  #define ARCH_X86 1
  #include <immintrin.h>
  #include <nmmintrin.h>   // SSE4.2
  #include <emmintrin.h>   // SSE2
  #ifdef _WIN32
    #include <intrin.h>
    #define CPUID(info, leaf)        __cpuid((int*)(info), (int)(leaf))
    #define CPUIDEX(info, leaf, sub) __cpuidex((int*)(info), (int)(leaf), (int)(sub))
    static inline uint64_t xgetbv0() { return _xgetbv(0); }
  #else
    #include <cpuid.h>
    #define CPUID(info, leaf)        __cpuid(leaf, info[0], info[1], info[2], info[3])
    #define CPUIDEX(info, leaf, sub) __cpuid_count(leaf, sub, info[0], info[1], info[2], info[3])
    static inline uint64_t xgetbv0() {
        uint64_t v; __asm__ volatile("xgetbv":"=A"(v):"c"(0)); return v;
    }
  #endif
#endif

// ─── Result tracking ────────────────────────────────────────────────────────
static int g_pass = 0, g_fail = 0;
static void report(const char* name, int ok) {
    printf("    %-42s %s\n", name, ok ? "PASS" : "FAIL");
    if (ok) g_pass++; else g_fail++;
}
static void section(const char* name) { printf("  [%s]\n", name); }

#ifdef ARCH_X86

// ─── CPU info ────────────────────────────────────────────────────────────────
static void get_cpu_brand(char* out) {
    int info[4];
    CPUID(info, 0x80000000);
    if ((unsigned)info[0] < 0x80000004) { strcpy(out, "Unknown"); return; }
    CPUID(info, 0x80000002); memcpy(out,    info, 16);
    CPUID(info, 0x80000003); memcpy(out+16, info, 16);
    CPUID(info, 0x80000004); memcpy(out+32, info, 16);
    out[48] = '\0';
}

struct Features {
    int sse2, sse41, sse42, popcnt;
    int avx, avx2, fma, bmi1, bmi2;
    int avx512f;
    int osxsave, os_ymm;
};

static Features detect() {
    Features f = {};
    int info[4];
    CPUID(info, 1);
    f.sse2    = (info[3] >> 26) & 1;
    f.sse41   = (info[2] >> 19) & 1;
    f.sse42   = (info[2] >> 20) & 1;
    f.popcnt  = (info[2] >> 23) & 1;
    f.avx     = (info[2] >> 28) & 1;
    f.fma     = (info[2] >> 12) & 1;
    f.osxsave = (info[2] >> 27) & 1;
    CPUIDEX(info, 7, 0);
    f.avx2    = (info[1] >>  5) & 1;
    f.bmi1    = (info[1] >>  3) & 1;
    f.bmi2    = (info[1] >>  8) & 1;
    f.avx512f = (info[1] >> 16) & 1;
    if (f.osxsave) f.os_ymm = (xgetbv0() & 0x6) == 0x6;
    return f;
}

// ═══════════════════════════════════════════════════════════════════════════
// TIER 1 — SSE2
// ═══════════════════════════════════════════════════════════════════════════
static int run_sse2_tests() {
    int saved_pass = g_pass, saved_fail = g_fail;

    // 128-bit int add (epi32)
    {
        int32_t a[4]={1,2,3,4}, b[4]={4,3,2,1}, c[4]={};
        __m128i r = _mm_add_epi32(_mm_loadu_si128((__m128i*)a),
                                  _mm_loadu_si128((__m128i*)b));
        _mm_storeu_si128((__m128i*)c, r);
        int ok=1; for(int i=0;i<4;i++) ok &= (c[i]==5);
        report("_mm_add_epi32", ok);
    }
    // 128-bit float add
    {
        float a[4]={1,2,3,4}, b[4]={1,1,1,1}, c[4]={};
        __m128 r = _mm_add_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
        _mm_storeu_ps(c, r);
        int ok=1; for(int i=0;i<4;i++) ok &= (c[i]==(float)(i+2));
        report("_mm_add_ps", ok);
    }
    // 128-bit double add
    {
        double a[2]={1.0,2.0}, b[2]={0.5,0.5}, c[2]={};
        __m128d r = _mm_add_pd(_mm_loadu_pd(a), _mm_loadu_pd(b));
        _mm_storeu_pd(c, r);
        int ok = (fabs(c[0]-1.5)<1e-12 && fabs(c[1]-2.5)<1e-12);
        report("_mm_add_pd", ok);
    }
    // float mul
    {
        float a[4]={2,4,6,8}, b[4]={2,2,2,2}, c[4]={};
        __m128 r = _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
        _mm_storeu_ps(c, r);
        int ok=1; for(int i=0;i<4;i++) ok &= (c[i]==(float)(i+1)*4);
        report("_mm_mul_ps", ok);
    }
    // shuffle epi32
    {
        int32_t a[4]={1,2,3,4}, c[4]={};
        __m128i r = _mm_shuffle_epi32(_mm_loadu_si128((__m128i*)a), 0x1b);
        _mm_storeu_si128((__m128i*)c, r);
        int ok = (c[0]==4 && c[1]==3 && c[2]==2 && c[3]==1);
        report("_mm_shuffle_epi32 (reverse)", ok);
    }
    // compare epi32
    {
        int32_t a[4]={1,2,3,4}, b[4]={1,0,3,0}, c[4]={};
        __m128i r = _mm_cmpeq_epi32(_mm_loadu_si128((__m128i*)a),
                                    _mm_loadu_si128((__m128i*)b));
        _mm_storeu_si128((__m128i*)c, r);
        int ok = (c[0]==-1 && c[1]==0 && c[2]==-1 && c[3]==0);
        report("_mm_cmpeq_epi32", ok);
    }
    // movemask
    {
        float a[4]={-1.0f, 1.0f, -1.0f, 1.0f};
        int mask = _mm_movemask_ps(_mm_loadu_ps(a));
        report("_mm_movemask_ps", mask == 0b0101);
    }
    // sqrt
    {
        float a[4]={1,4,9,16}, c[4]={};
        __m128 r = _mm_sqrt_ps(_mm_loadu_ps(a));
        _mm_storeu_ps(c, r);
        int ok=1; for(int i=0;i<4;i++) ok &= (fabsf(c[i]-(float)(i+1)) < 1e-5f);
        report("_mm_sqrt_ps", ok);
    }

    return (g_fail == saved_fail) ? 1 : 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// TIER 2 — SSE4.2
// ═══════════════════════════════════════════════════════════════════════════
static int run_sse42_tests() {
    int saved_pass = g_pass, saved_fail = g_fail;

    // epi32 max (SSE4.1)
    {
        int32_t a[4]={1,-2,3,-4}, b[4]={-1,2,-3,4}, c[4]={};
        __m128i r = _mm_max_epi32(_mm_loadu_si128((__m128i*)a),
                                  _mm_loadu_si128((__m128i*)b));
        _mm_storeu_si128((__m128i*)c, r);
        int ok=1; for(int i=0;i<4;i++) ok &= (c[i]>0);
        report("_mm_max_epi32 (SSE4.1)", ok);
    }
    // blendv (SSE4.1)
    {
        float a[4]={1,2,3,4}, b[4]={10,20,30,40}, c[4]={};
        float mask[4]={-1,0,-1,0};
        __m128 r = _mm_blendv_ps(_mm_loadu_ps(a), _mm_loadu_ps(b),
                                 _mm_loadu_ps(mask));
        _mm_storeu_ps(c, r);
        int ok = (c[0]==10 && c[1]==2 && c[2]==30 && c[3]==4);
        report("_mm_blendv_ps (SSE4.1)", ok);
    }
    // dp_ps dot product (SSE4.1)
    {
        float a[4]={1,2,3,4}, b[4]={1,2,3,4}, c[4]={};
        __m128 r = _mm_dp_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), 0xFF);
        _mm_storeu_ps(c, r);
        // 1+4+9+16 = 30
        int ok = (fabsf(c[0]-30.0f) < 1e-4f);
        report("_mm_dp_ps dot product (SSE4.1)", ok);
    }
    // epi8 signed compare gt (SSE2 actually, but common in SSE4.2 paths)
    {
        int8_t a[16], b[16], c[16]={};
        for(int i=0;i<16;i++) { a[i]=(int8_t)i; b[i]=7; }
        __m128i r = _mm_cmpgt_epi8(_mm_loadu_si128((__m128i*)a),
                                   _mm_loadu_si128((__m128i*)b));
        _mm_storeu_si128((__m128i*)c, r);
        int ok = (c[7]==0 && c[8]==-1);
        report("_mm_cmpgt_epi8", ok);
    }
    // POPCNT (SSE4.2)
    {
        uint32_t x = 0b10110101;
        uint32_t r = _mm_popcnt_u32(x); // 5 bits set
        report("_mm_popcnt_u32 (POPCNT)", r == 5);
    }
    // CRC32 (SSE4.2)
    {
        uint32_t crc = _mm_crc32_u32(0xFFFFFFFF, 0x12345678);
        // Just verify it runs and returns something non-trivial
        report("_mm_crc32_u32 (SSE4.2)", crc != 0 && crc != 0xFFFFFFFF);
    }
    // string compare (SSE4.2 PCMPISTRI)
    {
        // Find first occurrence of 'l' in "hello"
        const char* haystack = "hello world";
        const char* needle   = "l";
        __m128i vh = _mm_loadu_si128((__m128i*)haystack);
        __m128i vn = _mm_loadu_si128((__m128i*)needle);
        int idx = _mm_cmpistri(vn, vh, _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT);
        report("_mm_cmpistri PCMPISTRI (SSE4.2)", idx == 2); // 'l' at index 2
    }

    return (g_fail == saved_fail) ? 1 : 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// TIER 3 — AVX2 (key subset)
// ═══════════════════════════════════════════════════════════════════════════
static int run_avx2_tests() {
    int saved_pass = g_pass, saved_fail = g_fail;

    // 256-bit int add
    {
        int32_t a[8]={1,2,3,4,5,6,7,8}, b[8]={8,7,6,5,4,3,2,1}, c[8]={};
        __m256i r = _mm256_add_epi32(_mm256_loadu_si256((__m256i*)a),
                                     _mm256_loadu_si256((__m256i*)b));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==9);
        report("_mm256_add_epi32", ok);
    }
    // 256-bit float mul
    {
        float a[8]={1,2,3,4,5,6,7,8}, b[8]={2,2,2,2,2,2,2,2}, c[8]={};
        __m256 r = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
        _mm256_storeu_ps(c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==(float)(i+1)*2);
        report("_mm256_mul_ps", ok);
    }
    // FMA
    {
        float a[8]={1,2,3,4,5,6,7,8}, b[8]={2,2,2,2,2,2,2,2};
        float cc[8]={1,1,1,1,1,1,1,1}, r[8]={};
        __m256 vr = _mm256_fmadd_ps(_mm256_loadu_ps(a),
                                    _mm256_loadu_ps(b),
                                    _mm256_loadu_ps(cc));
        _mm256_storeu_ps(r, vr);
        int ok=1; for(int i=0;i<8;i++) ok &= (r[i]==(float)(i+1)*2+1);
        report("_mm256_fmadd_ps (FMA)", ok);
    }
    // gather
    {
        int32_t base[16]; for(int i=0;i<16;i++) base[i]=i*10;
        int32_t idx[8]={0,2,4,6,8,10,12,14}, c[8]={};
        __m256i r = _mm256_i32gather_epi32(base, _mm256_loadu_si256((__m256i*)idx), 4);
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==idx[i]*10);
        report("_mm256_i32gather_epi32", ok);
    }
    // variable shift (AVX2-only)
    {
        int32_t a[8]={1,1,1,1,1,1,1,1}, sh[8]={0,1,2,3,4,5,6,7}, c[8]={};
        __m256i r = _mm256_sllv_epi32(_mm256_loadu_si256((__m256i*)a),
                                      _mm256_loadu_si256((__m256i*)sh));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==(1<<i));
        report("_mm256_sllv_epi32 (variable shift, AVX2-only)", ok);
    }
    // permute across lanes
    {
        int32_t a[8]={10,20,30,40,50,60,70,80}, c[8]={};
        __m256i r = _mm256_permutevar8x32_epi32(
            _mm256_loadu_si256((__m256i*)a),
            _mm256_set_epi32(0,1,2,3,4,5,6,7));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok = (c[0]==80 && c[7]==10);
        report("_mm256_permutevar8x32_epi32", ok);
    }

    return (g_fail == saved_fail) ? 1 : 0;
}

#endif // ARCH_X86

// ─── main ────────────────────────────────────────────────────────────────────
int main() {
    printf("=================================================================\n");
    printf(" ISA Dispatch Probe — oneDAL codepath investigation\n");
    printf("=================================================================\n");

#ifdef ARCH_ARM
    printf("CPU:  ARM / Apple Silicon\n");
    printf("ARCH: arm64 — x86 ISA tiers not applicable\n\n");
    printf("TIER 1 SSE2:   N/A (ARM)\n");
    printf("TIER 2 SSE4.2: N/A (ARM)\n");
    printf("TIER 3 AVX2:   N/A (ARM)\n\n");
    printf("DISPATCH RECOMMENDATION: Use ARM NEON / SVE codepath\n");
    printf("=================================================================\n");
    return 2;
#else
    char brand[49]={};
    get_cpu_brand(brand);
    Features f = detect();

    printf("CPU:    %s\n", brand);
    printf("CPUID:  SSE2=%-3s SSE4.1=%-3s SSE4.2=%-3s POPCNT=%-3s\n",
           f.sse2?"YES":"NO", f.sse41?"YES":"NO", f.sse42?"YES":"NO", f.popcnt?"YES":"NO");
    printf("        AVX=%-3s  AVX2=%-3s  FMA=%-3s   BMI1=%-3s  BMI2=%-3s\n",
           f.avx?"YES":"NO", f.avx2?"YES":"NO", f.fma?"YES":"NO",
           f.bmi1?"YES":"NO", f.bmi2?"YES":"NO");
    printf("OS/VM:  OSXSAVE=%-3s  YMM_enabled=%-3s\n\n",
           f.osxsave?"YES":"NO", f.os_ymm?"YES":"NO");

    // ── TIER 1: SSE2 ────────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("TIER 1 — SSE2 (baseline, x86_64 all CPUs since 2003)\n");
    if (f.sse2) {
        section("SSE2 tests");
        int ok = run_sse2_tests();
        printf("  => %s\n\n", ok ? "✅ SSE2 PASS" : "❌ SSE2 FAIL");
    } else {
        printf("  SKIPPED — SSE2 not in CPUID (pre-2003 CPU?)\n\n");
    }

    // ── TIER 2: SSE4.2 ──────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("TIER 2 — SSE4.2 (Nehalem+, 2008)\n");
    if (f.sse42) {
        section("SSE4.1 / SSE4.2 tests");
        int ok = run_sse42_tests();
        printf("  => %s\n\n", ok ? "✅ SSE4.2 PASS" : "❌ SSE4.2 FAIL");
    } else {
        printf("  SKIPPED — SSE4.2 not available\n\n");
    }

    // ── TIER 3: AVX2 ────────────────────────────────────────────────────────
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("TIER 3 — AVX2+FMA (Haswell+, 2013)\n");
    if (f.avx2 && f.os_ymm) {
        section("AVX2 + FMA tests");
        int ok = run_avx2_tests();
        printf("  => %s\n\n", ok ? "✅ AVX2 PASS" : "❌ AVX2 FAIL");
    } else if (f.avx2 && !f.os_ymm) {
        printf("  SKIPPED — AVX2 in CPUID but OS/hypervisor disabled YMM state\n");
        printf("  (VM without AVX passthrough — Rosetta 2, old VMware, etc.)\n\n");
    } else {
        printf("  SKIPPED — AVX2 not available\n\n");
    }

    // ── Dispatch recommendation ─────────────────────────────────────────────
    printf("=================================================================\n");
    printf(" DISPATCH RECOMMENDATION\n");
    printf("=================================================================\n");
    if (f.avx2 && f.os_ymm) {
        printf(" ✅ USE AVX2 path  (+ FMA=%s, BMI1=%s, BMI2=%s)\n",
               f.fma?"YES":"NO", f.bmi1?"YES":"NO", f.bmi2?"YES":"NO");
        if (f.avx512f) printf(" 🚀 AVX-512 also available\n");
    } else if (f.sse42) {
        printf(" ⚠️  USE SSE4.2 path (AVX2 unavailable)\n");
        if (f.avx2 && !f.os_ymm)
            printf("     Reason: OS/hypervisor masked YMM state (Rosetta 2 / VM)\n");
    } else if (f.sse2) {
        printf(" ⚠️  USE SSE2 path (SSE4.2 + AVX2 unavailable)\n");
    } else {
        printf(" ❌ NO SIMD — scalar fallback only\n");
    }
    printf("=================================================================\n");

    return g_fail > 0 ? 1 : 0;
#endif
}
