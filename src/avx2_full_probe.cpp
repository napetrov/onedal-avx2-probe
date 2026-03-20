/**
 * AVX2 Full Capability Probe for oneDAL
 * Tests all major AVX2 instruction groups:
 *   - Integer arithmetic (epi8/16/32/64)
 *   - Float arithmetic (AVX, FMA)
 *   - Shuffle / permute
 *   - Gather loads
 *   - Bitwise / shift / compare
 *   - Blend / mask
 *   - Type conversions
 *   - BMI1/BMI2
 *
 * Build Linux/macOS:
 *   g++ -O2 -mavx2 -mfma -mbmi -mbmi2 -o avx2_full_probe src/avx2_full_probe.cpp -lm
 *
 * Build Windows (MSVC):
 *   cl /O2 /arch:AVX2 /EHsc src/avx2_full_probe.cpp /Fe:avx2_full_probe.exe
 *
 * Build Windows (MinGW/clang-cl):
 *   g++ -O2 -mavx2 -mfma -mbmi -mbmi2 -o avx2_full_probe.exe src/avx2_full_probe.cpp
 *
 * macOS Apple Silicon (M1/M2/M3):
 *   The probe will detect NO AVX2 (ARM CPU) and exit gracefully.
 *
 * Output: per-bucket PASS/FAIL + final verdict
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// ─── Platform detection ──────────────────────────────────────────────────────
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__)
  // Apple Silicon or ARM — no x86 intrinsics
  #define ARCH_ARM 1
#else
  #define ARCH_X86 1
  #include <immintrin.h>
  #ifdef _WIN32
    #include <intrin.h>
    #define CPUID(info, leaf)        __cpuid((int*)(info), (int)(leaf))
    #define CPUIDEX(info, leaf, sub) __cpuidex((int*)(info), (int)(leaf), (int)(sub))
    static inline uint64_t _xgetbv_val(unsigned idx) { return _xgetbv(idx); }
  #else
    #include <cpuid.h>
    #define CPUID(info, leaf)        __cpuid(leaf, info[0], info[1], info[2], info[3])
    #define CPUIDEX(info, leaf, sub) __cpuid_count(leaf, sub, info[0], info[1], info[2], info[3])
    static inline uint64_t _xgetbv_val(unsigned idx) {
        uint64_t v; __asm__ volatile("xgetbv" : "=A"(v) : "c"(idx)); return v;
    }
  #endif
#endif

// ─── helpers ────────────────────────────────────────────────────────────────
static int g_pass = 0, g_fail = 0;

static void report(const char* name, int ok) {
    printf("  %-40s %s\n", name, ok ? "PASS" : "FAIL");
    if (ok) g_pass++; else g_fail++;
}

static void section(const char* name) {
    printf("\n[%s]\n", name);
}

// ─── CPUID / OS detection ───────────────────────────────────────────────────
#ifdef ARCH_ARM
static void get_cpu_brand(char* out) { strcpy(out, "ARM / Apple Silicon"); }
static int has_avx2=0, has_fma=0, has_avx512f=0, has_avx512bw=0, has_avx512vl=0;
static int has_bmi1=0, has_bmi2=0, os_avx_ok=0;
static void detect_features() {}
#else
static void get_cpu_brand(char* out) {
    int info[4];
    CPUID(info, 0x80000000);
    if ((unsigned)info[0] < 0x80000004) { strcpy(out, "Unknown"); return; }
    CPUID(info, 0x80000002); memcpy(out,    info, 16);
    CPUID(info, 0x80000003); memcpy(out+16, info, 16);
    CPUID(info, 0x80000004); memcpy(out+32, info, 16);
    out[48] = '\0';
}

static int has_avx2, has_fma, has_avx512f, has_avx512bw, has_avx512vl;
static int has_bmi1, has_bmi2;
static int os_avx_ok;

static void detect_features() {
    int info[4];
    CPUID(info, 1);
    int osxsave = (info[2] >> 27) & 1;
    has_fma = (info[2] >> 12) & 1;
    CPUIDEX(info, 7, 0);
    has_avx2     = (info[1] >> 5)  & 1;
    has_bmi1     = (info[1] >> 3)  & 1;
    has_bmi2     = (info[1] >> 8)  & 1;
    has_avx512f  = (info[1] >> 16) & 1;
    has_avx512bw = (info[1] >> 30) & 1;
    has_avx512vl = (info[1] >> 31) & 1;

    os_avx_ok = 0;
    if (osxsave) {
        uint64_t xcr0 = _xgetbv_val(0);
        os_avx_ok = (xcr0 & 0x6) == 0x6;
    }
}
#endif // ARCH_ARM

// ─── Bucket 1: Integer arithmetic ────────────────────────────────────────────
#ifdef ARCH_X86
static void test_integer_arith() {
    section("BUCKET 1: Integer Arithmetic (epi8/16/32/64)");

    // epi32 add
    {
        int32_t a[8]={1,2,3,4,5,6,7,8}, b[8]={8,7,6,5,4,3,2,1}, c[8]={};
        __m256i r = _mm256_add_epi32(_mm256_loadu_si256((__m256i*)a),
                                     _mm256_loadu_si256((__m256i*)b));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==9);
        report("_mm256_add_epi32", ok);
    }
    // epi32 mul (low 32 bits)
    {
        int32_t a[8]={1,2,3,4,5,6,7,8}, b[8]={2,2,2,2,2,2,2,2}, c[8]={};
        __m256i r = _mm256_mullo_epi32(_mm256_loadu_si256((__m256i*)a),
                                       _mm256_loadu_si256((__m256i*)b));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==(i+1)*2);
        report("_mm256_mullo_epi32", ok);
    }
    // epi16 add saturate
    {
        int16_t a[16], b[16], c[16]={};
        for(int i=0;i<16;i++) { a[i]=30000; b[i]=30000; }
        __m256i r = _mm256_adds_epi16(_mm256_loadu_si256((__m256i*)a),
                                      _mm256_loadu_si256((__m256i*)b));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<16;i++) ok &= (c[i]==32767);
        report("_mm256_adds_epi16 (saturate)", ok);
    }
    // epi64 add
    {
        int64_t a[4]={1LL<<40,2,3,4}, b[4]={1,2,3,4}, c[4]={};
        __m256i r = _mm256_add_epi64(_mm256_loadu_si256((__m256i*)a),
                                     _mm256_loadu_si256((__m256i*)b));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1;
        ok &= (c[0] == (1LL<<40)+1);
        ok &= (c[1] == 4); ok &= (c[2] == 6); ok &= (c[3] == 8);
        report("_mm256_add_epi64", ok);
    }
    // epi8 abs
    {
        int8_t a[32], c[32]={};
        for(int i=0;i<32;i++) a[i]=-i;
        __m256i r = _mm256_abs_epi8(_mm256_loadu_si256((__m256i*)a));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<32;i++) ok &= (c[i]==(int8_t)(i < 128 ? i : 128));
        report("_mm256_abs_epi8", ok);
    }
    // epi32 max/min
    {
        int32_t a[8]={1,-2,3,-4,5,-6,7,-8}, b[8]={-1,2,-3,4,-5,6,-7,8}, c[8]={};
        __m256i r = _mm256_max_epi32(_mm256_loadu_si256((__m256i*)a),
                                     _mm256_loadu_si256((__m256i*)b));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]>0);
        report("_mm256_max_epi32", ok);
    }
    // epu32 min
    {
        uint32_t a[8]={1,2,3,4,5,6,7,8}, b[8]={8,7,6,5,4,3,2,1}, c[8]={};
        __m256i r = _mm256_min_epu32(_mm256_loadu_si256((__m256i*)a),
                                     _mm256_loadu_si256((__m256i*)b));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==( a[i]<b[i]?a[i]:b[i] ));
        report("_mm256_min_epu32", ok);
    }
}

// ─── Bucket 2: Float / FMA ────────────────────────────────────────────────────
static void test_float_fma() {
    section("BUCKET 2: Float Arithmetic + FMA");

    // float add
    {
        float a[8]={1,2,3,4,5,6,7,8}, b[8]={1,1,1,1,1,1,1,1}, c[8]={};
        __m256 r = _mm256_add_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
        _mm256_storeu_ps(c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==(float)(i+2));
        report("_mm256_add_ps", ok);
    }
    // float mul
    {
        float a[8]={1,2,3,4,5,6,7,8}, b[8]={2,2,2,2,2,2,2,2}, c[8]={};
        __m256 r = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
        _mm256_storeu_ps(c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==(float)(i+1)*2);
        report("_mm256_mul_ps", ok);
    }
    // float div
    {
        float a[8]={2,4,6,8,10,12,14,16}, b[8]={2,2,2,2,2,2,2,2}, c[8]={};
        __m256 r = _mm256_div_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
        _mm256_storeu_ps(c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==(float)(i+1));
        report("_mm256_div_ps", ok);
    }
    // float sqrt
    {
        float a[8]={1,4,9,16,25,36,49,64}, c[8]={};
        __m256 r = _mm256_sqrt_ps(_mm256_loadu_ps(a));
        _mm256_storeu_ps(c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (fabsf(c[i]-(float)(i+1)) < 1e-5f);
        report("_mm256_sqrt_ps", ok);
    }
    // double add
    {
        double a[4]={1.0,2.0,3.0,4.0}, b[4]={0.5,0.5,0.5,0.5}, c[4]={};
        __m256d r = _mm256_add_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(b));
        _mm256_storeu_pd(c, r);
        int ok=1; for(int i=0;i<4;i++) ok &= (fabs(c[i]-(i+1.5)) < 1e-12);
        report("_mm256_add_pd", ok);
    }
    // FMA: a*b+c
    if (has_fma) {
        float a[8]={1,2,3,4,5,6,7,8}, b[8]={2,2,2,2,2,2,2,2};
        float cc[8]={1,1,1,1,1,1,1,1}, r[8]={};
        __m256 vr = _mm256_fmadd_ps(_mm256_loadu_ps(a),_mm256_loadu_ps(b),_mm256_loadu_ps(cc));
        _mm256_storeu_ps(r, vr);
        int ok=1; for(int i=0;i<8;i++) ok &= (r[i]==(float)(i+1)*2+1);
        report("_mm256_fmadd_ps (FMA)", ok);

        // FMA double
        double da[4]={1,2,3,4}, db[4]={3,3,3,3}, dc[4]={0.5,0.5,0.5,0.5}, dr[4]={};
        __m256d vdr = _mm256_fmadd_pd(_mm256_loadu_pd(da),_mm256_loadu_pd(db),_mm256_loadu_pd(dc));
        _mm256_storeu_pd(dr, vdr);
        int ok2=1; for(int i=0;i<4;i++) ok2 &= (fabs(dr[i]-((i+1)*3+0.5)) < 1e-10);
        report("_mm256_fmadd_pd (FMA)", ok2);
    } else {
        printf("  %-40s SKIPPED (no FMA in CPUID)\n", "FMA tests");
    }
    // reciprocal approx
    {
        float a[8]={1,2,4,8,16,32,64,128}, c[8]={};
        __m256 r = _mm256_rcp_ps(_mm256_loadu_ps(a));
        _mm256_storeu_ps(c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (fabsf(c[i] - 1.0f/a[i]) < 1e-3f);
        report("_mm256_rcp_ps (approx)", ok);
    }
}

// ─── Bucket 3: Shuffle / Permute ─────────────────────────────────────────────
static void test_shuffle_permute() {
    section("BUCKET 3: Shuffle / Permute");

    // permute epi32 across lanes
    {
        int32_t a[8]={10,20,30,40,50,60,70,80}, c[8]={};
        __m256i r = _mm256_permutevar8x32_epi32(
            _mm256_loadu_si256((__m256i*)a),
            _mm256_set_epi32(0,1,2,3,4,5,6,7));  // reverse
        _mm256_storeu_si256((__m256i*)c, r);
        int ok = (c[0]==80 && c[7]==10);
        report("_mm256_permutevar8x32_epi32 (reverse)", ok);
    }
    // permute2x128
    {
        int32_t a[8]={1,2,3,4,5,6,7,8}, b[8]={9,10,11,12,13,14,15,16}, c[8]={};
        __m256i r = _mm256_permute2x128_si256(
            _mm256_loadu_si256((__m256i*)a),
            _mm256_loadu_si256((__m256i*)b), 0x21); // hi(a) | lo(b)
        _mm256_storeu_si256((__m256i*)c, r);
        int ok = (c[0]==5 && c[4]==9);
        report("_mm256_permute2x128_si256", ok);
    }
    // shuffle epi32 within lanes
    {
        int32_t a[8]={1,2,3,4,5,6,7,8}, c[8]={};
        __m256i r = _mm256_shuffle_epi32(_mm256_loadu_si256((__m256i*)a), 0x1b); // reverse within 128b
        _mm256_storeu_si256((__m256i*)c, r);
        int ok = (c[0]==4 && c[3]==1 && c[4]==8 && c[7]==5);
        report("_mm256_shuffle_epi32", ok);
    }
    // unpack lo/hi epi32
    {
        int32_t a[8]={1,2,3,4,5,6,7,8}, b[8]={10,20,30,40,50,60,70,80}, c[8]={};
        __m256i r = _mm256_unpacklo_epi32(_mm256_loadu_si256((__m256i*)a),
                                          _mm256_loadu_si256((__m256i*)b));
        _mm256_storeu_si256((__m256i*)c, r);
        // within each 128b lane: interleave lo halves
        int ok = (c[0]==1 && c[1]==10 && c[2]==2 && c[3]==20);
        report("_mm256_unpacklo_epi32", ok);
    }
    // shuffle epi8 (pshufb)
    {
        uint8_t a[32], mask[32], c[32]={};
        for(int i=0;i<32;i++) { a[i]=(uint8_t)i; mask[i]=(uint8_t)(15-i%16); }
        __m256i r = _mm256_shuffle_epi8(_mm256_loadu_si256((__m256i*)a),
                                        _mm256_loadu_si256((__m256i*)mask));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok = (c[0]==15 && c[15]==0 && c[16]==31 && c[31]==16);
        report("_mm256_shuffle_epi8 (pshufb)", ok);
    }
}

// ─── Bucket 4: Gather Loads ───────────────────────────────────────────────────
static void test_gather() {
    section("BUCKET 4: Gather Loads");

    // gather int32 with 32b indices
    {
        int32_t base[16];
        for(int i=0;i<16;i++) base[i]=i*10;
        int32_t idx[8]={0,2,4,6,8,10,12,14};
        __m256i vi = _mm256_loadu_si256((__m256i*)idx);
        __m256i r  = _mm256_i32gather_epi32(base, vi, 4); // scale=4 bytes
        int32_t c[8]={};
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==idx[i]*10);
        report("_mm256_i32gather_epi32", ok);
    }
    // gather float with 32b indices
    {
        float base[16];
        for(int i=0;i<16;i++) base[i]=(float)i*1.5f;
        int32_t idx[8]={1,3,5,7,9,11,13,15};
        __m256i vi = _mm256_loadu_si256((__m256i*)idx);
        __m256 r   = _mm256_i32gather_ps(base, vi, 4);
        float c[8]={};
        _mm256_storeu_ps(c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (fabsf(c[i]-idx[i]*1.5f) < 1e-5f);
        report("_mm256_i32gather_ps", ok);
    }
    // gather double with 32b indices
    {
        double base[8];
        for(int i=0;i<8;i++) base[i]=(double)i*2.5;
        int32_t idx[4]={0,2,4,6};
        __m128i vi = _mm_loadu_si128((__m128i*)idx);
        __m256d r  = _mm256_i32gather_pd(base, vi, 8);
        double c[4]={};
        _mm256_storeu_pd(c, r);
        int ok=1; for(int i=0;i<4;i++) ok &= (fabs(c[i]-idx[i]*2.5) < 1e-10);
        report("_mm256_i32gather_pd", ok);
    }
}

// ─── Bucket 5: Bitwise / Shift / Compare ─────────────────────────────────────
static void test_bitwise_shift_cmp() {
    section("BUCKET 5: Bitwise / Shift / Compare");

    // AND / OR / XOR / ANDNOT
    {
        uint32_t a[8], b[8], c[8]={};
        for(int i=0;i<8;i++) { a[i]=0xF0F0F0F0u; b[i]=0x0F0F0F0Fu; }
        _mm256_storeu_si256((__m256i*)c,
            _mm256_and_si256(_mm256_loadu_si256((__m256i*)a),
                             _mm256_loadu_si256((__m256i*)b)));
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==0);
        report("_mm256_and_si256", ok);

        _mm256_storeu_si256((__m256i*)c,
            _mm256_or_si256(_mm256_loadu_si256((__m256i*)a),
                            _mm256_loadu_si256((__m256i*)b)));
        ok=1; for(int i=0;i<8;i++) ok &= (c[i]==0xFFFFFFFFu);
        report("_mm256_or_si256", ok);

        _mm256_storeu_si256((__m256i*)c,
            _mm256_xor_si256(_mm256_loadu_si256((__m256i*)a),
                             _mm256_loadu_si256((__m256i*)b)));
        ok=1; for(int i=0;i<8;i++) ok &= (c[i]==0xFFFFFFFFu);
        report("_mm256_xor_si256", ok);
    }
    // logical shift left epi32
    {
        int32_t a[8]={1,2,3,4,5,6,7,8}, c[8]={};
        __m256i r = _mm256_slli_epi32(_mm256_loadu_si256((__m256i*)a), 3);
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==(i+1)*8);
        report("_mm256_slli_epi32 (<<3)", ok);
    }
    // arithmetic shift right epi32
    {
        int32_t a[8]={-8,-16,-24,-32,-40,-48,-56,-64}, c[8]={};
        __m256i r = _mm256_srai_epi32(_mm256_loadu_si256((__m256i*)a), 3);
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==-(i+1));
        report("_mm256_srai_epi32 (>>3 signed)", ok);
    }
    // variable shift epi32 (AVX2-only)
    {
        int32_t a[8]={1,1,1,1,1,1,1,1}, sh[8]={0,1,2,3,4,5,6,7}, c[8]={};
        __m256i r = _mm256_sllv_epi32(_mm256_loadu_si256((__m256i*)a),
                                      _mm256_loadu_si256((__m256i*)sh));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==(1<<i));
        report("_mm256_sllv_epi32 (variable shift)", ok);
    }
    // compare epi32 ==
    {
        int32_t a[8]={1,2,3,4,5,6,7,8}, b[8]={1,0,3,0,5,0,7,0}, c[8]={};
        __m256i r = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i*)a),
                                       _mm256_loadu_si256((__m256i*)b));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok = (c[0]==-1 && c[1]==0 && c[2]==-1 && c[3]==0);
        report("_mm256_cmpeq_epi32", ok);
    }
    // compare epi32 >
    {
        int32_t a[8]={5,5,5,5,5,5,5,5}, b[8]={1,2,3,4,5,6,7,8}, c[8]={};
        __m256i r = _mm256_cmpgt_epi32(_mm256_loadu_si256((__m256i*)a),
                                       _mm256_loadu_si256((__m256i*)b));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok = (c[0]==-1 && c[4]==0 && c[7]==0);
        report("_mm256_cmpgt_epi32", ok);
    }
    // movemask
    {
        int32_t a[8]={-1,0,-1,0,-1,0,-1,0};
        int mask = _mm256_movemask_epi8(_mm256_loadu_si256((__m256i*)a));
        // each -1 word = 4 bytes of 0xFF → bits set
        int ok = (mask != 0);
        report("_mm256_movemask_epi8", ok);
    }
    // testz (all-zero test)
    {
        __m256i zero = _mm256_setzero_si256();
        int ok = _mm256_testz_si256(zero, zero); // 1 if all bits zero
        report("_mm256_testz_si256", ok);
    }
}

// ─── Bucket 6: Blend / Mask ───────────────────────────────────────────────────
static void test_blend_mask() {
    section("BUCKET 6: Blend / Mask");

    // blendv ps
    {
        float a[8]={1,2,3,4,5,6,7,8}, b[8]={10,20,30,40,50,60,70,80}, c[8]={};
        // mask: select b for even positions (bit31 set)
        float mask[8]={-1,0,-1,0,-1,0,-1,0};
        __m256 r = _mm256_blendv_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b),
                                    _mm256_loadu_ps(mask));
        _mm256_storeu_ps(c, r);
        int ok = (c[0]==10 && c[1]==2 && c[2]==30 && c[3]==4);
        report("_mm256_blendv_ps", ok);
    }
    // blend epi32 immediate
    {
        int32_t a[8]={1,2,3,4,5,6,7,8}, b[8]={10,20,30,40,50,60,70,80}, c[8]={};
        __m256i r = _mm256_blend_epi32(_mm256_loadu_si256((__m256i*)a),
                                       _mm256_loadu_si256((__m256i*)b), 0xAA); // alt positions
        _mm256_storeu_si256((__m256i*)c, r);
        int ok = (c[0]==1 && c[1]==20 && c[2]==3 && c[3]==40);
        report("_mm256_blend_epi32", ok);
    }
    // masked store
    {
        int32_t dst[8]={0,0,0,0,0,0,0,0};
        int32_t src[8]={1,2,3,4,5,6,7,8};
        int32_t msk[8]={-1,0,-1,0,-1,0,-1,0};
        _mm256_maskstore_epi32(dst, _mm256_loadu_si256((__m256i*)msk),
                               _mm256_loadu_si256((__m256i*)src));
        int ok = (dst[0]==1 && dst[1]==0 && dst[2]==3 && dst[3]==0);
        report("_mm256_maskstore_epi32", ok);
    }
    // masked load
    {
        int32_t src[8]={10,20,30,40,50,60,70,80};
        int32_t msk[8]={-1,0,-1,0,-1,0,-1,0};
        int32_t c[8]={};
        __m256i r = _mm256_maskload_epi32(src, _mm256_loadu_si256((__m256i*)msk));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok = (c[0]==10 && c[1]==0 && c[2]==30 && c[3]==0);
        report("_mm256_maskload_epi32", ok);
    }
}

// ─── Bucket 7: Type conversions ────────────────────────────────────────────────
static void test_conversions() {
    section("BUCKET 7: Type Conversions");

    // epi32 → float
    {
        int32_t a[8]={1,2,3,4,5,6,7,8}; float c[8]={};
        __m256 r = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)a));
        _mm256_storeu_ps(c, r);
        int ok=1; for(int i=0;i<8;i++) ok &= (c[i]==(float)(i+1));
        report("_mm256_cvtepi32_ps", ok);
    }
    // float → epi32 (truncate)
    {
        float a[8]={1.9f,2.1f,3.5f,4.0f,5.9f,6.1f,7.0f,8.8f}; int32_t c[8]={};
        __m256i r = _mm256_cvttps_epi32(_mm256_loadu_ps(a));
        _mm256_storeu_si256((__m256i*)c, r);
        int ok = (c[0]==1 && c[1]==2 && c[2]==3 && c[7]==8);
        report("_mm256_cvttps_epi32", ok);
    }
    // epi16 → epi32 sign extend
    {
        int16_t a[8]={-1,2,-3,4,-5,6,-7,8}; int32_t c[8]={};
        __m128i va = _mm_loadu_si128((__m128i*)a);
        __m256i r = _mm256_cvtepi16_epi32(va);
        _mm256_storeu_si256((__m256i*)c, r);
        int ok = (c[0]==-1 && c[1]==2 && c[2]==-3);
        report("_mm256_cvtepi16_epi32 (sign extend)", ok);
    }
    // epu8 → epi32 zero extend
    {
        uint8_t a[8]={200,100,50,25,10,5,2,1}; int32_t c[8]={};
        __m128i va = _mm_loadu_si128((__m128i*)a);
        __m256i r = _mm256_cvtepu8_epi32(va);
        _mm256_storeu_si256((__m256i*)c, r);
        int ok = (c[0]==200 && c[1]==100 && c[7]==1);
        report("_mm256_cvtepu8_epi32 (zero extend)", ok);
    }
    // float → double (128b → 256b)
    {
        float a[4]={1.0f,2.0f,3.0f,4.0f}; double c[4]={};
        __m256d r = _mm256_cvtps_pd(_mm_loadu_ps(a));
        _mm256_storeu_pd(c, r);
        int ok = (fabs(c[0]-1.0)<1e-10 && fabs(c[3]-4.0)<1e-10);
        report("_mm256_cvtps_pd", ok);
    }
}

// ─── Bucket 8: BMI1 / BMI2 ────────────────────────────────────────────────────
static void test_bmi() {
    section("BUCKET 8: BMI1 / BMI2");

    if (!has_bmi1) { printf("  SKIPPED (no BMI1)\n"); return; }

    // BLSR: reset lowest set bit
    {
        uint32_t x = 0b10110100;
        uint32_t r = _blsr_u32(x); // clears lowest set bit → 0b10110000
        report("_blsr_u32 (reset lowest set bit)", r == 0b10110000u);
    }
    // TZCNT: trailing zeros
    {
        uint32_t x = 0b10110100;
        uint32_t r = _tzcnt_u32(x); // 2 trailing zeros
        report("_tzcnt_u32", r == 2);
    }
    // LZCNT via __builtin
    {
        uint32_t x = 1u << 20;
        int r = __builtin_clz(x); // 32-21=11
        report("__builtin_clz (LZCNT)", r == 11);
    }
    if (!has_bmi2) { printf("  BMI2: SKIPPED\n"); return; }

    // PEXT: parallel bits extract
    {
        uint32_t x = 0b11001010, mask = 0b01010101;
        uint32_t r = _pext_u32(x, mask); // extract bits at mask positions
        // mask bits at pos 0,2,4,6 → x bits: 0,0,1,0 → 0b0010 = 2
        report("_pext_u32 (BMI2)", r == 2);
    }
    // PDEP: parallel bits deposit
    {
        uint32_t x = 0b0110, mask = 0b10101010;
        uint32_t r = _pdep_u32(x, mask);
        // deposit bits of x into mask positions: bit0→pos1, bit1→pos3 → 0b00001100
        report("_pdep_u32 (BMI2)", r == 0b00001100u);
    }
}
#endif // ARCH_X86

// ─── main ─────────────────────────────────────────────────────────────────────
int main() {
    char brand[49]={};
    get_cpu_brand(brand);
    detect_features();

    printf("=============================================================\n");
    printf(" AVX2 Full Probe — oneDAL Baseline Investigation\n");
    printf("=============================================================\n");
    printf("CPU:      %s\n", brand);

#ifdef ARCH_ARM
    printf("ARCH:     ARM / Apple Silicon\n");
    printf("\n=== VERDICT ===\n");
    printf("  ⚠️  ARM CPU — AVX2 is x86-only, not applicable here.\n");
    printf("     oneDAL AVX2 baseline would NOT apply to this platform.\n");
    printf("=============================================================\n");
    return 2;
#else
    printf("CPUID:    AVX2=%s  FMA=%s  BMI1=%s  BMI2=%s  AVX512F=%s\n",
           has_avx2?"YES":"NO", has_fma?"YES":"NO",
           has_bmi1?"YES":"NO", has_bmi2?"YES":"NO", has_avx512f?"YES":"NO");
    printf("OS/VM:    AVX(YMM)=%s  AVX-512(ZMM)=%s\n\n",
           os_avx_ok?"YES":"NO",
           (has_avx512f && os_avx_ok)?"CHECK":"NO");

    if (!has_avx2 || !os_avx_ok) {
        printf("❌ FATAL: AVX2 not available — skipping all tests.\n");
        return 2;
    }

    test_integer_arith();
    test_float_fma();
    test_shuffle_permute();
    test_gather();
    test_bitwise_shift_cmp();
    test_blend_mask();
    test_conversions();
    test_bmi();

    int total = g_pass + g_fail;
    printf("\n=============================================================\n");
    printf(" RESULTS: %d / %d PASSED\n", g_pass, total);
    if (g_fail == 0) {
        printf(" ✅ ALL PASS — AVX2 baseline fully functional\n");
    } else {
        printf(" ❌ %d FAILURES — AVX2 partially broken\n", g_fail);
    }
    printf("=============================================================\n");
    return g_fail > 0 ? 1 : 0;
#endif // !ARCH_ARM
}
