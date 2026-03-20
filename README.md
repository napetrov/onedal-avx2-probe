# onedal-avx2-probe

Experimental repository to answer the question: **can AVX2 be used as the minimum CPU baseline for oneDAL?**

## What the probe does

`src/avx2_full_probe.cpp` checks:

1. **CPUID flags** (AVX2/FMA/BMI/AVX-512)
2. **OS/hypervisor state** (XSAVE/XGETBV: YMM/ZMM registers enabled)
3. **Actual intrinsic execution** across 8 buckets:
   - Bucket 1: Integer arithmetic (epi8/16/32/64)
   - Bucket 2: Float arithmetic + FMA
   - Bucket 3: Shuffle / Permute
   - Bucket 4: Gather loads
   - Bucket 5: Bitwise / Shift / Compare
   - Bucket 6: Blend / Mask
   - Bucket 7: Type conversions
   - Bucket 8: BMI1 / BMI2

Final verdict:
- `✅ ALL PASS — AVX2 baseline fully functional`
- `❌ N FAILURES — AVX2 partially broken`

## Local run

```bash
bash scripts/run_probe.sh
```

Artifacts written to:
- `out/probe.log` — full output
- `out/summary.txt` — cpu/arch/rc/verdict one-liners

## CI

### GitHub Actions
File: `.github/workflows/avx2-matrix.yml`

Currently enabled runners:

**Linux (GitHub-hosted):**
- ubuntu-20.04
- ubuntu-22.04
- ubuntu-24.04

**macOS:**
- macos-12 / macos-13 — Intel x86_64, native AVX2
- macos-14 / macos-15 — Apple Silicon + **Rosetta 2** (x86_64 binary via `arch -x86_64`)
  - Expected result: `❌ FATAL: AVX2 not available` — Rosetta 2 does not support AVX/AVX2

**Windows:**
- windows-2019 / windows-2022 — MSVC x64

**Self-hosted cloud runners** (AWS/GCP/Azure) — prepared but disabled:
```yaml
if: ${{ false }}   # ← flip to true after registering runners
```
Enable with a single line change once runners are registered with matching labels.

### Other CI providers
Templates included:
- `.gitlab-ci.yml`
- `.circleci/config.yml`
- `azure-pipelines.yml`

## How to interpret results

| Result | Meaning |
|--------|---------|
| CPUID AVX2=YES + OS AVX(YMM)=YES + all buckets PASS | ✅ Safe for AVX2 baseline |
| CPUID AVX2=YES + OS AVX(YMM)=NO | ⚠️ CPU supports AVX2 but OS/hypervisor has disabled YMM state (common in some VMs) |
| CPUID AVX2=NO | ❌ Pre-Haswell CPU, AVX2 baseline will not work |
| ARCH=ARM | ⚠️ Not applicable (x86-only feature) |
| x86_64 under Rosetta 2 | ❌ Rosetta does not expose AVX/AVX2 to translated binaries |
| Partial bucket FAIL | ❌ Risk for baseline, needs further investigation |

## Next steps

Collect results across providers/instance types and decide on oneDAL policy:
- Hard AVX2 baseline (drop SSE4.2 fallback)
- Keep SSE4.2 fallback path
- Mixed runtime dispatch policy
