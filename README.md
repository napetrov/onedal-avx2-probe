# onedal-avx2-probe

Экспериментальный репозиторий для решения вопроса: можно ли сделать **AVX2 минимальным baseline** для oneDAL.

## Что делает probe

`src/avx2_full_probe.cpp` проверяет:

1. **CPUID флаги** (AVX2/FMA/BMI/AVX-512)
2. **OS/hypervisor состояние** (XSAVE/XGETBV: YMM/ZMM)
3. **Реальное выполнение интринзиков** по bucket'ам:
   - Bucket 1: Integer arithmetic
   - Bucket 2: Float + FMA
   - Bucket 3: Shuffle/Permute
   - Bucket 4: Gather
   - Bucket 5: Bitwise/Shift/Compare
   - Bucket 6: Blend/Mask
   - Bucket 7: Conversions
   - Bucket 8: BMI1/BMI2

В конце печатает итог:
- `✅ ALL PASS — AVX2 baseline fully functional`
- `❌ N FAILURES — AVX2 partially broken`

## Локальный запуск

```bash
bash scripts/run_probe.sh
```

Артефакты:
- `out/probe.log`
- `out/summary.txt`

## CI

### GitHub Actions
Файл: `.github/workflows/avx2-matrix.yml`

Сейчас включены GitHub-hosted раннеры:
- ubuntu-20.04
- ubuntu-22.04
- ubuntu-24.04

Есть заготовка под self-hosted cloud buckets (AWS/GCP/Azure) — сейчас выключено через:
```yaml
if: ${{ false }}
```
После подключения self-hosted раннеров включается одной правкой.

### Другие CI провайдеры
Шаблоны добавлены:
- `.gitlab-ci.yml`
- `.circleci/config.yml`
- `azure-pipelines.yml`

## Как интерпретировать результат

- **CPUID=YES + OS AVX(YMM)=YES + all buckets PASS** → безопасно для AVX2 baseline
- **CPUID=YES + OS AVX(YMM)=NO** → CPU умеет, но VM/OS не дает использовать AVX
- **CPUID AVX2=NO** → baseline AVX2 не будет работать на этой машине
- **Частичные bucket FAIL** → риск для baseline, нужна дополнительная валидация/исключения

## Следующий шаг

Собрать матрицу результатов по провайдерам/типам инстансов и принять решение по policy:
- hard AVX2 baseline
- fallback path (SSE4.2)
- или mixed dispatch policy
