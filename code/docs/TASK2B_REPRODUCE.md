# Воспроизведение второго цикла задачи 2

Новый test открыт; повторение воспроизводит опубликованный эксперимент и
не создаёт новую независимую проверку. Первая работа и Task2-1 не меняются.

## Быстрая проверка без исходных данных/GPU

```powershell
uv sync --locked --extra dev --python 3.11.9
uv run --locked python scripts/verify_task2b.py
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked pytest -q
```

Метрики пересчитываются из24 массивов raw top10. Хешей недостаточно: verifier
сверяет per-query показатели, user-cluster bootstrap и event-weighted RM CI.

## Полный запуск

Нужны pinned Yambda listens из [инструкции первого цикла](TASK2_REPRODUCE.md),
его сохранённые `sampled_events.parquet`, `sample.json`, `catalog.npz` и три
SFT/RM веса в `runs/task2_final/models`. SFT обучается исходным протоколом
Task2, новые настройки по его test не выбираются. Веса не входят в compact ZIP.
Используйте neural окружение из той же инструкции: Python3.11.9,
torch2.5.1+cu121, зависимости основного uv.lock. CPU воспроизводит проверку;
описанный полный fit выполнялся с CUDA/bfloat16.

Начинать полный fit следует в чистой копии репозитория без `runs/task2b`,
сохранив перечисленные исходные данные/веса. Команды не удаляют старые runs.

```powershell
.venv/Scripts/python.exe -c "from pathlib import Path; from rec.task2b import reserve, prepare; r=Path('runs/task2b'); reserve(r); prepare(r)"
.venv/Scripts/python.exe scripts/research_task2b.py rm
.venv/Scripts/python.exe scripts/task2b_reward_controls.py
foreach ($taskSeed in 17,42,73) {
  .venv-neural/Scripts/python.exe scripts/research_task2b.py cache --seed $taskSeed
  .venv-neural/Scripts/python.exe scripts/research_task2b.py adapter --seed $taskSeed
}
.venv-neural/Scripts/python.exe scripts/task2b_reward_controls.py --policy
.venv/Scripts/python.exe scripts/audit_task2b.py
.venv/Scripts/python.exe scripts/research_task2b.py freeze
.venv-neural/Scripts/python.exe scripts/evaluate_task2b.py forecast
.venv-neural/Scripts/python.exe scripts/evaluate_task2b.py labels
.venv/Scripts/python.exe scripts/evaluate_task2b.py results
.venv/Scripts/python.exe scripts/evaluate_task2b.py publish
.venv/Scripts/python.exe scripts/analyze_task2b.py --create-lengths
```

`labels` отказывается перезаписывать открытые метки; `forecast` отказывается
работать после seal/labels. `freeze` проверяется перед прогнозами и метками.
На Windows при ограниченном host commit GPU может выдать OOM даже при свободной
VRAM. В фактическом запуске test-кэши считались отдельными процессами после
подготовки test.npz неудачной попыткой forecast, до seal/labels:

```powershell
foreach ($taskSeed in 17,42,73) {
  .venv-neural/Scripts/python.exe -c "import sys; sys.path.insert(0,'scripts'); from research_task2b import cache; cache($taskSeed,'test')"
}
.venv-neural/Scripts/python.exe scripts/evaluate_task2b.py forecast
```

В актуальном коде освобождается Arrow pool перед GPU; экономный fallback
repeat математически эквивалентен прежнему. Модели и параметры при исправлении
не менялись. `evidence/runtime_fix.json` фиксирует сбои, старый freeze и
побайтное совпадение22 ранее вычисленных прогнозов. Supplemental diagnostics
и описание технических сбоев опубликованы отдельно от основного генератора
evidence; первичные метрики полностью воспроизводятся командами выше.

`analyze_task2b.py` добавляет описательный аудит длительности по metadata,
известным к train cutoff270, и сохраняет полный массив для CPU-пересчёта.
Первичный кэш duration200 с неизвестными значениями0 сохранён. Этот аудит
выполнен после test, не участвовал в выборе и не менял policy/RM.
Технические before/after freeze и журнал ошибок относятся к опубликованному
историческому запуску; новый fit может иметь другую техническую историю.

## Статья и графики

```powershell
uv run --locked python scripts/publish_task2b.py
uv run --locked python scripts/build_task2b_paper.py
```

Нужен XeLaTeX. `paper/task2b/article.tex` редактируется вручную; publisher
обновляет численные блоки, фигуры и полный журнал, не перезаписывая текст.
`output/pdf/task2-research-paper.pdf` — новый PDF; прежние статьи сохраняются.
