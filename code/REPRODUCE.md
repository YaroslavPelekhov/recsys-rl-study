# Воспроизведение работы 2

Команды выполняются из `code/`. Проверка опубликованного результата и
повторное обучение — разные режимы. Для первого не нужны Yambda, GPU и веса.

```powershell
uv sync --locked --extra dev --python 3.11.9
uv run --locked python scripts/verify_submission.py
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked pytest -q
```

Verifier проверяет два цикла, raw top10, интервалы, RM errors, rollout checks,
неизменность критического кода и связь единой статьи с результатами.
Оба tests открыты; новая настройка по ним не является подтверждением.
Тест PyTorch пропускается в core-окружении. Для нейронных unit tests:

```powershell
uv sync --locked --extra dev --extra neural
uv run --locked --extra neural pytest -q tests/test_task2.py tests/test_task2b.py
```

## Цикл A: полное обучение GPT-2 и REINFORCE

Источник — pinned Yambda flat/50m/listens, скачивание проверяет SHA.
Из прежних исследований нужны только включённые sample manifests v3/v4.
Исходные данные и веса не входят в архив; полный прогон требует CUDA/памяти
и доступа к официальному источнику. В опубликованном запуске использовались
Python 3.11.9, torch 2.5.1+cu121, RTX4070TiSUPER16GB и 48GB RAM.

Точные этапы приведены в [docs/TASK2_REPRODUCE.md](docs/TASK2_REPRODUCE.md):
download → reserve → prepare → development → controls → frozen forecasts
→ labels → aggregate/RM/rollouts → audit. Сохраняйте новый output report
в отдельную папку, как в той инструкции. Публичные `reports/task2*` —
зафиксированный эталон, а не место для новых поисков.

## Цикл B: copy, сильная RM, joint/gated

Следуйте [docs/TASK2B_REPRODUCE.md](docs/TASK2B_REPRODUCE.md).
Он переиспользует три SFT/RM веса цикла A, его train sample и catalog в
`runs/task2_final/`. В чистой копии можно сразу дать первому циклу это имя
run; не перезаписывайте существующий `runs/task2_final`.
Второй цикл использует `runs/task2b`, исключает все старые UID и сначала
фиксирует новую когорту. Последовательность RM → cache → adapter → controls
→ audit → freeze → forecasts → labels → results описана точно.

Для сравнения переобученного прогона достаточно `runs/task2b/results.json`.
Команда `evaluate_task2b.py publish` предназначена для создания отчёта и
переписывает одноимённую папку публикации: в рабочей копии для воспроизведения
её можно пропустить, сохранив эталонные reports. Supplemental runtime history
описывает именно опубликованный запуск; новый fit не обязан воспроизвести
его технические сбои или бинарные идентификаторы CatBoost.
Метрики могут немного различаться между CUDA/CPU/backend. Сохранённые
предсказания позволяют независимо проверить опубликованные выводы.

## Статья

```powershell
uv run --locked python scripts/build_article.py
```

XeLaTeX собирает единую статью из `paper/submission_task2/article.tex`
в `../article-rebuilt.pdf`. Первоначальные статьи обоих циклов сохранены
для аудита; главный файл для сдачи находится в корне комплекта.
Для Overleaf используйте корневой `latex.zip`, compiler XeLaTeX,
main `article.tex`, два прохода; BibTeX не требуется.
