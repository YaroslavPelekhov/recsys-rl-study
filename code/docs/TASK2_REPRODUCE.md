# Воспроизведение второй задачи

Основной отчёт: [reports/task2/REPORT.md](../reports/task2/REPORT.md). Первая задача, её статья и результаты не заменены.

## Быстрая проверка без данных и GPU

Python3.11.9 и uv; зависимости зафиксированы в uv.lock:

```powershell
uv sync --locked --extra dev
uv run --locked python scripts/verify_task2.py
uv run --locked python scripts/publish_task2.py --check
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked pytest -q
```

Проверка читает27 top10 prediction arrays и исходные held-out labels, заново считает ranking-метрики/срезы/seeds, user-cluster CI, RM metrics и sanity checks rollout. Хеш проверяется для каждого компактного файла; один лишь совпавший checksums.json не заменяет пересчёт. Нейронный unit test запускается при установленном PyTorch; в core environment он пропускается.

## Полный эксперимент

Проверенная среда: Windows, i7-14700KF,48GB RAM, RTX4070TiSUPER16GB, PyTorch2.5.1+cu121. В исследовании использована отдельная `.venv-neural` с pinned core packages. Для нового окружения:

```powershell
uv sync --locked --extra dev --extra neural
```

Далее `uv run --locked --extra neural python ...` выполняет GPU-этапы. У сборок PyTorch/CUDA возможны различия последних битов; опубликованный exact replay проверяет сохранённые веса в исходной среде, а не обещает одинаковые веса на любом устройстве. Для точного tested CUDA build можно отдельно установить torch2.5.1+cu121 из официального PyTorch cu121 index, сохранив версии core packages. Политики малы, но нужен запас свободного диска/Windows host commit; запуск при почти полном диске вызвал технические ошибки. Данные читаются порциями, RM contexts — read-only memmap. Содержимое NTFS-сжатых файлов не меняется.

Запуск из корня репозитория; используйте новое имя run, чтобы не перезаписать исходные результаты:

```powershell
uv run --locked python scripts/download_task2.py
uv run --locked --extra neural python scripts/research_task2.py reserve --root runs/task2-reproduction
uv run --locked --extra neural python scripts/research_task2.py prepare --root runs/task2-reproduction
uv run --locked --extra neural python scripts/research_task2.py development --root runs/task2-reproduction
uv run --locked --extra neural python scripts/task2_controls.py --root runs/task2-reproduction
uv run --locked --extra neural python scripts/evaluate_task2.py forecast --root runs/task2-reproduction
uv run --locked --extra neural python scripts/evaluate_task2.py open --root runs/task2-reproduction
uv run --locked --extra neural python scripts/evaluate_task2.py aggregate --root runs/task2-reproduction --output reports/task2-reproduction/evidence
uv run --locked --extra neural python scripts/evaluate_task2.py reward --root runs/task2-reproduction --output reports/task2-reproduction/evidence
uv run --locked --extra neural python scripts/evaluate_task2.py rollouts --root runs/task2-reproduction --output reports/task2-reproduction/evidence
uv run --locked --extra neural python scripts/audit_task2.py --root runs/task2-reproduction --output reports/task2-reproduction/evidence/audit.json
uv run --locked python scripts/publish_task2.py --run runs/task2-reproduction --report reports/task2-reproduction
uv run --locked python scripts/verify_task2.py --report-dir reports/task2-reproduction
```

`development` обучает SFT seed17 до12000 updates и выбирает checkpoint только по search. Затем4 RM,15 RL trials и по выбранным настройкам отдельные policy seeds17/42/73, CE/H1 controls. Рецепт и SHA сохраняются до forecast. Данные test загружаются как исходные строки, но target item/outcome не используются при прогнозе. `forecast` сохраняет все24 neural и3 non-neural top10 arrays, затем prediction seal. Только `open` строит оценочные labels. Код запрещает добавлять новые forecasts после открытия labels. Это логическая изоляция признаков/выбора, не физически отдельное хранилище исходного CSV/Parquet.

Sampling исключает UID из опубликованных manifests v3/v4, сохраняет все строки1200 выбранных UID,800/400 делятся отдельно. Коллекция отражает активную популяцию исходного Yambda, не случайных пользователей всего сервиса. RM target Activity требует24h наблюдения; train examples заканчиваются до269, внешний test до299, поэтому label horizon помещается в источник.

Пилот сохранён в опубликованном evidence/pilot_logs/ и pilot_trials.json; он не нужен для финальных весов. Его SFT recipe воспроизводится командой `research_task2.py sft --root НОВЫЙ_RUN --budget 1500` после reserve/prepare, RL trials — фазой `rl --kind ... --beta ...`. Досрочно прерванный ensemble fit не считается завершённым. Поправка бюджета явно development-driven; это не неизменная preregistration с самого первого пилота.

## Код и доказательства

| Файл | Ответственность |
|---|---|
| [task2_data.py](../src/rec/task2_data.py) | user sampling, словарь по прошлому, timestamps, causal prefixes, targets |
| [task2_models.py](../src/rec/task2_models.py) | GPT-2 blocks, RM, REINFORCE и rewards |
| [research_task2.py](../scripts/research_task2.py) | обучение, search и freeze |
| [evaluate_task2.py](../scripts/evaluate_task2.py) | forecasts до labels, метрики и rollout audit |
| [audit_task2.py](../scripts/audit_task2.py) | полнота source rows, future mutation, exact top10 replay |
| [verify_task2.py](../scripts/verify_task2.py) | компактная независимая от GPU проверка результатов |
| [publish_task2.py](../scripts/publish_task2.py) | таблицы, numerical LaTeX commands и векторные графики |

Сырые данные, train arrays и weights находятся в data/ и runs/. Точный перечень компактных доказательств и их SHA — в reports/task2/evidence/checksums.json. Модели и raw Parquet не нужны для проверки опубликованных чисел. RM predictions на generated actions не являются контрфактуальной разметкой.

## LaTeX и графики

Текст редактируется в [paper/task2/article.tex](../paper/task2/article.tex); numbers.tex и figures обновляются из evidence. Используются XeLaTeX и системные Times New Roman/Arial либо fallback fonts. Нужны обычные пакеты fontspec, babel Russian, geometry, amsmath, booktabs, microtype, titlesec, fancyhdr, caption, hyperref.

```powershell
uv run --locked python scripts/build_task2_paper.py
```

Графики — vector PDF/SVG, исходные численные данные — evidence/figure_data.json. Для обновления артефактов исходного запуска: `uv run --locked python scripts/publish_task2.py`. Обычная проверка использует `--check` и не переписывает их. Для нового report создаётся отдельная папка графиков/numbers.tex под paper/; исходная статья не переписывается. Исторический журнал пилота при отсутствии runs/task2 берётся из опубликованного evidence и не выдаётся за новое обучение. Статья проверена рендером всех страниц; optional PDF environment описан в requirements-paper.txt, Poppler используется для PNG.
