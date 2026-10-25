# Код исследования GPT-2, reward models и RL

[Обзор и схема архитектуры](../README.md) · [Итоговая статья](../article.pdf) ·
[Точные команды воспроизведения](REPRODUCE.md)

## Два обучающих маршрута

**Цикл A:** causal prefix → GPT-2/SFT → neural RM → пятишаговый REINFORCE.
Обновляется policy; reward models и SFT reference заморожены.

**Цикл B:** реальная история → frozen GPT-2 + copy → gate → CE/joint/gated.
Обновляются 16 параметров адаптера; decoder и CatBoost reward models заморожены.
Это адаптация H1 с фиксированным реальным состоянием.

| Компонент | Модули и входные точки |
| --- | --- |
| Выбор UID, causal prefixes и train-only vocabulary | [task2_data.py](src/rec/task2_data.py) |
| GPT-2, neural RM, REINFORCE и rewards | [task2_models.py](src/rec/task2_models.py) |
| A: подготовка, train, search, controls, freeze | [research_task2.py](scripts/research_task2.py) |
| B: full-ID histories и признаки сильной RM | [task2b.py](src/rec/task2b.py) |
| B: CatBoost RM, copy mixture, CE/joint/gated | [research_task2b.py](scripts/research_task2b.py) |
| Forecasts до labels и внешние оценки | [evaluate_task2.py](scripts/evaluate_task2.py), [evaluate_task2b.py](scripts/evaluate_task2b.py) |
| Независимый CPU replay | [verify_task2.py](scripts/verify_task2.py), [verify_task2b.py](scripts/verify_task2b.py) |
| Главная статья и численные блоки | [article.tex](paper/submission_task2/article.tex), [prepare_submission_paper.py](scripts/prepare_submission_paper.py) |

`reports/task2/` и `reports/task2b/` содержат прогнозы top10, user metrics,
RM errors, trials и freeze metadata. UID manifests `reports/v3/` и `reports/v4/`
нужны только для проверки непересечения когорт. `docs/TASK2*` фиксируют
протоколы и полные команды. Сырые данные, веса и окружения в Git не включены.

## Проверка опубликованного результата

Из этой папки, без Yambda, GPU и весов:

```powershell
uv sync --locked --extra dev --python 3.11.9
uv run --locked python scripts/verify_submission.py
uv run --locked ruff check src tests scripts
uv run --locked ruff format --check src tests scripts
uv run --locked pytest -q
```

Verifier пересчитывает оба цикла, ranking, интервалы и RM diagnostics,
проверяет критический код, manifests и числа статьи. В core-окружении один
нейронный тест пропускается; extra `neural` и CUDA описаны в [REPRODUCE.md](REPRODUCE.md).
Для полного fit нужны заново полученные данные и веса предыдущих этапов.

## Оформление

```powershell
uv run --locked python scripts/build_article.py
uv run --locked python scripts/draw_research_architecture.py --task 2 --output runs/architecture-preview
```

XeLaTeX создаёт `../article-rebuilt.pdf` из главного исходника.
Схема экспортируется отдельно в PDF/SVG/PNG. Исторические статьи циклов
сохранены в `paper/task2/` и `paper/task2b/` для аудита.

Прирост Copy+CE не является эффектом RL. Все tests открыты; новый подбор
требует новой внешней проверки. Общая библиотека `src/rec` сохранена целиком
для совместимости; архитектура этой работы сосредоточена в модулях таблицы.
