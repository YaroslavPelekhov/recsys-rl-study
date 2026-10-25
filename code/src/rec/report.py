from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from rec.io import file_hash, write_json

LABELS = {
    "popularity": "Popularity",
    "item_knn": "ItemKNN",
    "rrf": "RRF",
    "manual": "CatBoost · manual",
    "auto": "CatBoost · search",
    "all_features": "CatBoost · all",
    "no_relations": "CatBoost · no relation aggregates",
}


def markdown_table(frame: pd.DataFrame, decimals: int = 5) -> str:
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{decimals}f}"
        return str(value).replace("|", "\\|")

    rows = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    rows += [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(rows)


def build_report(run: Path, output: Path) -> None:
    if not (run / "complete.json").exists():
        raise ValueError("Only completed experiments can produce a results report")
    result = json.loads((run / "results.json").read_text("utf-8"))
    complete = json.loads((run / "complete.json").read_text("utf-8"))
    if file_hash(run / "results.json") != complete["results_sha256"]:
        raise ValueError("Results checksum differs from completion manifest")
    if file_hash(run / "frozen_recipe.json") != complete["recipe_sha256"]:
        raise ValueError("Frozen recipe checksum differs from completion manifest")
    config = json.loads((run / "config.json").read_text("utf-8"))
    sample = json.loads((run / "sample.json").read_text("utf-8"))
    recipe = json.loads((run / "frozen_recipe.json").read_text("utf-8"))
    resources = pd.read_json(run / "resources.json")
    trials = pd.read_json(run / "search_trials.json")
    audits = pd.DataFrame(
        [json.loads(path.read_text("utf-8")) for path in (run / "cache").glob("*_audit.json")]
    )
    audits = audits.sort_values("cutoff")
    test_audit = audits[audits.fold == "test"].iloc[0]
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    for name in (
        "config.json",
        "environment.json",
        "source_manifest.json",
        "protocol.json",
        "feature_groups.json",
        "frozen_recipe.json",
        "search_trials.json",
        "results.json",
        "resources.json",
        "complete.json",
        "sample.json",
    ):
        shutil.copy2(run / name, evidence / name)
    audits.to_csv(evidence / "fold_audit.csv", index=False)
    if (run / "replay_verification.json").exists():
        shutil.copy2(run / "replay_verification.json", evidence / "replay_verification.json")
    for method in LABELS:
        shutil.copy2(run / "test" / f"{method}_users.parquet", evidence / f"{method}_users.parquet")
    all_metrics = pd.DataFrame(result["test"])
    numeric = ["ndcg@10", "recall@10", "map@10", "hit_rate@10", "candidate_recall", "oracle_ndcg"]
    # Metric column names follow configured K, even when this is a small fixture run.
    k = config.get("k", 10)
    numeric = [name.replace("@10", f"@{k}") for name in numeric]
    grouped = all_metrics.groupby("method", sort=False)[numeric].mean()
    seed_sd = all_metrics.groupby("method", sort=False)[f"ndcg@{k}"].std(ddof=0).fillna(0)
    table = grouped.copy()
    table["NDCG seed std"] = seed_sd
    table = table.reset_index()
    table.to_csv(evidence / "test_summary.csv", index=False)
    confirmation = (
        pd.DataFrame(result["confirmation"])
        .groupby("method", sort=False)[numeric[:3]]
        .mean()
        .reset_index()
    )
    confirmation.to_csv(evidence / "confirmation_summary.csv", index=False)
    differences = result["paired_comparisons_vs_manual"]
    delta = differences["auto"]
    improvement = delta["difference_b_minus_a"]
    low, high = delta["ci95"]
    relative = improvement / grouped.loc["manual", f"ndcg@{k}"] * 100
    selected = recipe["auto"]["selected_groups"]
    if not selected:
        conclusion = (
            "Поиск не принял ни одной группы. Auto совпадает с manual по конструкции; "
            "нулевая разность и интервал [0, 0] в этом случае не являются свидетельством "
            "эквивалентности двух разных моделей. Самостоятельное сравнение автоматической "
            "генерации с ручным baseline даёт контроль all_features."
        )
    elif low > 0:
        conclusion = (
            "На этой выборке поиск улучшил NDCG: парный 95% интервал разности целиком выше нуля."
        )
    elif high < 0:
        conclusion = (
            "На этой выборке поиск ухудшил NDCG: парный 95% интервал разности целиком ниже нуля."
        )
    else:
        conclusion = "Парный 95% интервал разности пересекает ноль. Убедительного свидетельства превосходства одного подхода нет."
    cohorts = []
    cached_test = pd.read_parquet(
        run / "cache" / "test.parquet", columns=["user_id", "user_count_all"]
    )
    history_sizes = cached_test.groupby("user_id").user_count_all.first()
    for method in ("manual", "auto"):
        per_user = pd.read_parquet(run / "test" / f"{method}_users.parquet")
        per_user["history"] = per_user.user_id.map(history_sizes)
        per_user["cohort"] = pd.cut(
            per_user.history, [0, 1, 4, np.inf], labels=["1 record", "2–4 records", "5+ records"]
        )
        for cohort, block in per_user.groupby("cohort", observed=True):
            cohorts.append(
                {
                    "method": method,
                    "history": str(cohort),
                    "users": len(block),
                    f"ndcg@{k}": block.ndcg.mean(),
                    f"recall@{k}": block.recall.mean(),
                    "mean_future_targets": block.n_relevant.mean(),
                    "candidate_recall": block.candidate_recall.mean(),
                    "oracle_ndcg": block.oracle_ndcg.mean(),
                }
            )
    cohort_table = pd.DataFrame(cohorts)
    cohort_table.to_csv(evidence / "history_cohorts.csv", index=False)
    comparisons_table = pd.DataFrame(
        [
            {
                "method": name,
                "delta NDCG": values["difference_b_minus_a"],
                "CI low": values["ci95"][0],
                "CI high": values["ci95"][1],
            }
            for name, values in differences.items()
        ]
    )
    comparisons_table.to_csv(evidence / "paired_comparisons.csv", index=False)
    importance_files = list((run / "test").glob("auto_seed*_importance.csv"))
    importance = (
        pd.concat([pd.read_csv(path) for path in importance_files])
        .groupby("feature")
        .importance.mean()
        .sort_values(ascending=False)
    )
    importance.head(15).to_csv(evidence / "auto_top_features.csv")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f7f8fc",
            "axes.facecolor": "#f7f8fc",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), layout="constrained")
    ordered = grouped.sort_values(f"ndcg@{k}")
    colors = [
        "#167d8d" if name == "auto" else "#45536b" if name == "manual" else "#a9b5c8"
        for name in ordered.index
    ]
    bars = axes[0].barh(
        [LABELS[name] for name in ordered.index], ordered[f"ndcg@{k}"], color=colors
    )
    axes[0].bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    axes[0].set_xlim(0, float(ordered[f"ndcg@{k}"].max()) * 1.28)
    axes[0].set_xlabel(f"End-to-end NDCG@{k}")
    axes[0].set_title(f"KION · {sample['sample_users']:,} sampled users")
    names = list(differences)
    centers = np.array([differences[name]["difference_b_minus_a"] for name in names])
    lows = np.array([differences[name]["ci95"][0] for name in names])
    highs = np.array([differences[name]["ci95"][1] for name in names])
    axes[1].errorbar(
        centers,
        np.arange(len(names)),
        xerr=[centers - lows, highs - centers],
        fmt="o",
        color="#167d8d",
        capsize=5,
    )
    axes[1].axvline(0, color="#8c96a8", linestyle="--", linewidth=1)
    axes[1].set_yticks(
        np.arange(len(names)), [LABELS[name].replace("CatBoost · ", "") for name in names]
    )
    axes[1].set_xlabel(f"Delta NDCG@{k} versus manual")
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[1].set_title(f"Paired user bootstrap · {delta['users']:,} test users · 95% CI")
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    total_seconds = float(resources.seconds.sum())
    search_seconds = float(resources[resources.stage.str.startswith("search/")].seconds.sum())
    fold_table = audits[
        [
            "fold",
            "cutoff",
            "end_exclusive",
            "history_rows",
            "warm_active_users",
            "positive_candidates",
            "cold_item_targets",
        ]
    ]
    details = pd.DataFrame(
        [
            {
                "method": name,
                "features": len(recipe[name]["columns"]),
                "depth": recipe[name]["depth"],
                "trees": recipe[name]["iterations"],
            }
            for name in ("manual", "auto", "all_features", "no_relations")
        ]
    )
    at_budget_limit = [
        name
        for name in ("manual", "auto", "all_features", "no_relations")
        if recipe[name]["iterations"] == config.get("iterations", 350)
    ]
    budget_note = (
        "Для "
        + ", ".join(at_budget_limit)
        + " лучший checkpoint достиг верхней границы бюджета деревьев. "
        "Эксперимент сравнивает методы при этом ограниченном бюджете; "
        "он не устанавливает максимально достижимое качество CatBoost."
        if at_budget_limit
        else "Лучшие checkpoints всех вариантов лежат внутри заданного бюджета деревьев."
    )
    report = f"""# Автоматический поиск временных признаков для рекомендаций KION

## Результат

Полный эксперимент выполнен на **{sample["sample_users"]:,} случайных пользователях**, сохранены все **{sample["sample_rows"]:,}** их исходных записей. На тесте **{int(test_audit.warm_active_users):,}** пользователей с известной историей и хотя бы одним будущим новым объектом. Все значения ниже получены кодом; незавершённые прогоны сюда не попадают.

Поиск выбрал группы **{", ".join(selected) if selected else "ни одной: оставлен ручной baseline"}**. Разность search − manual: **{improvement:+.5f} NDCG@{k}** ({relative:+.2f}% относительно baseline), парный 95% CI **[{low:+.5f}; {high:+.5f}]**. {conclusion}

![Качество и доверительные интервалы](comparison.png)

## Задача и данные

Предсказываем множество новых для пользователя фильмов/сериалов, чья **последняя зарегистрированная дата просмотра** приходится на следующую неделю. Это исследование ранжирования неявного отклика, а не CTR. Не наблюдавшийся в будущем кандидат — unobserved, а не доказанный отрицательный интерес.

Использован [официальный KION](https://github.com/irsafilo/KION_DATASET), commit `f69775be31fa5779907cf0a92ddedb70037fb5ae`, полный `data_original.zip`. В исходнике {sample["source_rows"]:,} строк. Отдельный `interactions.csv.zip` в этом commit обрезан до 50 MiB и заканчивается неполной строкой; он не используется. Период фактических данных: 13.03–22.08.2021. Год в описании репозитория не заменяет аудит загруженных данных.

Выборка равномерна по уникальным пользователям таблицы взаимодействий, seed {sample["sample_seed"]}; после выбора оставлены все строки этих пользователей. Фильтрации по будущей активности при сэмплировании нет. Список ID и SHA-256 источника сохранены в `evidence/`. В выборке {sample["duplicate_user_item_rows"]} повторных строк пары user–item.

KION хранит `last_watch_dt`, `total_dur`, `watched_pct` для пары. Нельзя восстановить атомарные события и исторические значения накопленной длительности. Поэтому `total_dur` и `watched_pct` исключены полностью. Запись с будущей последней датой недоступна прошлому срезу, даже если фактический ранний просмотр мог существовать. Это ограничение реконструкции истории, которое кодом исправить нельзя.

Статические метаданные считаются доступными и неизменными; версии на прошлые даты отсутствуют. Причинный эффект рекомендаций, вероятность клика и online-эффект по этим данным не оцениваются.

## Временной протокол

{markdown_table(fold_table, 0)}

История среза строго `timestamp < cutoff`, метки — `[cutoff, cutoff + 7 days)`. Недели меток не пересекаются. Окно признака W включает `[cutoff − W, cutoff)`, однодневная точность источника сохраняется. Будущие записи не участвуют ни в популярности, ни в графе, ни в профилях.

Первые {config.get("train_snapshots", 4)} среза обучают модель. На `search` выбираются глубина ручного baseline, число деревьев каждого варианта и группы автопризнаков. Сетка глубин: {config.get("depths", [5, 7])}; максимум {config.get("iterations", 350)} деревьев, шаг проверки 25. Используется собственная end-to-end NDCG с полным знаменателем, а не встроенная NDCG только внутри кандидатов.

После поиска записан `frozen_recipe.json`, его SHA-256 **`{result["frozen_recipe_sha256"]}`**. Для независимой проверки модели переобучаются на train + search и оцениваются на `confirm`. Для финала — на train + search + confirm с неизменными параметрами. `confirm` не выбирает победителя или дополнительные признаки. Метки и кандидаты `test` материализуются только после фиксации рецепта и независимой проверки. После просмотра теста рецепты не меняются.

Пользователь включён в ranking-оценку, если имеет хотя бы одну историческую запись и будущий новый объект. Генератор сначала строит кандидатов для **всех** исторических пользователей, не принимая меток на вход. Фильтр по активности применяется затем для экономии расчёта признаков и определения измеримой популяции. Холодные пользователи исключены и учитываются аудитом: на тесте всего {int(test_audit.future_users_all)} будущих пользователей, из них {int(test_audit.warm_active_users)} тёплых. Метрики не описывают всех зарегистрированных пользователей.

## Кандидаты и сильный baseline

Общий каталог — объекты хотя бы с одной исторической записью в пользовательской выборке. В каждом срезе исключаются уже просмотренные пары. До {config.get("retrieval_per_source", 80)} объектов из каждого источника: популярность с half-life 7 дней; ItemKNN (cosine, shrinkage `c/(c+10)`, до {config.get("neighbors", 50)} соседей); косинусная жанровая близость, умноженная на сглаженную популярность. Нулевой сигнал не номинирует кандидата. Итоговый список до {config.get("candidates", 120)} объектов выбран reciprocal rank fusion `sum(1/(20+rank))`; ties разрешаются по item ID. Граф и профили используют последние 90 дней, до {config.get("max_history", 50)} последних записей пользователя, веса профиля имеют half-life 30 дней.

Положительные объекты из будущего **не добавляются** в кандидаты. Для всех сравниваемых методов кандидаты совпадают побитово. Popularity, ItemKNN и RRF — способы упорядочить этот же набор, не отдельный эксперимент с другим retrieval.

CatBoostRanker обучается с `YetiRank:mode=NDCG;top=10`, learning rate {config.get("learning_rate", 0.05)}, L2=5, CPU. Он использует категориальные характеристики пользователя/объекта, год выпуска, retrieval scores/source flags, давность активности и ручные агрегаты в окнах 7, 30 дней и по всей истории. Косвенные связи: жанры, режиссёры, страны, тип контента; статистики sum, max и сглаженная доля истории `sum / max(user_mass, 1)`. В этих агрегатах масса многозначной записи поровну распределяется между её категориями до применения временного ядра. Прямые счётчики user–item равны нулю в задаче новых объектов и не добавляются. ID не подаются как признаки. Запросы с постоянной меткой исключены только из обучения: ранжирующий градиент в них отсутствует; оценка сохраняет пользователей без найденных положительных кандидатов.

{markdown_table(details, 0)}

{budget_note}

## Исследовательский метод

Автоматическое следование связям между таблицами и композиция агрегатов известны как [Deep Feature Synthesis (Kanter & Veeramachaneni, 2015)](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf). В [Featuretools](https://docs.featuretools.com/en/stable/getting_started/handling_time.html) временная доступность задаётся cutoff time и training window. Здесь реализован небольшой специализированный вариант этой идеи: фиксированные пути, явная граница `< cutoff`, поиск временных kernels и групповой отбор под ranking-метрику. Featuretools не используется как зависимость или экспериментальный конкурент; превосходство над DFS не проверяется.

Разделение подбора и внешней оценки связано с риском переобучения критерия выбора, описанным [Cawley & Talbot, 2010](https://www.jmlr.org/beta/papers/v11/cawley10a.html). Это обоснование протокола, а не утверждение, что одна контрольная неделя устраняет весь selection bias.

Проверяем гипотезу: фиксированные человеческие окна 7/30/all можно улучшить автоматическим выбором temporal kernels поверх того же relational grammar. Это ограниченный воспроизводимый поиск признаков, без заявления об алгоритмической новизне или полном AutoML.

Грамматика: сущность user/item либо путь user → просмотренный item → значение категории ← candidate item; ядро прямоугольного окна или экспоненциального затухания; агрегаты count/sum/max/share. Ручной baseline содержит 71 признак при конфигурации по умолчанию; один дополнительный kernel даёт 15 колонок. Список окон поиска {config.get("search_windows", [2, 3, 5, 10, 14, 21, 45, 60])}, half-lives {config.get("search_half_lives", [3, 10, 30])}. Каждая группа использует все объявленные типы связей; топология связей и вид агрегаций здесь заданы человеком.

Жадный forward search: на каждом шаге проверить каждую оставшуюся группу, добавить лучшую при приросте NDCG не меньше {config.get("min_gain", 0.0005)}, максимум {config.get("search_steps", 2)} добавления. Пустой выбор допустим. Никаких решений по test. Глубина для поиска наследуется от baseline; у каждого варианта своё число деревьев по search. История всех {recipe["search_trials"]} запусков опубликована в `evidence/search_trials.json`. Один seed используется для поиска, {len(config.get("model_seeds", [17, 42, 73]))} заранее выбранных seed — для confirm/test. Это снижает стоимость, но оставляет риск переобучения на seed и одну search-неделю.

Восемь лучших проверок на search (это данные подбора, не независимая оценка):

{markdown_table(trials.sort_values("ndcg", ascending=False).head(8))}

Два контроля: `all_features` добавляет все доступные kernels и проверяет, нужен ли отбор; `no_relations` убирает ручные агрегаты косвенных связей, сохраняя общий retrieval и его жанровый score. Это абляция агрегатов, не полное отключение контентной информации. Вычислительный бюджет поиска больше двух запусков настройки baseline и показан явно.

## Итоговые метрики

{markdown_table(table)}

Основная метрика — macro NDCG@{k} с бинарной релевантностью. IDCG использует **все** будущие новые объекты пользователя, включая не попавшие в кандидаты и холодные объекты. Recall делится на всё множество будущих целей. MAP использует знаменатель `min(K, number of relevant items)`. У пользователя без retrieved positives все эти метрики нулевые. `candidate_recall` показывает ограничение retrieval, `oracle_ndcg` — верхнюю границу NDCG при идеальном порядке текущих кандидатов.

Числа CatBoost — средние метрики отдельных seed {config.get("model_seeds", [17, 42, 73])}, не ансамбль предсказаний. Seed std — разброс NDCG между обучениями, не доверительный интервал. Candidate AUC есть в сыром JSON как вспомогательная диагностика; LogLoss не приводится, поскольку ranker не выдаёт калиброванную вероятность отклика и отрицательные примеры не являются подтверждёнными отказами.

Независимая неделя `confirm`:

{markdown_table(confirmation)}

Парный bootstrap: сначала усреднить метрику каждого пользователя по seed, затем {delta["resamples"]} раз с возвращением выбирать пользователей. Интервалы условны на этой выборке, периоде, кандидатах и обученных моделях; это не доказательство обобщения на всех клиентов. Основное заранее заданное сравнение — auto против manual; остальные интервалы описательные, поправка на множественные сравнения не применяется.

{markdown_table(comparisons_table)}

## Разбор результата

{conclusion} Относительное изменение auto: {relative:+.2f}%. Выбор групп нужно оценивать совместно с переносом на confirm, а не только с максимумом search. Победа на search после нескольких проб сама по себе не подтверждает гипотезу.

Длина доступной истории:

{markdown_table(cohort_table)}

У retrieval средний recall {grouped.loc["manual", "candidate_recall"]:.4f}, oracle NDCG {grouped.loc["manual", "oracle_ndcg"]:.4f}. Холодных объектов среди тёплых пользовательских целей: {int(test_audit.cold_item_targets)}/{int(test_audit.warm_future_rows)}. Эти цели остаются в знаменателях. При малом числе пользователей глобальная популярность и граф шумнее, а каталог меньше; усовершенствование ranker не исправляет отсутствующие кандидаты.

Топ-15 признаков auto, средняя CatBoost PredictionValuesChange importance:

{markdown_table(importance.head(15).rename("importance").reset_index(), 3)}

Importance описывает использование признака моделью, не причинный эффект. Соседние окна сильно коррелированы; вывод о необходимости компонента опирается на абляции, не только на importance.

## Затраты и масштабируемость

Сумма измеренных этапов: **{total_seconds:.1f} с**; поиск и контрольные настройки: **{search_seconds:.1f} с**; максимум sampled process RSS: **{resources.peak_process_rss_mb.max():.1f} MiB**. Внутри прогона этапы последовательны, {config.get("threads", 12)} CPU-потоков, без GPU. Основной прогон и репликация частично выполнялись одновременно: wall time отражает конкуренцию за ресурсы и не является изолированным hardware benchmark. Время включает загрузку/проверку данных; отдельные fit-этапы включают prediction, запись модели и importance. RSS опрашивается каждые 0,2 с, краткие пики могут быть пропущены. Полное окружение и длительности каждого этапа — в `evidence/`.

При N исторических записей, C кандидатных строках и G kernels агрегаты стоят примерно O(G·(N + C·число значений категорий)); текущая pandas-реализация заново разворачивает категории каждого kernel. Поиск требует до O(steps·groups) обучений. Самая опасная часть масштабирования — item–item произведение: до O(sum(history_length²)) до pruning; ограничение 50 записей снижает эту стоимость. Жанровые scores могут стать плотными, а retrieval здесь сортирует каталог для каждого пользователя. Этот reference implementation рассчитан на десятки тысяч пользователей, не на миллиарды событий. Для промышленного масштаба нужны разреженный top-K без полной сортировки, ANN, инкрементальные агрегаты и распределённый feature store. Линейный прогноз времени на миллиарды по этому прогону был бы необоснованным.

## Ограничения и следующий эксперимент

Одна пользовательская выборка и один финальный временной интервал; короткие истории KION; last-watch вместо атомарных событий; неизвестный exposure; неизменность метаданных лишь предположение. Тест уже открыт: изменение методов по этим числам потребует нового отложенного периода или другого датасета.

Следующая заранее фиксируемая проверка: несколько user-sample seeds и больший масштаб, несколько внешних rolling test-недель, затем Yambda с сохранением всей истории выбранных пользователей. Чтобы проверить необходимость обучаемого представления, можно добавить историю с learnable decay или sequence encoder, сохраняя тот же протокол доступной информации. Эти эксперименты здесь не выдаются за выполненные.

## Воспроизведение

Из корня репозитория:

```powershell
uv sync --locked --extra dev
uv run --locked pytest -q
uv run --locked rec run --config configs/main.json --output runs/reproduction
uv run --locked rec report --output runs/reproduction --report-dir reports/reproduction
```

Для быстрого smoke-run: `configs/pilot.json`. Каталог запуска не перезаписывается. `runs/` и `data/` исключены из git; полный компактный набор доказательств этого результата находится в `evidence/`. Он позволяет независимо пересчитать средние, интервалы и сопоставить конфигурацию без публикации исходного датасета. Полные predictions и `.cbm` сохраняются в локальном `runs/.../test/`.

## Источники

- [Petrov et al., MTS Kion Implicit Contextualised Sequential Dataset, 2022](https://arxiv.org/abs/2209.00325).
- [Официальный репозиторий KION, фиксированный commit](https://github.com/irsafilo/KION_DATASET/tree/f69775be31fa5779907cf0a92ddedb70037fb5ae).
- [CatBoost: ranking objectives, YetiRank mode NDCG](https://catboost.ai/docs/en/concepts/loss-functions-ranking).
- [Kanter & Veeramachaneni, Deep Feature Synthesis, 2015](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf).
- [Cawley & Talbot, On Over-fitting in Model Selection, JMLR 2010](https://www.jmlr.org/beta/papers/v11/cawley10a.html).
- [Yambda: официальный dataset card](https://huggingface.co/datasets/yandex/yambda) — направление следующей проверки, в текущем результате не используется.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    checksums = {
        str(path.relative_to(evidence)): file_hash(path)
        for path in sorted(evidence.iterdir())
        if path.is_file() and path.name != "checksums.json"
    }
    write_json(evidence / "checksums.json", checksums)
