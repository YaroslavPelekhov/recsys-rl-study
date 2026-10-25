# Задача 2: RL, выученная награда и вымышленная история

Протокол записан до обучения и открытия новых test-метрик, 21 октября 2026.
Задача 1 и её опубликованные результаты остаются отдельным исследованием.

## Вопрос

Может ли RL повысить предсказанную долю прослушивания, сохранив точность
next-item рекомендаций? Отдельно проверяем, насколько результат зависит от
добавления сгенерированных рекомендаций в историю как будто это прослушивания.

Самостоятельная проверяемая идея — штраф за расхождение reward model между
дополненной и исходной, фактически наблюдавшейся историей. Это инженерная
гипотеза, не заявление мирового приоритета и не доверительная граница.

## Данные и новое отложенное множество

Pinned Yambda flat/50m/listens.parquet, revision
dd6f3a19eef5866e346c3270e098baa641a44948. Исключить все UID v3 и v4.
Равномерно выбрать 1200 оставшихся UID, RNG 718281; все исходные строки
выбранных пользователей сохранить. Независимая перестановка RNG 718282
делит пользователей на development 800 и external 400. Отбор не зависит
от будущей активности. Manifest и SHA фиксируются до обучения.

Время — анонимизированные секунды, сутки = 86400, не календарные даты.
Обучение всех моделей и статистик: timestamp < day270. RM с меткой возврата
использует только примеры timestamp < day269, чтобы весь 24h label horizon
был внутри train. Search: days280..286 только development UID.
Внешняя проверка: days293..298 только external UID; до конца источника
остаётся полный день для наблюдения return24. Исторические строки
external UID до270 участвуют в обучении; их последующие метки не участвуют
в выборе настроек. Это временной holdout с отдельной user-когортой для
оценки, не cold-user задача и не новый сервис. Период частично совпадает
с прежними исследованиями, UID не пересекаются; это не независимая
репликация на новом будущем периоде.

Все модели заморожены после train. На eval используются реальные истории
строго timestamp < момента запроса; supervised обновления нет. События с
одинаковым timestamp не упорядочиваются как причинно последовательные:
loss на переходах внутри timestamp исключён, query содержит только более
ранние timestamps. В оценку попадают первые до4 неорганических событий
с уникальным timestamp за user-day. Это фиксированный ограниченный estimand
на logged recommendation opportunities, не все пользователи и не impressions.
Нет отбора по played_ratio. Тёплый пользователь имеет хотя бы один более
ранний item. Порядок и частота логирования считаются данными logging policy.

Каталог: top16384 item по train counts, ties по item ID. Кандидаты всех
методов — весь этот каталог. Иные items кодируются UNK в контексте и
остаются промахами в знаменателях ranking-метрик; positive injection нет.
Все исходные строки сохраняются; ограничены модельный контекст/словарь,
не исходные истории. Длительность кандидата — median track_length_seconds
только train-строк этого item. Никакие будущие метаданные не читаются.

## Baseline и reward model

GPT-2 с нуля: item embeddings, learned positional embeddings, 3 pre-LN
causal decoder blocks, width128, 4 heads, GELU tanh, MLP expansion4,
final LN, tied output embeddings, scaled residual initialization.
Контекст64. Это уменьшенная архитектура GPT-2, не текстовые pretrained weights.
Полная softmax по каталогу; PAD/UNK/BOS не рекомендуются.
AdamW lr3e-4, decay.01, batch64, максимум1500 updates, warmup100 и cosine.
Checkpoints300/600/900/1200/1500 выбираются только search NDCG@10, seed17;
seeds42/73 получают то же число updates. Все trajectories публикуются.
Контроли: train popularity, causal repeat по контексту, train first-order
Markov со сглаживанием popularity, дополнительное CE обучение120 updates.

RM: отдельная нейросеть с item embeddings и mean/last history pooling,
history–candidate interactions и train-only duration; два sigmoid head.
Первый предсказывает clip(played_ratio_pct/100,0,1), второй — наличие хотя
бы одного события в интервале [t+1h,t+24h). Это ассоциативный return proxy,
не causal retention. Контекст только item IDs, текущий outcome в input не
попадает. Обучение только на реально наблюдавшихся неорганических listens;
непрослушанные товары не размечаются нулевым reward. Три independent-init
RM seeds17/42/73, четвёртая audit model seed101 не используется в reward.
Fixed train budget800 updates, batch512, lr1e-3; BCE с fractional target
для ratio и binary target для return. Простые контроли: global mean и
item mean со shrinkage20. RM metrics: MSE, MAE, calibration и return AUC/Brier
на search и test logged actions, включая OOV. Audit model обучена на тех же
данных и не является независимой разметкой или oracle.

## RL и контролируемые варианты

REINFORCE в выученном симуляторе, on-policy sampling из полной softmax,
5 шагов с добавлением action ID в контекст (без выдуманных playback labels).
Все trajectories начинаются из реальных train prefixes, uniform по users,
затем uniform по доступным prefix positions пользователя. Нет ground-truth
future items в RL. gamma=.9, AdamW lr5e-5, batch96, 120 updates, grad clip1.
Reference — frozen SFT. Reward включает sampled log-policy/reference penalty
beta*log(pi/pi_ref); discounted return-to-go и leave-one-out batch baseline
задают policy gradient. Условия одинаковы для всех вариантов.

Пять rewards, beta в {.02,.1,.5}: (1) single RM ratio; (2) single RM
ratio*min(duration,600)/600; (3) single RM .5ratio+.5return24;
(4) mean ratio − .5 ensemble std; (5) то же минус абсолютное расхождение
mean ratio в дополненном и первоначальном контексте. Последний штраф =0
на первом шаге. Std — разброс трёх моделей, не calibrated uncertainty.
Всего15 search fits seed17; все checkpoints фиксированы на120 updates.
Для каждого reward выбрать beta с максимальным search predicted ratio,
среди вариантов с NDCG >= .95*SFT; если их нет — максимальный NDCG,
ties больший beta. Это выбор рабочей точки, не доказательство non-inferiority.

**Основной заранее заданный контраст: anchored − ensemble, beta=.1**
по user-macro NDCG@10. Оба варианта обязательно входят во внешний тест
независимо от результатов search. Дополнительно выбранные5 рабочих точек,
CE control и H1 ensemble при beta=.1. Seeds17/42/73 — отдельные полные
SFT/RL fits; RM ensemble общий фиксированный компонент.

## Оценка, фиксация и ограничения

Recall/NDCG/MAP@10 для одного следующего logged item, positive>=50 slice,
novel/repeat и OOV. Среднее queries внутри пользователя, затем users;
сначала mean model seeds для каждого query. Парный user-cluster bootstrap
3000 resamples RNG718283. Primary CI один; прочие интервалы описательные.
До test фиксируются recipe/code/data hashes, trials и model checkpoints.
Все test top10 predictions сохраняются до чтения target item/outcome.

Также: RM reward top1, audit reward, exact KL, entropy, global catalog
coverage, top1 concentration, repeat share, рекомендуемая длительность,
пять rollout steps, within-list duplicates, разница reward в реальном и
вымышленном контексте. Отдельно frozen-history rollout (контекст не меняется)
как diagnostic intervention, без переобучения по результату.

Model-based reward и согласие разных RM не равны реальному пользовательскому
эффекту. Logging propensities отсутствуют: IPS/DR causal оценки не заявляются.
Возврат — характеристика последующего поведения, не эффект данного трека.
No online A/B, нет свидетельства улучшения удовлетворённости или retention.
Ограниченный словарь и выборка, один период и вычислительный бюджет публикуются.
GPU/RSS/wall time измеряются; компактные таблицы и per-user evidence — reports/task2,
сырые строки/веса — runs/task2. Test после публикации считается открытым.
