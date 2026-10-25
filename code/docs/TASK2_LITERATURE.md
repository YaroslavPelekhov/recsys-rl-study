# Литература и границы вклада

Проверены первоисточники 21–22 октября 2026. Ниже перечислены идеи, которые
заимствуются, и то, что проверяет наш эксперимент. Это не систематический обзор.

| Работа | Что берём | Чем отличается наша проверка |
| --- | --- | --- |
| Radford et al., *Language Models are Unsupervised Multitask Learners*, 2019; [исходная архитектура GPT-2](https://github.com/openai/gpt-2/blob/master/src/model.py) | Causal decoder, pre-LN, learned positions, GELU, tied token/output embeddings | Маленькая сеть с нуля на item IDs, без переноса текстовых весов; словарь ограничен прошлым |
| Petrov & Macdonald, [*Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning*](https://arxiv.org/abs/2403.04875), GenRec 2024 | Различие ранжирования Top-K и последовательной генерации Next-K; RL для нескольких целей | В статье teacher–student и PPO, цели разнообразия/популярности. У нас next-item CE и REINFORCE по обученному feedback; центральный риск — ошибка reward model и вымышленная история |
| Matveev et al., [*Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction*](https://arxiv.org/abs/2607.02818v2), 2026 | Авторегрессионная policy, feedback model, необходимость оценивать session-level цели отдельно от retrieval | Их two-tower модель, многозвенные importance weights и OPE на Yambda-5B не воспроизводятся. Мы оптимизируем on-policy в learned surrogate, не выдаём симулятор за реальную среду. Их добавление out-of-index target в evaluation index не используется: наши OOV всегда промах |
| Gao, Schulman & Hilton, [*Scaling Laws for Reward Model Overoptimization*](https://arxiv.org/abs/2210.10760), 2022 | Разделение оптимизируемой proxy-награды и внешней проверки, зависимость от силы оптимизации | У нас нет gold reward oracle. Даже независимая инициализация audit RM не даёт истинной награды; реальные observed outcomes доступны лишь на logged actions |
| Coste et al., [*Reward Model Ensembles Help Mitigate Overoptimization*](https://arxiv.org/abs/2310.02743), ICLR 2024 | Консервативная оптимизация ансамбля, совместимость с KL штрафом | Ensemble mean−std — известный контроль. Добавляем отдельный штраф за сдвиг reward между исходной и авторегрессионно дополненной историей и изолируем его эффект |
| Yu et al., [*MOPO: Model-based Offline Policy Optimization*](https://arxiv.org/abs/2005.13239), NeurIPS 2020 | Риск ухода модельной среды за поддержку логов, штрафы за неопределённость | Мы не учим пользовательскую transition dynamics и не переносим теоретическую lower-bound гарантию MOPO на наш эвристический штраф |
| Xu et al., [*Adaptive Loss Balancing for Noise-Robust GRPO in Generative Recommendation*](https://arxiv.org/abs/2606.08480), препринт2026 | Неодинаковая надёжность ranker reward для разных примеров и стабилизация supervised loss | Их per-sample admission gate и NLL+GRPO не реализованы. Наш штраф ограничивает зависимость reward от generated context; CE control сам по себе не заменяет joint NLL+RL baseline |

## Проверяемое дополнение

Для начальной реальной истории h0, её версии ht с добавленными generated
items и действия a обозначим среднее ratio трёх RM через m(h,a), а их
population std через s(h,a). Известный контроль: rE=m(ht,a)−0.5s(ht,a).
Предложенная проверка: rA=rE−|m(ht,a)−m(h0,a)|.

На первом шаге штраф нулевой. На последующих он не даёт оптимизатору
бесплатно получать всю прибавку только за счёт смены контекста самим собой.
Штраф может также мешать полезной адаптации, не различает ошибочный и
правильный сдвиг и не восстанавливает настоящую реакцию пользователя.
Поэтому нужны matched-beta контроль, H1 ablation и отдельная оценка на
реальных prefixes. Положительный результат не превращает эту комбинацию
в доказанно новый общий RL алгоритм.

## Какие выводы нельзя сделать

Высокий predicted return24 не доказывает улучшение retention. Доля
прослушивания и длительность не измеряют удовлетворённость. В логах нет
вероятностей показов и контрфактуальных откликов; мы не называем простое
усреднение RM предсказаний IPS, DR или unbiased off-policy value.
Пять выбранных треков — список рекомендаций в симуляторе, а не пять
фактических действий пользователя.
