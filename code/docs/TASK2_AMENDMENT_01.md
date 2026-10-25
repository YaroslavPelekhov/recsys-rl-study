# Уточнение бюджета после development-пилота

22 октября 2026, до построения external queries, forecasts или labels.

Первый SFT seed17 за1500 updates: search NDCG@10 .00957560; train CE
снизилась с9.1008 на300 до7.9507 на1500, лучший checkpoint — граница бюджета.
Это слишком слабая основа для вывода о полезности RL. Девять завершённых
RL trials (ratio/seconds/return ×3beta) сохранены в runs/task2, а также
частично начатый ensemble trial. Test ни разу не оценивался. Пилот не
используется как подтверждающее сравнение и не скрывается.

Для финального цикла увеличить SFT budget до12000 updates с нуля,
warmup300, cosine до12000, search checkpoints через1500. Это примерно
10 проходов по train chunks с выборкой с возвращением, не доказательство
сходимости. Выбор checkpoint по тому же development; остальные seeds
получают выбранное число updates и тот же полный12000-step LR schedule.
Архитектура, данные, словарь, RM, RL rewards/сетка/primary contrast неизменны.
Raw sample, train arrays и RM weights переиспользуются по SHA, без нового
сэмплирования. Финальный run: runs/task2_final. Исходный протокол остаётся
доступен; это зарегистрированная development-driven поправка, не исходная
полностью неизменная preregistration.

Технически effective SFT batch64 обрабатывается четырьмя microbatch16.
Ограничение cuBLAS workspace :16:8 устраняет ошибки host commit на Windows.
Оно не меняет целевую функцию. Существующие модели/библиотеки были сжаты
NTFS без изменения содержимого; исходные данные/старые результаты не удалены.
