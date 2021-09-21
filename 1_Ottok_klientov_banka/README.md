# YaPracticum

Отток клиентов банка:
Задача - Анализ оттока клиентов из банка для выбор стратегии (удержание старых клиентов или привлечение новых клиентов)

Проведена работа с несбалансированными данными. Спрогнозирована вероятность ухода клиента из банка в ближайшее время. Построена модель с предельно большим значением F1-меры с последующей проверкой на тестовой выборке. Доведена метрика до 0.59. Дополнительно измерен AUC-ROC, соотнесен с F1-мерой. Проведенная нами исследование закономерно показало, что предобработка данных и балансировка целевых классов в обучающих выборках, значительно сказывается на улучшении качества предсказания целевого признака.

Инструменты: Pandas, Matplotlib, Seaborn, numpy, sklearn, math, Машинное Обучение