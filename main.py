import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_prepare_data(filepath):
    """Загрузка данных и их первичная обработка."""
    # Загрузка данных
    sales_dist = pd.read_csv(filepath)
    print("Первые пять записей из датасета:")
    print(sales_dist.head())

    # Переименование столбцов
    sales_dist = sales_dist.rename(columns={'annual net sales': 'sales', 'number of stores in district': 'stores'})
    print("Первые пять записей после переименования столбцов:")
    print(sales_dist.head())

    # Удаление столбца 'district'
    sales = sales_dist.drop('district', axis=1)
    print("Первые пять записей после удаления столбца 'district':")
    print(sales.head())

    return sales


def plot_data(sales):
    """Визуализация данных."""
    y = sales['sales']
    x = sales['stores']

    # График продаж в зависимости от количества магазинов
    plt.figure(figsize=(20, 10))
    plt.scatter(x, y, s=200, color='blue', alpha=0.6, edgecolor='black', linewidth=1, label='Годовые чистые продажи')
    plt.ylabel('Годовые чистые продажи', fontsize=30)
    plt.xlabel('Количество магазинов в районе', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('initial_data_plot.png')

    # Расчет и вывод уравнения линейной регрессии
    m, b = np.polyfit(x, y, 1)
    print(f'Угловой коэффициент линии {m:.2f}.')
    print(f'Точка пересечения с осью Y {b:.2f}.')
    print(f'Уравнение линейной регрессии y = {m:.2f}x + {b:.2f}.')

    y_mean = y.mean()
    x_mean = x.mean()
    print(f'Центроид для этого набора данных x = {x_mean:.2f} и y = {y_mean:.2f}.')

    # Второй график с линейной регрессией
    plt.figure(figsize=(20, 10))
    plt.scatter(x, y, s=200, color='green', alpha=0.6, edgecolor='black', linewidth=1, label='Годовые чистые продажи')
    plt.plot(x_mean, y_mean, 'r^', markersize=30, label='Центроид')
    plt.plot(x, m * x + b, color='purple', linestyle='-', linewidth=4, label='Прямая линейной регрессии')
    plt.xlabel('Количество магазинов в районе', fontsize=30)
    plt.ylabel('Годовые чистые продажи', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.annotate('Центроид', xy=(x_mean, y_mean), xytext=(x_mean - 3, y_mean + 20),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.savefig('regression_line_plot.png')


def predict_sales(query, m, b):
    """Предсказание годовых продаж на основе количества магазинов."""
    if query >= 1:
        prediction = m * query + b
        return prediction
    else:
        print("Для предсказания годовых чистых продаж необходимо иметь хотя бы 1 магазин в районе.")
        return None


# Основной блок кода
if __name__ == "__main__":
    filepath = 'stores-dist.csv'
    sales_data = load_and_prepare_data(filepath)
    plot_data(sales_data)

    # Пример использования функции предсказания
    query = 10  # Пример: предсказание для района с 10 магазинами
    m, b = np.polyfit(sales_data['stores'], sales_data['sales'], 1)  # Вычисление параметров линейной регрессии
    predicted_sales = predict_sales(query, m, b)
    if predicted_sales is not None:
        print(
            f"Прогнозируемые годовые чистые продажи для района с {query} магазинами составляют {predicted_sales:.2f}.")
