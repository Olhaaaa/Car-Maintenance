import numpy as np
from weibull_service import WeibullForecastService


def generate_synthetic_data():
    """
    Генерація синтетичних даних для гальмівних колодок Daewoo Lanos із піком замін близько 25,000 км.
    :return: times, events
    """
    # Повні дані: 38 подій
    failures = [
        45000, 16000, 17000, 35000, 19000, 36000, 34500, 41000, 22000, 34600,
        23000, 38000, 34000, 43000, 35000, 36000, 25000, 37000, 26000, 27000,
        34000, 38000, 38000, 42200, 29000, 30000, 37000, 31000, 32000, 32000,
        33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000
    ]
    # Цензуровані дані: 18 подій
    censored = [
        30000, 26000, 33000, 12000, 36000, 37000, 38000, 12000, 34000, 25000,
        42000, 43000, 26000, 14000, 46000, 37000, 38000, 29000
    ]


    times = failures + censored
    events = [1] * len(failures) + [0] * len(censored)
    return times, events


def run_tests():
    """
    Запуск тестів для перевірки алгоритму.
    """
    times, events = generate_synthetic_data()

    # Тестові сценарії
    test_cases = [
        {
            'name': 'Поточний пробіг 15,000 км',
            'current_mileage': 15000,
            'recommended_mileage': 20000
        },
        {
            'name': 'Поточний пробіг 18,000 км',
            'current_mileage': 18000,
            'recommended_mileage': 20000
        }
    ]

    for test_case in test_cases:
        print(f"\n=== Прогнозування заміни гальмівних колодок")
        service = WeibullForecastService(times, events)
        forecast = service.generate_forecast(
            current_mileage=test_case['current_mileage'],
            recommended_mileage=test_case['recommended_mileage']
        )

        print(f"Бета (форма): {forecast['beta']:.2f}")
        print(f"Ета (масштаб): {forecast['eta']:.0f} км")
        print(f"Пік замін: {forecast['peak_mileage']:.0f} км")
        print(f"Ймовірність відмови (p): {forecast['probability']:.0%}")
        print(f"Прогноз: {forecast['message']}")
        print(f"Графік збережено як weibull_distribution.png")


if __name__ == '__main__':
    run_tests()