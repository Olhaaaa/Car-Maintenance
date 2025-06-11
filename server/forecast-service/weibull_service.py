import numpy as np
import pandas as pd
from lifelines import WeibullFitter
import matplotlib.pyplot as plt
from scipy.special import gamma

class WeibullForecastService:
    def __init__(self, times, events):
        """
        Ініціалізація сервісу для прогнозування пробігу відмови.
        :param times: Список пробігів (км).
        :param events: Список статусів (1 = відмова, 0 = цензуровано).
        """
        if not times or not events or len(times) != len(events):
            raise ValueError("Вхідні дані порожні або мають різну довжину")
        if sum(events) == 0:
            raise ValueError("Немає нецензурованих подій для оцінки параметрів")

        self.data = pd.DataFrame({'Time': times, 'Event': events})
        self.wf = WeibullFitter()
        self.wf.fit(durations=self.data['Time'], event_observed=self.data['Event'])
        self.beta = self.wf.rho_  # Параметр форми
        self.eta = self.wf.lambda_  # Параметр масштабу

        if self.beta <= 0 or self.eta <= 0:
            raise ValueError("Некоректні параметри Вейбулла: beta або eta <= 0")

    def calculate_failure_probability(self, t):
        """
        Обчислення ймовірності відмови F(t).
        :param t: Пробіг (км).
        :return: F(t).
        """
        if t < 0:
            raise ValueError("Пробіг не може бути від'ємним")
        return 1 - np.exp(-((t / self.eta) ** self.beta))

    def calculate_density(self, t):
        """
        Обчислення щільності розподілу f(t).
        :param t: Пробіг (км).
        :return: f(t).
        """
        if t < 0:
            raise ValueError("Пробіг не може бути від'ємним")
        return (self.beta / self.eta) * ((t / self.eta) ** (self.beta - 1)) * np.exp(-((t / self.eta) ** self.beta))

    def calculate_theoretical_variance(self):
        """
        Обчислення теоретичної дисперсії розподілу Вейбулла.
        :return: Дисперсія.
        """
        return self.eta**2 * (gamma(1 + 2/self.beta) - (gamma(1 + 1/self.beta))**2)

    def find_peak_replacements(self, bin_width=None):
        """
        Знаходження піку замін у нецензурованих даних.
        :param bin_width: Ширина бінів для гістограми (км). Якщо None, обирається автоматично.
        :return: Пробіг піку та відповідна ймовірність F(t).
        """
        failures = self.data[self.data['Event'] == 1]['Time']
        if len(failures) < 5:
            # Використовуємо моду Вейбулла як оцінку піку
            peak_mileage = self.eta * ((self.beta - 1) / self.beta) ** (1 / self.beta) if self.beta > 1 else self.eta
            p = self.calculate_failure_probability(peak_mileage)
            return peak_mileage, p

        # Адаптивна ширина бінів (правило Фрідмана-Діаконіса)
        if bin_width is None:
            iqr = np.percentile(failures, 75) - np.percentile(failures, 25)
            bin_width = 2 * iqr * len(failures) ** (-1/3) if iqr > 0 else 1000

        hist, bins = np.histogram(failures, bins=np.arange(0, max(failures) + bin_width, bin_width))
        peak_bin = bins[np.argmax(hist)]
        peak_mileage = peak_bin + bin_width / 2  # Середина біна
        p = self.calculate_failure_probability(peak_mileage)
        return peak_mileage, p

    def calculate_variance(self):
        """
        Обчислення дисперсії. Використовує емпіричну дисперсію, якщо достатньо даних, інакше теоретичну.
        :return: Дисперсія.
        """
        failures = self.data[self.data['Event'] == 1]['Time']
        if len(failures) > 5:
            return np.var(failures, ddof=1)
        return self.calculate_theoretical_variance()

    def select_dynamic_p(self, peak_mileage, base_p):
        """
        Динамічний вибір відсотка ймовірності відмови p.
        :param peak_mileage: Пробіг піку замін.
        :param base_p: Базова ймовірність F(peak_mileage).
        :return: Адаптований p.
        """
        variance = self.calculate_variance()
        median_failure = np.median(self.data[self.data['Event'] == 1]['Time']) if sum(self.data['Event']) > 0 else self.eta
        variance_threshold = (0.1 * median_failure) ** 2  # Поріг залежить від медіани

        p = base_p
        if variance > variance_threshold:
            p = min(p + 0.05, 0.85)
        if base_p < 0.7:
            p = max(0.65, base_p)  # Злегка підвищено нижню межу для консервативності

        return p

    def predict_failure_mileage(self, p):
        """
        Прогнозування пробігу відмови для заданого p.
        :param p: Ймовірність відмови F(t).
        :return: Прогнозований пробіг (км).
        """
        if not 0 < p < 1:
            raise ValueError("Ймовірність p має бути в межах (0, 1)")
        return self.eta * (-np.log(1 - p)) ** (1 / self.beta)

    def plot_distribution(self, current_mileage, recommended_mileage, forecast_mileage,
                         output_file='weibull_distribution.png'):
        """
        Побудова графіку розподілу.
        :param current_mileage: Поточний пробіг (км).
        :param recommended_mileage: Рекомендація виробника (км).
        :param forecast_mileage: Прогнозований пробіг відмови (км).
        :param output_file: Ім'я файлу для збереження графіку.
        """
        if current_mileage < 0 or recommended_mileage < 0 or forecast_mileage < 0:
            raise ValueError("Пробіг не може бути від'ємним")

        # Адаптивний діапазон для графіку
        failures = self.data[self.data['Event'] == 1]['Time']
        max_range = max(max(failures, default=50000), forecast_mileage, recommended_mileage) * 1.5
        t = np.linspace(0, max_range, 1000)
        density = [self.calculate_density(ti) for ti in t]

        # Побудова графіку
        plt.figure(figsize=(10, 6))

        # Гістограма нецензурованих даних
        if len(failures) > 0:
            # Адаптивна кількість бінів
            iqr = np.percentile(failures, 75) - np.percentile(failures, 25)
            bin_width = 2 * iqr * len(failures) ** (-1/3) if iqr > 0 else 1000
            num_bins = int(max_range / bin_width)
            plt.hist(failures, bins=num_bins, density=True, alpha=0.5, label='Емпірична гістограма (відмови)', color='skyblue')

        # Щільність вейбулівського розподілу
        plt.plot(t, density, 'r-', label='Вейбулівський розподіл', linewidth=2)

        # Позначки
        # # plt.axvline(current_mileage, color='green', linestyle='--', label=f'Поточний пробіг ({current_mileage:,} км)')
        # plt.axvline(recommended_mileage, color='blue', linestyle='--',
        #             label=f'Рекомендація ({recommended_mileage:,} км)')
        plt.axvline(forecast_mileage, color='purple', linestyle='--',
                    label=f'Прогноз відмови ({forecast_mileage:,} км)')

        # Налаштування графіку
        plt.title('Вейбулівський розподіл пробігу до відмови')
        plt.xlabel('Пробіг (км)')
        plt.ylabel('Щільність')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Збереження та відображення
        plt.savefig(output_file)
        plt.close()

    def generate_forecast(self, current_mileage, recommended_mileage=20000):
        """
        Генерація прогнозу для інтерфейсу.
        :param current_mileage: Поточний пробіг (км).
        :param recommended_mileage: Рекомендація виробника (км).
        :return: Словник із даними для слайдбара.
        """
        try:
            if current_mileage < 0:
                raise ValueError("Поточний пробіг не може бути від'ємним")

            # Знаходження піку замін
            peak_mileage, base_p = self.find_peak_replacements()

            # Динамічний вибір p
            p = self.select_dynamic_p(peak_mileage, base_p)

            # Прогнозований пробіг
            forecast_mileage = self.predict_failure_mileage(p)

            # Залишковий пробіг
            residual_mileage = max(0, forecast_mileage - current_mileage)

            # Побудова графіку
            self.plot_distribution(current_mileage, recommended_mileage, forecast_mileage)

            return {
                'current_mileage': current_mileage,
                'recommended_mileage': recommended_mileage,
                'forecast_mileage': round(forecast_mileage),
                'residual_mileage': round(residual_mileage),
                'probability': p,
                'message': (
                    f"Прогнозований пробіг відмови: {round(forecast_mileage):,} км "
                ),
                'beta': self.beta,
                'eta': self.eta,
                'peak_mileage': peak_mileage,
                'variance': self.calculate_variance()
            }
        except Exception as e:
            raise ValueError(f"Помилка при генерації прогнозу: {str(e)}")