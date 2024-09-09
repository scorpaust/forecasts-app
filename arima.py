import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Preparando os dados (dados do seu array de input)
data = [1.053, 0.971, 0.959, 0.262, 0.208, 0.741, 3.336, 8.584, 6.158, 15.05]
dates = pd.date_range(start='2023-10-01', periods=len(data), freq='MS')
data_series = pd.Series(data, index=dates)

# Tentando ajustar os parâmetros do SARIMA (capturando sazonalidade)
try:
    sarima_model = SARIMAX(data_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Adicionando componente sazonal
    sarima_fit = sarima_model.fit(disp=False)

    # Previsão para os próximos 12 meses (Jan/25 a Dez/25)
    forecast_sarima = sarima_fit.get_forecast(steps=12)
    forecast_index = pd.date_range(start='2025-01-01', periods=12, freq='MS')
    forecast_sarima_values = forecast_sarima.predicted_mean
    forecast_sarima_conf_int = forecast_sarima.conf_int()

    # Evitar valores negativos
    forecast_sarima_values[forecast_sarima_values < 0] = 0  # Limitando valores a zero para evitar negativos

    # Gráfico dos resultados
    plt.figure(figsize=(10, 6))
    plt.plot(data_series, label='Consumo Real (Out/23 a Jul/24)', color='blue')
    plt.plot(forecast_index, forecast_sarima_values, label='Previsão SARIMA (2025)', color='green')
    plt.fill_between(forecast_index, 
                     forecast_sarima_conf_int.iloc[:, 0], 
                     forecast_sarima_conf_int.iloc[:, 1], 
                     color='lightgreen', alpha=0.5)
    plt.title('Previsão SARIMA para Consumo de Chamadas (2025)')
    plt.xlabel('Data')
    plt.ylabel('Consumo (Euros)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Exibindo os valores previstos
    forecast_2025 = pd.DataFrame({'Data': forecast_index, 'Previsão': forecast_sarima_values})
    print(forecast_2025)

except np.linalg.LinAlgError as e:
    print(f"Erro numérico durante o ajuste do modelo: {e}")
