import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Preparando os dados
data = [30.672, 29.158, 23.274, 24.751, 33.08, 23.879, 45.606, 43.994, 21.708, 33.187, 30.672, 29.158]
dates = pd.date_range(start='2023-10-01', periods=len(data), freq='MS')
data_series = pd.Series(data, index=dates)

# Aplicando Holt-Winters sem tendência e sem sazonalidade
hw_model = ExponentialSmoothing(data_series, trend="multiplicative", seasonal=None)
hw_fit = hw_model.fit()

# Previsão para os próximos 12 meses
forecast_hw = hw_fit.forecast(steps=12)
forecast_index = pd.date_range(start='2025-01-01', periods=12, freq='MS')

# Gráfico dos resultados
plt.figure(figsize=(10, 6))
plt.plot(data_series, label='Consumo Real (Out/23 a Jul/24)', color='blue')
plt.plot(forecast_index, forecast_hw, label='Previsão Holt-Winters (2025)', color='green')
plt.title('Previsão Holt-Winters sem Tendência para Consumo de Chamadas (2025)')
plt.xlabel('Data')
plt.ylabel('Consumo (Euros)')
plt.legend()
plt.grid(True)
plt.show()

# Exibindo os valores previstos
forecast_2025_hw = pd.DataFrame({'Data': forecast_index, 'Previsão': forecast_hw})
print(forecast_2025_hw)
