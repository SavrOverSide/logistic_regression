import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Заголовок страницы
st.title('Анализ Логистической Регрессии')

# Загрузка файла CSV
uploaded_file = st.file_uploader("Загрузите файл .csv", type=["csv"])

if uploaded_file is not None:
    # Чтение CSV файла
    data = pd.read_csv(uploaded_file)
    st.write("Данные загружены успешно!")
    
    # Выбор столбцов для анализа
    columns = st.multiselect("Выберите столбцы для анализа", data.columns.tolist())
    
    if columns:
        # Отображение выбранных столбцов
        st.write("Выбранные столбцы:", columns)
        target_column = st.selectbox("Выберите целевой столбец", columns)
        
        if target_column:
            # Логистическая регрессия
            X = data[columns].drop(target_column, axis=1)
            y = data[target_column]
            model = LogisticRegression()
            model.fit(X, y)
            
            # Результаты логистической регрессии
            coefficients = model.coef_[0]
            intercept = model.intercept_[0]
            result = {col: coef for col, coef in zip(X.columns, coefficients)}
            st.write("Результаты логистической регрессии (веса столбцов):", result)
            st.write("Свободный член (intercept):", intercept)
            
            # Визуализация
            st.write("Построение scatter plot")
            scatter_x = st.selectbox("Выберите столбец для оси X", columns)
            scatter_y = st.selectbox("Выберите столбец для оси Y", columns)
            
            if scatter_x and scatter_y:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=data, x=scatter_x, y=scatter_y, hue=target_column)
                plt.title(f'Scatter plot: {scatter_x} vs {scatter_y}')
                st.pyplot(plt)