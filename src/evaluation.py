import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

def evaluate_predictions(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }

def graficar_ganancia(df, col_score):
    # Ordenar y calcular (Lógica de negocio intacta)
    df_sorted = df.sort_values(by=col_score, ascending=False).reset_index(drop=True)
    total_registros = len(df_sorted)
    pct_registros = (df_sorted.index + 1) / total_registros
    
    total_probabilidad = df_sorted[col_score].sum()
    prob_acumulada = df_sorted[col_score].cumsum() / total_probabilidad
    
    # 1. Creación de la figura
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # 2. Configuración de FONDO NEGRO
    fig.patch.set_facecolor('#000000') # Fondo externo
    ax.set_facecolor('#000000')        # Fondo interno del gráfico
    
    # 3. Curvas (Usar colores brillantes para el contraste)
    ax.plot(pct_registros, prob_acumulada, 
            label='Modelo Predictivo', 
            color='#00E676', # Verde neón brillante para contraste en negro
            linewidth=2.5)
    
    ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', label='Aleatorio')
    
    # 4. Textos en BLANCO
    ax.set_title('Curva de Ganancia Esperada (Base sin contactar)', color='white', fontsize=14, pad=15)
    ax.set_xlabel('Proporción Acumulada de Registros', color='white', fontsize=12)
    ax.set_ylabel('Probabilidad Acumulada de Conversión', color='white', fontsize=12)
    
    # 5. Formato de ejes (Ticks) en BLANCO
    ax.tick_params(colors='white', labelsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # 6. Bordes del gráfico (Spines) en BLANCO o GRIS
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444') # Gris oscuro para no saturar
    
    # 7. Cuadrícula (Grid) y Leyenda adaptada
    ax.grid(axis='both', linestyle=':', alpha=0.3, color='white')
    
    # Para que la leyenda se vea bien en fondo oscuro
    legend = ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='#111111', edgecolor='#444444')
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    
    return fig

# Forma de uso en Streamlit:
# fig_ganancia = graficar_ganancia_esperada_nuevos(df_cargado, 'score_normalizado')
# st.pyplot(fig_ganancia)

def graficar_ganancia_comparativa(df, col_score, col_target):
    """
    Genera un gráfico de ganancia acumulada comparando la predicción vs la realidad.
    df: DataFrame de pandas.
    col_score: String con el nombre de la columna predictiva (ej. 'score_normalizado').
    col_target: String con el nombre de la columna real (ej. 'contacto_positivo').
    """
    
    # 1. Ordenar el DataFrame priorizando los mejores scores
    df_sorted = df.sort_values(by=col_score, ascending=False).reset_index(drop=True)
    
    # 2. Calcular Eje X: Proporción acumulada de registros contactados
    total_registros = len(df_sorted)
    pct_registros = (df_sorted.index + 1) / total_registros
    
    # 3. Calcular Eje Y (Esperado): Probabilidad acumulada (Score Predictivo)
    total_probabilidad = df_sorted[col_score].sum()
    ganancia_esperada = df_sorted[col_score].cumsum() / total_probabilidad
    
    # 4. Calcular Eje Y (Real): Conversiones acumuladas reales
    total_conversiones = df_sorted[col_target].sum()
    ganancia_real = df_sorted[col_target].cumsum() / total_conversiones
    
    # 5. Creación de la figura y configuración de FONDO NEGRO
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#000000') 
    ax.set_facecolor('#000000')        
    
    # 6. Trazar las Curvas
    # Curva 1: Ganancia Real (Línea sólida y brillante)
    ax.plot(pct_registros, ganancia_real, 
            label='Ganancia Real (Contactos Positivos)', 
            color='#00E676', # Verde neón brillante
            linewidth=3)
            
    # Curva 2: Ganancia Esperada (Línea punteada para diferenciar estimación)
    ax.plot(pct_registros, ganancia_esperada, 
            label='Ganancia Esperada (Score del Modelo)', 
            color='#29B6F6', # Azul claro / Celeste
            linestyle='-.', 
            linewidth=2)
    
    # Curva 3: Línea Base (Selección Aleatoria)
    ax.plot([0, 1], [0, 1], color='#888888', linestyle='--', label='Selección Aleatoria sin IA')
    
    # 7. Textos y Ejes en BLANCO
    ax.set_title('Comparativa de Rendimiento: Ganancia Esperada vs. Real', color='white', fontsize=14, pad=15)
    ax.set_xlabel('Proporción Acumulada de Leads Contactados', color='white', fontsize=12)
    ax.set_ylabel('Proporción Acumulada de Éxitos', color='white', fontsize=12)
    
    # Formato de porcentaje para los ejes
    ax.tick_params(colors='white', labelsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Bordes (Spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444') 
    
    # Cuadrícula y Leyenda
    ax.grid(axis='both', linestyle=':', alpha=0.3, color='white')
    legend = ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='#111111', edgecolor='#444444')
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    
    return fig

# Forma de uso en tu módulo de Streamlit:
# fig_comparativa = graficar_ganancia_comparativa_dark(df_historico, 'score_normalizado', 'contacto_positivo')
# st.pyplot(fig_comparativa)
