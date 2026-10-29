# CardioRiesgo · ACA 1 Aprendizaje Automático (CUN) · Grupo 13

Aplicación en Streamlit que estima la probabilidad de enfermedad coronaria con una regresión logística
entrenada con el dataset UCI Heart Disease (Cleveland, 303 pacientes).

- AUC en validación cruzada: 0.901
- AUC en el conjunto de prueba (61 pacientes): 0.961
- Umbrales: tamizaje 0.29 y balanceado 0.50

## Contenido

- `app.py`: la aplicación.
- `requirements.txt`: librerías necesarias.
- `modelo/modelo_cardio.joblib`: pipeline entrenado (preprocesamiento + regresión logística).
- `modelo/metadata.json`: métricas, umbrales, textos y enlaces que muestra la app.
- `data/heart_clean.csv`: datos limpios con la marca de entrenamiento o prueba.
- `data/plantilla_pacientes.csv`: ejemplo para la evaluación por lote.
- `.streamlit/config.toml`: temas claro y oscuro, colores y tipografías de la interfaz.

## Ejecutar en el computador

```
pip install -r requirements.txt
streamlit run app.py
```

## Publicar en Streamlit Community Cloud

1. Subir el contenido de esta carpeta a un repositorio público de GitHub.
2. Entrar a https://share.streamlit.io, crear una app y elegir el repositorio, la rama `main` y el archivo `app.py`.
3. En App URL escribir `cardioriesgo-aca1-g13`. En Advanced settings elegir Python 3.12.
4. Pulsar Deploy.

Para actualizar los enlaces del video y del repositorio se edita `modelo/metadata.json`, sección `proyecto.enlaces`.

## Integrantes

Daimer Cardona Rojas · Cristhian Eduardo Cortes Martha · Juan Manuel Guerrero Tupaz ·
Luis Alberto Ospina Delgado · Jhonny Mauricio Rosero Carpio

Proyecto académico. El modelo no está validado para uso clínico.
