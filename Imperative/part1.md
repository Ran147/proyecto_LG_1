# Variables utilizadas en el proyecto y su uso

## Variables de los datos académicos

- **student_id**: Identificador único de cada estudiante. Se utiliza para relacionar información personal, inscripciones y resultados académicos.
- **name**: Nombre del estudiante. Solo para identificación en los datos.
- **cumulative_gpa**: Promedio general acumulado del estudiante. Se usa como variable dependiente o independiente en análisis estadísticos.
- **course_id**: Identificador único de cada curso. Relaciona inscripciones, evaluaciones y temas.
- **title**: Nombre del curso. Solo para referencia.
- **topics**: Lista de temas cubiertos en cada curso. Cada tema tiene un topic_id y un nombre.
- **topic_id**: Identificador único de cada tema. Se usa para análisis de desempeño por tema y correlaciones.
- **enrollment_id**: Identificador único de cada inscripción de estudiante en un curso. Relaciona estudiante, curso, evaluaciones y resultados.
- **semester**: Semestre en el que se cursa la materia. Se usa para análisis de series temporales.
- **year**: Año en el que se cursa la materia. Se usa para análisis de series temporales.
- **attendance_percentage**: Porcentaje de asistencia del estudiante en el curso. Variable relevante en análisis de regresión y clusters.
- **final_grade**: Calificación final obtenida en el curso. Variable dependiente en varios análisis.
- **evaluations**: Lista de evaluaciones realizadas en el curso (quiz, assignment, midterm, project, final).
- **evaluation_id**: Identificador único de cada evaluación.
- **type**: Tipo de evaluación (quiz, assignment, midterm, project, final). Permite análisis por tipo de evaluación.
- **weight**: Peso de la evaluación en la calificación final.
- **time_taken**: Momento en que se realiza la evaluación (early, mid, late, final). Útil para análisis de desempeño temprano.
- **score**: Calificación obtenida en la evaluación.
- **topic_scores**: Lista de calificaciones por tema en cada evaluación.

## Variables en los análisis estadísticos

- **feature_importance**: Importancia de cada variable independiente en la regresión lineal múltiple.
- **model_metrics**: Métricas de desempeño del modelo (R-squared, error cuadrático medio, etc.).
- **variable_influence**: Influencia y significancia estadística de cada variable en la regresión.
- **logistic_regression_data**: Datos usados para la regresión logística (desempeño temprano, GPA, asistencia, aprobado/reprobado).
- **cluster_data**: Datos agrupados para análisis de clusters (asistencia, calificación final, desempeño por tema).
- **time_series_data**: Datos agregados por period (semestre/año) para análisis de tendencias.
- **correlation_data**: Matriz de correlaciones entre temas.

Estas variables permiten realizar análisis estadísticos, visualizaciones y obtener conclusiones sobre el desempeño académico de los estudiantes.

---

## Análisis Multivariable

El análisis multivariable realizado en el proyecto corresponde principalmente a la **regresión lineal múltiple** para predecir el desempeño académico (calificación final) a partir de varias variables independientes.

### Variables utilizadas:
- **Calificación final** (*final_grade*): Variable dependiente, objetivo de la predicción.
- **Promedio de quizzes** (*avg_quiz_score*): Variable independiente.
- **Promedio de tareas** (*avg_assignment_score*): Variable independiente.
- **Promedio de exámenes parciales** (*avg_midterm_score*): Variable independiente.
- **Promedio de proyectos** (*avg_project_score*): Variable independiente.
- **Promedio de examen final** (*avg_final_score*): Variable independiente.
- **GPA acumulado** (*cumulative_gpa*): Variable independiente.
- **Porcentaje de asistencia** (*attendance_percentage*): Variable independiente.

### Insights esperados:
- Identificar cuáles variables tienen mayor peso o importancia en la predicción del desempeño final del estudiante.
- Determinar si la asistencia, el GPA previo o el desempeño en ciertos tipos de evaluación (quizzes, tareas, exámenes, proyectos) son predictores significativos.
- Detectar relaciones no evidentes entre variables (por ejemplo, si la asistencia tiene un impacto menor al esperado).
- Obtener métricas de ajuste del modelo (R², error cuadrático medio) para evaluar la calidad de la predicción.
- Proveer recomendaciones basadas en los resultados, como enfocar esfuerzos en mejorar el desempeño en variables con mayor influencia.

---

## Otros Análisis Estadísticos

### 1. Regresión Logística
**Variables utilizadas:**
- **Desempeño temprano** (*early_performance*): Promedio de evaluaciones realizadas al inicio del curso.
- **GPA acumulado** (*cumulative_gpa*): Promedio general previo del estudiante.
- **Porcentaje de asistencia** (*attendance_percentage*): Asistencia al curso.
- **Aprobado/Reprobado** (*passed*): Variable dependiente (1 si aprobó, 0 si reprobó).

**Insights esperados:**
- Determinar qué tan predictivo es el desempeño temprano y la asistencia para identificar estudiantes en riesgo de reprobar.
- Identificar patrones que permitan intervenciones tempranas.
- Visualizar la distribución de aprobados y reprobados según variables clave.

### 2. Análisis de Clusters
**Variables utilizadas:**
- **Porcentaje de asistencia** (*attendance_percentage*).
- **Calificación final** (*final_grade*).
- **Desempeño por tema** (*topic_scores*): Promedio de calificaciones por tema.

**Insights esperados:**
- Identificar grupos de estudiantes con patrones similares de desempeño y asistencia.
- Detectar perfiles de estudiantes (por ejemplo, alto desempeño y alta asistencia vs. bajo desempeño y baja asistencia).
- Analizar el desempeño por tema dentro de cada cluster.

### 3. Análisis de Series Temporales
**Variables utilizadas:**
- **Año** (*year*) y **semestre** (*semester*).
- **Calificación final promedio** (*avg_final_grade*).
- **Asistencia promedio** (*avg_attendance*).
- **Desempeño por tema a lo largo del tiempo** (*topic_scores*).

**Insights esperados:**
- Observar tendencias en el desempeño académico y la asistencia a lo largo de los periodos.
- Detectar mejoras o retrocesos en temas específicos a través del tiempo.
- Apoyar la toma de decisiones para ajustar planes de estudio o intervenciones.

### 4. Heatmap de Correlación
**Variables utilizadas:**
- **Desempeño por tema** (*topic_scores*): Calificaciones obtenidas en cada tema.

**Insights esperados:**
- Visualizar la fuerza y dirección de la relación entre pares de temas.
- Identificar temas que suelen estar correlacionados positiva o negativamente.
- Detectar áreas donde el refuerzo de un tema puede impactar en el desempeño de otros.

---

## Estructura del JSON y Resumen de Datos

El archivo `academic-performance-mock-data.json` contiene datos simulados de rendimiento académico estructurados en tres colecciones principales:

### 1. Estudiantes (`students`)
- Contiene información básica de cada estudiante
- Cada estudiante tiene un ID único, nombre y GPA acumulado
- El conjunto de datos incluye 5 estudiantes con GPAs que van desde 2.8 hasta 3.9

### 2. Cursos (`courses`)
- Define los cursos disponibles en el sistema
- Cada curso tiene un ID único, título y lista de temas
- Los temas están organizados jerárquicamente dentro de cada curso
- El conjunto de datos incluye 5 cursos (CS101, CS201, MATH202, CS301, CS401)

### 3. Inscripciones (`enrollments`)
- Registra la información de cada estudiante inscrito en un curso
- Cada inscripción tiene un ID único, referencias al estudiante y al curso, semestre, año, porcentaje de asistencia y calificación final
- Contiene evaluaciones detalladas para cada inscripción
- Las evaluaciones incluyen diferentes tipos (quiz, assignment, midterm, project, final)
- Cada evaluación tiene un peso específico, momento de realización y calificación
- Las evaluaciones también incluyen calificaciones por tema, permitiendo análisis granular del desempeño

### Relaciones entre entidades
- Los estudiantes se relacionan con cursos a través de las inscripciones
- Las inscripciones conectan estudiantes, cursos y evaluaciones
- Las evaluaciones se relacionan con temas específicos del curso a través de las calificaciones por tema

### Resumen de datos
- 5 estudiantes
- 5 cursos
- 11 inscripciones
- 55 evaluaciones (5 por inscripción)
- 15 temas distribuidos entre los cursos
- Período de datos: Otoño 2023 - Primavera 2024

Esta estructura jerárquica permite realizar análisis multidimensionales, como los descritos en las secciones anteriores, facilitando la identificación de patrones y correlaciones en el rendimiento académico.
