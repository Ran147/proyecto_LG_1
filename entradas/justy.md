# Selección y Diseño de Variables para el Manejo de la Información

## Variables Seleccionadas

El diseño de la base de datos se centra en representar de manera estructurada la información relevante para el análisis del rendimiento académico de los estudiantes. Las variables seleccionadas y su justificación son las siguientes:

### 1. Estudiantes (`students`)
- **id**: Identificador único del estudiante. Permite relacionar registros de manera inequívoca.
- **name**: Nombre del estudiante. Facilita la interpretación y presentación de resultados.
- **program**: Programa académico al que pertenece el estudiante. Permite análisis por carrera o área de estudio.
- **gpa**: Promedio general acumulado. Es un indicador global del rendimiento académico.
- **courses_taken**: Lista de cursos tomados. Permite rastrear la trayectoria académica y analizar el impacto de diferentes cursos en el rendimiento.

### 2. Cursos (`courses`)
- **id**: Identificador único del curso. Esencial para la vinculación con evaluaciones y estudiantes.
- **name**: Nombre del curso. Facilita la interpretación de los resultados.
- **semester** y **academic_year**: Permiten análisis temporales y comparaciones entre cohortes.
- **credits**: Número de créditos. Útil para ponderar el impacto de cada curso en el rendimiento global.
- **professor** y **department**: Permiten análisis por docente o área académica.
- **topics**: Lista de temas cubiertos en el curso, cada uno con:
  - **id**: Identificador del tema.
  - **name**: Nombre del tema.
  - **evaluations**: Evaluaciones asociadas al tema.
- **student_grades**: Calificaciones finales de los estudiantes en el curso. Permite análisis de rendimiento global por curso.

### 3. Evaluaciones (`evaluations`)
- **id**: Identificador único de la evaluación.
- **course_id**: Relaciona la evaluación con un curso específico.
- **type**: Tipo de evaluación (quiz, midterm, final, proyecto, etc.). Permite análisis diferenciados por tipo de instrumento.
- **date**: Fecha de la evaluación. Esencial para análisis temporales y tendencias.
- **topics_covered**: Temas evaluados. Permite análisis de rendimiento por tema.
- **passing_score** y **excellence_threshold**: Umbrales para aprobar y para excelencia. Permiten calcular tasas de éxito y excelencia.
- **student_scores**: Calificaciones de los estudiantes, tanto totales como desglosadas por tema.
- **groups** (opcional): Información sobre grupos en evaluaciones grupales, incluyendo miembros y calificación grupal.

#### Justificación de la Selección de Variables

Las variables fueron seleccionadas para permitir un análisis multivariado y flexible del rendimiento académico, considerando tanto dimensiones individuales (estudiante, curso, tema) como colectivas (grupos, cohortes). El diseño facilita la integración de información y la extracción de métricas relevantes para la toma de decisiones educativas, permitiendo responder preguntas como: ¿qué temas presentan mayor dificultad?, ¿qué estudiantes muestran mayor mejora?, ¿cómo se relacionan los desempeños entre diferentes temas o cursos?, etc.

---

# Selección y Justificación de Estadísticas para Análisis Multivariables

## Análisis Realizados

El archivo `statistical_analysis.ml` implementa los siguientes análisis estadísticos sobre la base de datos:

### 1. Rendimiento por Tema y Curso
- **Qué mide:** Calcula la media, desviación estándar, calificación mínima y máxima para cada tema en cada curso.
- **Justificación:** Permite identificar temas con mayor o menor rendimiento, así como la dispersión de resultados, lo que es útil para detectar áreas de mejora o excelencia.

### 2. Tendencias de Rendimiento de Estudiantes
- **Qué mide:** Analiza la evolución de las calificaciones de cada estudiante a lo largo del tiempo, calculando la pendiente de mejora, calificación inicial, final y mejora total.
- **Justificación:** Permite identificar estudiantes con progresión positiva o negativa, facilitando intervenciones personalizadas y el seguimiento de trayectorias académicas.

### 3. Puntos Críticos de Rendimiento (Dificultad y Tasa de Fracaso por Tema)
- **Qué mide:** Calcula la dificultad de cada tema (basada en la media de calificaciones) y la tasa de fracaso (proporción de calificaciones menores a 60).
- **Justificación:** Identifica temas problemáticos donde los estudiantes tienden a obtener bajas calificaciones, lo que puede guiar ajustes curriculares o refuerzos específicos.

### 4. Correlaciones entre Temas
- **Qué mide:** Calcula la correlación estadística entre las calificaciones obtenidas en pares de temas.
- **Justificación:** Permite descubrir relaciones entre el dominio de diferentes temas, lo que puede indicar dependencias conceptuales o áreas donde el aprendizaje en un tema impacta en otro.

### 5. Tasas de Aprobación y Excelencia por Evaluación
- **Qué mide:** Calcula la proporción de estudiantes que aprueban y que alcanzan la excelencia en cada evaluación.
- **Justificación:** Proporciona métricas clave para evaluar la efectividad de las evaluaciones y el nivel de logro de los estudiantes.

### 6. Análisis de Desempeño Grupal
- **Qué mide:** Para evaluaciones grupales, calcula el promedio, desviación estándar y número de miembros por grupo.
- **Justificación:** Permite analizar la dinámica y el rendimiento de los grupos, identificando posibles desigualdades o sinergias en el trabajo colaborativo.

---

## Justificación General de los Análisis

La selección de estos análisis responde a la necesidad de obtener una visión integral y detallada del rendimiento académico, tanto a nivel individual como colectivo. Se eligieron métricas descriptivas (media, desviación estándar, tasas) y relacionales (correlaciones) que permiten:

- Identificar patrones y tendencias relevantes.
- Detectar áreas de dificultad o excelencia.
- Relacionar el desempeño entre diferentes dimensiones (temas, cursos, estudiantes, grupos).
- Facilitar la toma de decisiones informadas para la mejora continua del proceso educativo.

Estos análisis multivariables son fundamentales para comprender la complejidad del aprendizaje y orientar acciones pedagógicas basadas en evidencia.
