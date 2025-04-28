import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Configurar estilo para todos los gráficos
plt.style.use('ggplot')
sns.set_palette("viridis")

class VisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualización de Análisis de Rendimiento Académico")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Load data
        self.data = self.load_data()
        if not self.data:
            messagebox.showerror("Error", "Error al cargar los datos. Por favor, verifique la ruta del archivo.")
            root.destroy()
            return
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create title
        title_label = ttk.Label(
            self.main_frame, 
            text="Visualización de Análisis de Rendimiento Académico", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Create menu frame
        self.menu_frame = ttk.LabelFrame(self.main_frame, text="Opciones de Análisis", padding="10")
        self.menu_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create buttons for each analysis
        ttk.Button(
            self.menu_frame, 
            text="1. Análisis de Regresión Lineal Múltiple", 
            command=self.show_multiple_linear_regression
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            self.menu_frame, 
            text="2. Análisis de Regresión Logística", 
            command=self.show_logistic_regression
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            self.menu_frame, 
            text="3. Análisis de Clusters", 
            command=self.show_cluster_analysis
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            self.menu_frame, 
            text="4. Análisis de Series Temporales", 
            command=self.show_time_series_analysis
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            self.menu_frame, 
            text="5. Análisis de Correlación", 
            command=self.show_correlation_heatmap
        ).pack(fill=tk.X, pady=5)
        
        # Create visualization frame
        self.viz_frame = ttk.LabelFrame(self.main_frame, text="Visualización", padding="10")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create text frame for displaying statistics
        self.text_frame = ttk.LabelFrame(self.main_frame, text="Estadísticas", padding="10")
        self.text_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.text_widget = tk.Text(self.text_frame, height=5, wrap=tk.WORD)
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Initialize current figure
        self.current_figure = None
        self.current_canvas = None
    
    def load_data(self):
        """Load the statistical analysis data from the JSON file."""
        try:
            with open('salidas/statistical-analysis-results.json', 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print("Error: statistical-analysis-results.json file not found.")
            return None
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in the file.")
            return None
    
    def clear_visualization(self):
        """Clear the current visualization."""
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None
        if self.current_figure:
            plt.close(self.current_figure)
            self.current_figure = None
        self.text_widget.delete(1.0, tk.END)
    
    def create_figure(self, figsize=(10, 6)):
        """Create a new figure and canvas."""
        self.clear_visualization()
        self.current_figure = Figure(figsize=figsize, dpi=100)
        self.current_canvas = FigureCanvasTkAgg(self.current_figure, master=self.viz_frame)
        self.current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return self.current_figure
    
    def update_text(self, text):
        """Update the text widget with statistics."""
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, text)
    
    def clear_viz_frame(self):
        """Remove all widgets from the visualization frame."""
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
    
    def show_multiple_linear_regression(self):
        """Show multiple linear regression analysis options."""
        self.clear_visualization()
        self.clear_viz_frame()
        
        # Create options frame
        options_frame = ttk.Frame(self.viz_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(options_frame, text="Seleccionar visualización:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for visualization options
        options = ["Importancia de Características", "Métricas del Modelo", "Influencia de Variables"]
        selected_option = tk.StringVar()
        selected_option.set(options[0])
        
        dropdown = ttk.Combobox(options_frame, textvariable=selected_option, values=options, state="readonly")
        dropdown.pack(side=tk.LEFT, padx=5)
        
        # Function to update visualization based on selection
        def update_viz():
            option = selected_option.get()
            if option == "Importancia de Características":
                self.plot_feature_importance()
            elif option == "Métricas del Modelo":
                self.plot_model_metrics()
            elif option == "Influencia de Variables":
                self.plot_variable_influence()
        
        # Create button to apply selection
        ttk.Button(options_frame, text="Mostrar", command=update_viz).pack(side=tk.LEFT, padx=5)
        
        # Show default visualization
        self.plot_feature_importance()
    
    def plot_feature_importance(self):
        """Plot feature importance for multiple linear regression."""
        fig = self.create_figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Extract feature importance data
        feature_importance = self.data['multiple_linear_regression']['model']['feature_importance']
        
        # Sort by standardized coefficient for better visualization
        feature_importance.sort(key=lambda x: x['standardized_coefficient'], reverse=True)
        
        # Create a bar chart of feature importance
        features = [item['feature'] for item in feature_importance]
        coefficients = [item['standardized_coefficient'] for item in feature_importance]
        
        bars = ax.barh(features, coefficients, color='skyblue')
        
        # Add value labels on the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center')
        
        ax.set_title('Importancia de Características en la Predicción de Calificación Final', fontsize=16)
        ax.set_xlabel('Coeficiente Estandarizado', fontsize=14)
        ax.set_ylabel('Característica', fontsize=14)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Análisis de Importancia de Características: Este gráfico muestra la importancia relativa de cada característica en la predicción de la calificación final. Los coeficientes estandarizados más altos indican una influencia más fuerte.")
    
    def plot_model_metrics(self):
        """Plot model metrics for multiple linear regression."""
        fig = self.create_figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Extract metrics
        metrics = self.data['multiple_linear_regression']['model']['metrics']
        
        # Create a bar chart of metrics
        metric_names = ['R-cuadrado', 'R-cuadrado Ajustado', 'Error Cuadrático Medio']
        metric_values = [metrics['r_squared'], metrics['adjusted_r_squared'], metrics['mean_squared_error']]
        
        bars = ax.bar(metric_names, metric_values, color=['green', 'blue', 'red'])
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        ax.set_title('Métricas de Rendimiento del Modelo', fontsize=16)
        ax.set_ylabel('Valor', fontsize=14)
        ax.set_ylim(0, max(metric_values) * 1.2)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text(f"Métricas de Rendimiento del Modelo:\nR-cuadrado: {metrics['r_squared']:.4f}\nR-cuadrado Ajustado: {metrics['adjusted_r_squared']:.4f}\nError Cuadrático Medio: {metrics['mean_squared_error']:.4f}\nTamaño de la Muestra: {metrics['sample_size']}")
    
    def plot_variable_influence(self):
        """Plot variable influence for multiple linear regression."""
        fig = self.create_figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Extract variable influence data
        variable_influence = self.data['multiple_linear_regression']['model']['variable_influence']
        
        # Create a table-like visualization
        variables = [var['variable'] for var in variable_influence]
        coefficients = [var['coefficient'] for var in variable_influence]
        p_values = [var['p_value'] for var in variable_influence]
        significance = [var['significance'] for var in variable_influence]
        
        # Create a bar chart of coefficients
        bars = ax.barh(variables, coefficients, color='lightblue')
        
        # Add value labels on the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f} (p={p_values[i]})', ha='left', va='center')
        
        ax.set_title('Influencia de Variables en la Calificación Final', fontsize=16)
        ax.set_xlabel('Coeficiente', fontsize=14)
        ax.set_ylabel('Variable', fontsize=14)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        text = "Influencia de Variables:\n"
        for var in variable_influence:
            text += f"{var['variable']}: {var['significance']} (p-value: {var['p_value']})\n"
        self.update_text(text)
    
    def show_logistic_regression(self):
        """Show logistic regression analysis options."""
        self.clear_visualization()
        self.clear_viz_frame()
        
        # Create options frame
        options_frame = ttk.Frame(self.viz_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(options_frame, text="Seleccionar visualización:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for visualization options
        options = ["Gráfico de Dispersión", "Estadísticas Resumen"]
        selected_option = tk.StringVar()
        selected_option.set(options[0])
        
        dropdown = ttk.Combobox(options_frame, textvariable=selected_option, values=options, state="readonly")
        dropdown.pack(side=tk.LEFT, padx=5)
        
        # Function to update visualization based on selection
        def update_viz():
            option = selected_option.get()
            if option == "Gráfico de Dispersión":
                self.plot_logistic_scatter()
            elif option == "Estadísticas Resumen":
                self.plot_logistic_summary()
        
        # Create button to apply selection
        ttk.Button(options_frame, text="Mostrar", command=update_viz).pack(side=tk.LEFT, padx=5)
        
        # Show default visualization
        self.plot_logistic_scatter()
    
    def plot_logistic_scatter(self):
        """Plot 3D scatter plot for logistic regression."""
        fig = self.create_figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract logistic regression data
        lr_data = self.data['logistic_regression']['logistic_regression_data']
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(lr_data)
        
        # Create a 3D scatter plot
        scatter = ax.scatter(
            df['early_performance'],  # x-axis: early performance
            df['cumulative_gpa'],     # y-axis: cumulative GPA
            df['attendance_percentage'],  # z-axis: attendance percentage
            c=df['passed'],           # color: passed (0 or 1)
            cmap='viridis',
            s=100,
            alpha=0.7
        )
        
        # Add a colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Aprobado (1) / Reprobado (0)', fontsize=12)
        
        # Set labels
        ax.set_xlabel('Calificación Inicial', fontsize=12)
        ax.set_ylabel('Promedio Acumulado', fontsize=12)
        ax.set_zlabel('Porcentaje de Asistencia', fontsize=12)
        
        # Set title
        ax.set_title('Visualización 3D de Factores de Rendimiento Estudiantil', fontsize=14)
        
        # Adjust the viewing angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Visualización 3D de Regresión Logística: Este gráfico muestra la relación entre la calificación inicial, el promedio acumulado, el porcentaje de asistencia y los resultados de aprobación/reprobación. El color indica si el estudiante aprobó (1) o reprobó (0).")
    
    def plot_logistic_summary(self):
        """Plot summary statistics for logistic regression."""
        fig = self.create_figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Extract logistic regression data
        lr_data = self.data['logistic_regression']['logistic_regression_data']
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(lr_data)
        
        # Calculate summary statistics
        total_students = len(df)
        passing_students = df['passed'].sum()
        failing_students = total_students - passing_students
        pass_rate = (passing_students / total_students) * 100
        
        # Create a pie chart of pass/fail distribution
        labels = ['Aprobado', 'Reprobado']
        sizes = [passing_students, failing_students]
        colors = ['lightgreen', 'lightcoral']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Distribución de Aprobados/Reprobados', fontsize=16)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        text = f"Estadísticas Resumen:\nTotal de Estudiantes: {total_students}\n"
        text += f"Estudiantes Aprobados: {passing_students}\n"
        text += f"Estudiantes Reprobados: {failing_students}\n"
        text += f"Tasa de Aprobación: {pass_rate:.2f}%\n\n"
        
        text += "Valores Promedio:\nEstudiantes Aprobados:\n"
        text += f"  Rendimiento Inicial: {passing_avg['early_performance']:.2f}\n"
        text += f"  Promedio Acumulado: {passing_avg['cumulative_gpa']:.2f}\n"
        text += f"  Asistencia: {passing_avg['attendance_percentage']:.2f}%\n"
        
        if not failing_avg.empty:
            text += "\nEstudiantes Reprobados:\n"
            text += f"  Rendimiento Inicial: {failing_avg['early_performance']:.2f}\n"
            text += f"  Promedio Acumulado: {failing_avg['cumulative_gpa']:.2f}\n"
            text += f"  Asistencia: {failing_avg['attendance_percentage']:.2f}%"
        
        self.update_text(text)
    
    def show_cluster_analysis(self):
        """Show cluster analysis options."""
        self.clear_visualization()
        self.clear_viz_frame()
        
        # Create options frame
        options_frame = ttk.Frame(self.viz_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(options_frame, text="Seleccionar visualización:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for visualization options
        options = ["Asistencia vs. Calificación Final", "Mapa de Calor de Rendimiento por Tema", "Distribución de Rendimiento"]
        selected_option = tk.StringVar()
        selected_option.set(options[0])
        
        dropdown = ttk.Combobox(options_frame, textvariable=selected_option, values=options, state="readonly")
        dropdown.pack(side=tk.LEFT, padx=5)
        
        # Function to update visualization based on selection
        def update_viz():
            option = selected_option.get()
            if option == "Asistencia vs. Calificación Final":
                self.plot_attendance_vs_grade()
            elif option == "Mapa de Calor de Rendimiento por Tema":
                self.plot_topic_performance_heatmap()
            elif option == "Distribución de Rendimiento":
                self.plot_performance_distribution()
        
        # Create button to apply selection
        ttk.Button(options_frame, text="Mostrar", command=update_viz).pack(side=tk.LEFT, padx=5)
        
        # Show default visualization
        self.plot_attendance_vs_grade()
    
    def plot_attendance_vs_grade(self):
        """Plot attendance vs. final grade for cluster analysis."""
        fig = self.create_figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Extract cluster data
        cluster_data = self.data['cluster_analysis']['cluster_data']
        
        # Convert to DataFrame
        df = pd.DataFrame(cluster_data)
        
        # Create a scatter plot of attendance vs final grade
        scatter = ax.scatter(df['attendance_percentage'], df['final_grade'], 
                           s=100, alpha=0.7, c=df['final_grade'], cmap='viridis')
        
        fig.colorbar(scatter, ax=ax, label='Calificación Final')
        ax.set_title('Asistencia vs. Calificación Final', fontsize=16)
        ax.set_xlabel('Porcentaje de Asistencia', fontsize=14)
        ax.set_ylabel('Calificación Final', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Asistencia vs. Calificación Final: Este gráfico de dispersión muestra la relación entre el porcentaje de asistencia y la calificación final. El gradiente de color indica el valor de la calificación final.")
    
    def plot_topic_performance_heatmap(self):
        """Plot topic performance heatmap for cluster analysis."""
        fig = self.create_figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Extract cluster data
        cluster_data = self.data['cluster_analysis']['cluster_data']
        
        # Extract all topic scores for each student
        topic_scores = []
        for student in cluster_data:
            for topic in student['topic_scores']:
                if topic['avg_score'] > 0:  # Only include non-zero scores
                    topic_scores.append({
                        'student_id': student['student_id'],
                        'topic_id': topic['topic_id'],
                        'score': topic['avg_score']
                    })
        
        # Convert to DataFrame
        topic_df = pd.DataFrame(topic_scores)
        
        # Create a heatmap of average scores by topic
        if not topic_df.empty:
            pivot_table = topic_df.pivot_table(
                values='score', 
                index='student_id', 
                columns='topic_id', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.1f', ax=ax)
            ax.set_title('Rendimiento por Tema', fontsize=16)
            ax.set_xlabel('ID del Tema', fontsize=14)
            ax.set_ylabel('ID del Estudiante', fontsize=14)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Rendimiento por Tema: Este mapa de calor muestra el rendimiento de cada estudiante en diferentes temas. Los colores más oscuros indican mayores puntuaciones.")
    
    def plot_performance_distribution(self):
        """Plot performance distribution for cluster analysis."""
        fig = self.create_figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Extract cluster data
        cluster_data = self.data['cluster_analysis']['cluster_data']
        
        # Convert to DataFrame
        df = pd.DataFrame(cluster_data)
        
        # Group students by performance level
        performance_levels = ['Reprobado', 'Debajo del Promedio', 'Promedio', 'Arriba del Promedio', 'Excelente']
        df['performance_level'] = pd.cut(
            df['final_grade'], 
            bins=[0, 60, 70, 80, 90, 100], 
            labels=performance_levels
        )
        
        performance_counts = df['performance_level'].value_counts()
        
        # Create a bar chart of performance distribution
        bars = ax.bar(performance_counts.index, performance_counts.values, color='skyblue')
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        ax.set_title('Distribución de Rendimiento', fontsize=16)
        ax.set_xlabel('Nivel de Rendimiento', fontsize=14)
        ax.set_ylabel('Número de Estudiantes', fontsize=14)
        ax.set_ylim(0, max(performance_counts.values) * 1.2)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        text = "Distribución de Rendimiento:\n"
        for level, count in performance_counts.items():
            text += f"{level}: {count} students ({count/len(df)*100:.1f}%)\n"
        
        text += f"\nSummary Statistics:\nTotal Students: {len(df)}\n"
        text += f"Average Final Grade: {df['final_grade'].mean():.2f}\n"
        text += f"Average Attendance: {df['attendance_percentage'].mean():.2f}%"
        
        self.update_text(text)
    
    def show_time_series_analysis(self):
        """Show time series analysis options."""
        self.clear_visualization()
        self.clear_viz_frame()
        
        # Create options frame
        options_frame = ttk.Frame(self.viz_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(options_frame, text="Seleccionar visualización:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for visualization options
        options = ["Calificación Final Promedio en el Tiempo", "Asistencia Promedio en el Tiempo", "Rendimiento por Tema en el Tiempo"]
        selected_option = tk.StringVar()
        selected_option.set(options[0])
        
        dropdown = ttk.Combobox(options_frame, textvariable=selected_option, values=options, state="readonly")
        dropdown.pack(side=tk.LEFT, padx=5)
        
        # Function to update visualization based on selection
        def update_viz():
            option = selected_option.get()
            if option == "Calificación Final Promedio en el Tiempo":
                self.plot_grade_over_time()
            elif option == "Asistencia Promedio en el Tiempo":
                self.plot_attendance_over_time()
            elif option == "Rendimiento por Tema en el Tiempo":
                self.plot_topic_performance_over_time()
        
        # Create button to apply selection
        ttk.Button(options_frame, text="Mostrar", command=update_viz).pack(side=tk.LEFT, padx=5)
        
        # Show default visualization
        self.plot_grade_over_time()
    
    def plot_grade_over_time(self):
        """Plot average final grade over time."""
        fig = self.create_figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Extract time series data
        time_series_data = self.data['time_series_analysis']['time_series_data']
        
        # Convert to DataFrame
        df = pd.DataFrame(time_series_data)
        
        # Group by semester and year
        df['time_period'] = df['semester'] + ' ' + df['year'].astype(str)
        time_periods = sorted(df['time_period'].unique())
        
        # Calculate average final grade for each time period
        avg_grades = df.groupby('time_period')['avg_final_grade'].mean()
        
        ax.plot(time_periods, avg_grades, marker='o', linewidth=2, markersize=8)
        ax.set_title('Calificación Final Promedio en el Tiempo', fontsize=16)
        ax.set_xlabel('Período', fontsize=14)
        ax.set_ylabel('Calificación Final Promedio', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Calificación Final Promedio en el Tiempo: Este gráfico de línea muestra cómo el promedio de calificación final ha cambiado a lo largo de diferentes períodos de tiempo.")
    
    def plot_attendance_over_time(self):
        """Plot average attendance over time."""
        fig = self.create_figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Extract time series data
        time_series_data = self.data['time_series_analysis']['time_series_data']
        
        # Convert to DataFrame
        df = pd.DataFrame(time_series_data)
        
        # Group by semester and year
        df['time_period'] = df['semester'] + ' ' + df['year'].astype(str)
        time_periods = sorted(df['time_period'].unique())
        
        # Calculate average attendance for each time period
        avg_attendance = df.groupby('time_period')['avg_attendance'].mean()
        
        ax.plot(time_periods, avg_attendance, marker='o', linewidth=2, markersize=8, color='green')
        ax.set_title('Asistencia Promedio en el Tiempo', fontsize=16)
        ax.set_xlabel('Período', fontsize=14)
        ax.set_ylabel('Porcentaje de Asistencia Promedio', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Asistencia Promedio en el Tiempo: Este gráfico de línea muestra cómo el promedio de asistencia ha cambiado a lo largo de diferentes períodos de tiempo.")
    
    def plot_topic_performance_over_time(self):
        """Plot topic performance over time."""
        fig = self.create_figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        
        # Extract time series data
        time_series_data = self.data['time_series_analysis']['time_series_data']
        
        # Extract all topic scores for each time period
        topic_scores = []
        for record in time_series_data:
            for topic in record['topic_scores']:
                if topic['avg_score'] > 0:  # Only include non-zero scores
                    topic_scores.append({
                        'time_period': record['semester'] + ' ' + str(record['year']),
                        'topic_id': topic['topic_id'],
                        'score': topic['avg_score']
                    })
        
        # Convert to DataFrame
        topic_df = pd.DataFrame(topic_scores)
        
        # Create a heatmap of average scores by topic and time period
        if not topic_df.empty:
            pivot_table = topic_df.pivot_table(
                values='score', 
                index='time_period', 
                columns='topic_id', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.1f', ax=ax)
            ax.set_title('Rendimiento por Tema en el Tiempo', fontsize=16)
            ax.set_xlabel('ID del Tema', fontsize=14)
            ax.set_ylabel('Período', fontsize=14)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Rendimiento por Tema en el Tiempo: Este mapa de calor muestra cómo el rendimiento en diferentes temas ha cambiado a lo largo del tiempo. Los colores más oscuros indican mayores puntuaciones.")
    
    def show_correlation_heatmap(self):
        """Show correlation heatmap analysis."""
        self.clear_visualization()
        self.clear_viz_frame()
        
        # Create options frame
        options_frame = ttk.Frame(self.viz_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(options_frame, text="Seleccionar visualización:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for visualization options
        options = ["Mapa de Calor de Correlación", "Correlaciones Fuertes"]
        selected_option = tk.StringVar()
        selected_option.set(options[0])
        
        dropdown = ttk.Combobox(options_frame, textvariable=selected_option, values=options, state="readonly")
        dropdown.pack(side=tk.LEFT, padx=5)
        
        # Function to update visualization based on selection
        def update_viz():
            option = selected_option.get()
            if option == "Mapa de Calor de Correlación":
                self.plot_correlation_heatmap()
            elif option == "Correlaciones Fuertes":
                self.plot_strong_correlations()
        
        # Create button to apply selection
        ttk.Button(options_frame, text="Mostrar", command=update_viz).pack(side=tk.LEFT, padx=5)
        
        # Show default visualization
        self.plot_correlation_heatmap()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap."""
        fig = self.create_figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
        # Extract correlation data
        correlation_data = self.data['correlation_heatmap']['correlation_data']
        
        # Convert to DataFrame
        df = pd.DataFrame(correlation_data)
        
        # Create a pivot table for the heatmap
        pivot_table = df.pivot(index='topic_id1', columns='topic_id2', values='correlation')
        
        # Create the heatmap
        sns.heatmap(pivot_table, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=ax)
        ax.set_title('Mapa de Calor de Correlación entre Temas', fontsize=16)
        ax.set_xlabel('ID del Tema', fontsize=14)
        ax.set_ylabel('ID del Tema', fontsize=14)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Mapa de Calor de Correlación: Este mapa de calor muestra las correlaciones entre diferentes temas. El rojo indica correlaciones positivas, el azul indica correlaciones negativas y la intensidad indica la fuerza de la correlación.")
    
    def plot_strong_correlations(self):
        """Plot strong correlations."""
        fig = self.create_figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Extract correlation data
        correlation_data = self.data['correlation_heatmap']['correlation_data']
        
        # Convert to DataFrame
        df = pd.DataFrame(correlation_data)
        
        # Find strong correlations (absolute value > 0.7)
        strong_correlations = df[abs(df['correlation']) > 0.7]
        strong_correlations = strong_correlations[strong_correlations['topic_id1'] != strong_correlations['topic_id2']]
        
        if not strong_correlations.empty:
            # Create a bar chart of strong correlations
            pairs = [f"{row['topic_id1']}-{row['topic_id2']}" for _, row in strong_correlations.iterrows()]
            correlations = strong_correlations['correlation'].values
            
            bars = ax.barh(pairs, correlations, color=['red' if c < 0 else 'blue' for c in correlations])
            
            # Add value labels on the bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
            
            ax.set_title('Correlaciones Fuertes entre Temas (|r| > 0.7)', fontsize=16)
            ax.set_xlabel('Coeficiente de Correlación', fontsize=14)
            ax.set_ylabel('Pares de Temas', fontsize=14)
            ax.set_xlim(-1, 1)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        text = "Correlaciones Fuertes (|r| > 0.7):\n"
        for _, row in strong_correlations.iterrows():
            text += f"{row['topic_id1']} - {row['topic_id2']}: {row['correlation']:.3f}\n"
        
        # Group correlations by strength
        correlation_strength = pd.cut(
            df['correlation'], 
            bins=[-1, -0.7, -0.3, 0.3, 0.7, 1], 
            labels=['Negativa Fuerte', 'Negativa Moderada', 'Débil', 'Positiva Moderada', 'Positiva Fuerte']
        )
        
        strength_counts = correlation_strength.value_counts()
        
        text += "\nDistribución de Fuerza de Correlación:\n"
        for strength, count in strength_counts.items():
            text += f"{strength}: {count} pairs ({count/len(df)*100:.1f}%)\n"
        
        self.update_text(text)

def main():
    root = tk.Tk()
    app = VisualizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
