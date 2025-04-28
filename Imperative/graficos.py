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

# Set style for all plots
plt.style.use('ggplot')
sns.set_palette("viridis")

class VisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Academic Performance Analysis Visualization")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")
        
        # Load data
        self.data = self.load_data()
        if not self.data:
            messagebox.showerror("Error", "Failed to load data. Please check the file path.")
            root.destroy()
            return
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create title
        title_label = ttk.Label(
            self.main_frame, 
            text="Academic Performance Analysis Visualization", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Create menu frame
        self.menu_frame = ttk.LabelFrame(self.main_frame, text="Analysis Options", padding="10")
        self.menu_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create buttons for each analysis
        ttk.Button(
            self.menu_frame, 
            text="1. Multiple Linear Regression Analysis", 
            command=self.show_multiple_linear_regression
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            self.menu_frame, 
            text="2. Logistic Regression Analysis", 
            command=self.show_logistic_regression
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            self.menu_frame, 
            text="3. Cluster Analysis", 
            command=self.show_cluster_analysis
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            self.menu_frame, 
            text="4. Time Series Analysis", 
            command=self.show_time_series_analysis
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            self.menu_frame, 
            text="5. Correlation Heatmap Analysis", 
            command=self.show_correlation_heatmap
        ).pack(fill=tk.X, pady=5)
        
        # Create visualization frame
        self.viz_frame = ttk.LabelFrame(self.main_frame, text="Visualization", padding="10")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create text frame for displaying statistics
        self.text_frame = ttk.LabelFrame(self.main_frame, text="Statistics", padding="10")
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
        
        ttk.Label(options_frame, text="Select visualization:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for visualization options
        options = ["Feature Importance", "Model Metrics", "Variable Influence"]
        selected_option = tk.StringVar()
        selected_option.set(options[0])
        
        dropdown = ttk.Combobox(options_frame, textvariable=selected_option, values=options, state="readonly")
        dropdown.pack(side=tk.LEFT, padx=5)
        
        # Function to update visualization based on selection
        def update_viz():
            option = selected_option.get()
            if option == "Feature Importance":
                self.plot_feature_importance()
            elif option == "Model Metrics":
                self.plot_model_metrics()
            elif option == "Variable Influence":
                self.plot_variable_influence()
        
        # Create button to apply selection
        ttk.Button(options_frame, text="Show", command=update_viz).pack(side=tk.LEFT, padx=5)
        
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
        
        ax.set_title('Feature Importance in Predicting Final Grade', fontsize=16)
        ax.set_xlabel('Standardized Coefficient', fontsize=14)
        ax.set_ylabel('Feature', fontsize=14)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Feature Importance Analysis: This chart shows the relative importance of each feature in predicting the final grade. Higher standardized coefficients indicate stronger influence.")
    
    def plot_model_metrics(self):
        """Plot model metrics for multiple linear regression."""
        fig = self.create_figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        # Extract metrics
        metrics = self.data['multiple_linear_regression']['model']['metrics']
        
        # Create a bar chart of metrics
        metric_names = ['R-squared', 'Adjusted R-squared', 'Mean Squared Error']
        metric_values = [metrics['r_squared'], metrics['adjusted_r_squared'], metrics['mean_squared_error']]
        
        bars = ax.bar(metric_names, metric_values, color=['green', 'blue', 'red'])
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        ax.set_title('Model Performance Metrics', fontsize=16)
        ax.set_ylabel('Value', fontsize=14)
        ax.set_ylim(0, max(metric_values) * 1.2)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text(f"Model Performance Metrics:\nR-squared: {metrics['r_squared']:.4f}\nAdjusted R-squared: {metrics['adjusted_r_squared']:.4f}\nMean Squared Error: {metrics['mean_squared_error']:.4f}\nSample Size: {metrics['sample_size']}")
    
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
        
        ax.set_title('Variable Influence on Final Grade', fontsize=16)
        ax.set_xlabel('Coefficient', fontsize=14)
        ax.set_ylabel('Variable', fontsize=14)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        text = "Variable Influence:\n"
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
        
        ttk.Label(options_frame, text="Select visualization:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for visualization options
        options = ["Scatter Plot", "Summary Statistics"]
        selected_option = tk.StringVar()
        selected_option.set(options[0])
        
        dropdown = ttk.Combobox(options_frame, textvariable=selected_option, values=options, state="readonly")
        dropdown.pack(side=tk.LEFT, padx=5)
        
        # Function to update visualization based on selection
        def update_viz():
            option = selected_option.get()
            if option == "Scatter Plot":
                self.plot_logistic_scatter()
            elif option == "Summary Statistics":
                self.plot_logistic_summary()
        
        # Create button to apply selection
        ttk.Button(options_frame, text="Show", command=update_viz).pack(side=tk.LEFT, padx=5)
        
        # Show default visualization
        self.plot_logistic_scatter()
    
    def plot_logistic_scatter(self):
        """Plot scatter plot for logistic regression."""
        fig = self.create_figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Extract logistic regression data
        lr_data = self.data['logistic_regression']['logistic_regression_data']
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(lr_data)
        
        # Create a scatter plot of early performance vs. cumulative GPA
        scatter = ax.scatter(df['early_performance'], df['cumulative_gpa'], 
                           c=df['passed'], cmap='viridis', s=100, alpha=0.7)
        
        # Add a colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Passed (1) / Failed (0)', fontsize=12)
        
        ax.set_title('Early Performance vs. Cumulative GPA', fontsize=16)
        ax.set_xlabel('Early Performance Score', fontsize=14)
        ax.set_ylabel('Cumulative GPA', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Logistic Regression Scatter Plot: This visualization shows the relationship between early performance, cumulative GPA, and pass/fail outcomes.")
    
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
        labels = ['Passed', 'Failed']
        sizes = [passing_students, failing_students]
        colors = ['lightgreen', 'lightcoral']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Pass/Fail Distribution', fontsize=16)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        text = f"Summary Statistics:\nTotal Students: {total_students}\n"
        text += f"Passing Students: {passing_students}\n"
        text += f"Failing Students: {failing_students}\n"
        text += f"Pass Rate: {pass_rate:.2f}%\n\n"
        
        # Calculate and display average values for passing vs failing students
        passing_avg = df[df['passed'] == 1][['early_performance', 'cumulative_gpa', 'attendance_percentage']].mean()
        failing_avg = df[df['passed'] == 0][['early_performance', 'cumulative_gpa', 'attendance_percentage']].mean()
        
        text += "Average Values:\nPassing Students:\n"
        text += f"  Early Performance: {passing_avg['early_performance']:.2f}\n"
        text += f"  Cumulative GPA: {passing_avg['cumulative_gpa']:.2f}\n"
        text += f"  Attendance: {passing_avg['attendance_percentage']:.2f}%\n"
        
        if not failing_avg.empty:
            text += "\nFailing Students:\n"
            text += f"  Early Performance: {failing_avg['early_performance']:.2f}\n"
            text += f"  Cumulative GPA: {failing_avg['cumulative_gpa']:.2f}\n"
            text += f"  Attendance: {failing_avg['attendance_percentage']:.2f}%"
        
        self.update_text(text)
    
    def show_cluster_analysis(self):
        """Show cluster analysis options."""
        self.clear_visualization()
        self.clear_viz_frame()
        
        # Create options frame
        options_frame = ttk.Frame(self.viz_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(options_frame, text="Select visualization:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for visualization options
        options = ["Attendance vs. Final Grade", "Topic Performance Heatmap", "Performance Distribution"]
        selected_option = tk.StringVar()
        selected_option.set(options[0])
        
        dropdown = ttk.Combobox(options_frame, textvariable=selected_option, values=options, state="readonly")
        dropdown.pack(side=tk.LEFT, padx=5)
        
        # Function to update visualization based on selection
        def update_viz():
            option = selected_option.get()
            if option == "Attendance vs. Final Grade":
                self.plot_attendance_vs_grade()
            elif option == "Topic Performance Heatmap":
                self.plot_topic_performance_heatmap()
            elif option == "Performance Distribution":
                self.plot_performance_distribution()
        
        # Create button to apply selection
        ttk.Button(options_frame, text="Show", command=update_viz).pack(side=tk.LEFT, padx=5)
        
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
        
        fig.colorbar(scatter, ax=ax, label='Final Grade')
        ax.set_title('Attendance vs. Final Grade', fontsize=16)
        ax.set_xlabel('Attendance Percentage', fontsize=14)
        ax.set_ylabel('Final Grade', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Attendance vs. Final Grade: This scatter plot shows the relationship between attendance percentage and final grade. The color gradient indicates the final grade value.")
    
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
            ax.set_title('Student Performance by Topic', fontsize=16)
            ax.set_xlabel('Topic ID', fontsize=14)
            ax.set_ylabel('Student ID', fontsize=14)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Topic Performance Heatmap: This heatmap shows the performance of each student across different topics. Darker colors indicate higher scores.")
    
    def plot_performance_distribution(self):
        """Plot performance distribution for cluster analysis."""
        fig = self.create_figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Extract cluster data
        cluster_data = self.data['cluster_analysis']['cluster_data']
        
        # Convert to DataFrame
        df = pd.DataFrame(cluster_data)
        
        # Group students by performance level
        df['performance_level'] = pd.cut(
            df['final_grade'], 
            bins=[0, 60, 70, 80, 90, 100], 
            labels=['Failing', 'Below Average', 'Average', 'Above Average', 'Excellent']
        )
        
        performance_counts = df['performance_level'].value_counts()
        
        # Create a bar chart of performance distribution
        bars = ax.bar(performance_counts.index, performance_counts.values, color='skyblue')
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        ax.set_title('Performance Distribution', fontsize=16)
        ax.set_xlabel('Performance Level', fontsize=14)
        ax.set_ylabel('Number of Students', fontsize=14)
        ax.set_ylim(0, max(performance_counts.values) * 1.2)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        text = "Performance Distribution:\n"
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
        
        ttk.Label(options_frame, text="Select visualization:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for visualization options
        options = ["Average Final Grade Over Time", "Average Attendance Over Time", "Topic Performance Over Time"]
        selected_option = tk.StringVar()
        selected_option.set(options[0])
        
        dropdown = ttk.Combobox(options_frame, textvariable=selected_option, values=options, state="readonly")
        dropdown.pack(side=tk.LEFT, padx=5)
        
        # Function to update visualization based on selection
        def update_viz():
            option = selected_option.get()
            if option == "Average Final Grade Over Time":
                self.plot_grade_over_time()
            elif option == "Average Attendance Over Time":
                self.plot_attendance_over_time()
            elif option == "Topic Performance Over Time":
                self.plot_topic_performance_over_time()
        
        # Create button to apply selection
        ttk.Button(options_frame, text="Show", command=update_viz).pack(side=tk.LEFT, padx=5)
        
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
        ax.set_title('Average Final Grade Over Time', fontsize=16)
        ax.set_xlabel('Time Period', fontsize=14)
        ax.set_ylabel('Average Final Grade', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Average Final Grade Over Time: This line plot shows how the average final grade has changed over different time periods.")
    
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
        ax.set_title('Average Attendance Over Time', fontsize=16)
        ax.set_xlabel('Time Period', fontsize=14)
        ax.set_ylabel('Average Attendance (%)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=45)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Average Attendance Over Time: This line plot shows how the average attendance has changed over different time periods.")
    
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
            ax.set_title('Topic Performance Over Time', fontsize=16)
            ax.set_xlabel('Topic ID', fontsize=14)
            ax.set_ylabel('Time Period', fontsize=14)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Topic Performance Over Time: This heatmap shows how performance in different topics has changed over time. Darker colors indicate higher scores.")
    
    def show_correlation_heatmap(self):
        """Show correlation heatmap analysis."""
        self.clear_visualization()
        self.clear_viz_frame()
        
        # Create options frame
        options_frame = ttk.Frame(self.viz_frame)
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(options_frame, text="Select visualization:").pack(side=tk.LEFT, padx=5)
        
        # Create dropdown for visualization options
        options = ["Correlation Heatmap", "Strong Correlations"]
        selected_option = tk.StringVar()
        selected_option.set(options[0])
        
        dropdown = ttk.Combobox(options_frame, textvariable=selected_option, values=options, state="readonly")
        dropdown.pack(side=tk.LEFT, padx=5)
        
        # Function to update visualization based on selection
        def update_viz():
            option = selected_option.get()
            if option == "Correlation Heatmap":
                self.plot_correlation_heatmap()
            elif option == "Strong Correlations":
                self.plot_strong_correlations()
        
        # Create button to apply selection
        ttk.Button(options_frame, text="Show", command=update_viz).pack(side=tk.LEFT, padx=5)
        
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
        ax.set_title('Topic Correlation Heatmap', fontsize=16)
        ax.set_xlabel('Topic ID', fontsize=14)
        ax.set_ylabel('Topic ID', fontsize=14)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        self.update_text("Topic Correlation Heatmap: This heatmap shows the correlations between different topics. Red indicates positive correlations, blue indicates negative correlations, and the intensity indicates the strength of the correlation.")
    
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
            
            ax.set_title('Strong Topic Correlations (|r| > 0.7)', fontsize=16)
            ax.set_xlabel('Correlation Coefficient', fontsize=14)
            ax.set_ylabel('Topic Pairs', fontsize=14)
            ax.set_xlim(-1, 1)
        
        fig.tight_layout()
        self.current_canvas.draw()
        
        # Update text with summary
        text = "Strong Correlations (|r| > 0.7):\n"
        for _, row in strong_correlations.iterrows():
            text += f"{row['topic_id1']} - {row['topic_id2']}: {row['correlation']:.3f}\n"
        
        # Group correlations by strength
        correlation_strength = pd.cut(
            df['correlation'], 
            bins=[-1, -0.7, -0.3, 0.3, 0.7, 1], 
            labels=['Strong Negative', 'Moderate Negative', 'Weak', 'Moderate Positive', 'Strong Positive']
        )
        
        strength_counts = correlation_strength.value_counts()
        
        text += "\nCorrelation Strength Distribution:\n"
        for strength, count in strength_counts.items():
            text += f"{strength}: {count} pairs ({count/len(df)*100:.1f}%)\n"
        
        self.update_text(text)

def main():
    root = tk.Tk()
    app = VisualizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
