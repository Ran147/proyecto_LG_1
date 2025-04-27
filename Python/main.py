# main.py

import tkinter as tk
from tkinter import Button
from data_loader import DataLoader
import visualizer

class StatisticsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Estadísticas V1.0")
        self.loader = DataLoader()
        self.loader.load_data()
        self.build_menu()

    def build_menu(self):
        """Crea la pantalla principal de botones."""
        tk.Label(self.root, text="📈 Menú de Visualización de Estadísticas 📉", font=("Arial", 16)).pack(pady=20)

        Button(self.root, text="1. Promedio de Rendimiento por Tema", width=40, command=self.show_topic_performance).pack(pady=5)
        Button(self.root, text="2. Mejora de Estudiantes", width=40, command=self.show_student_trends).pack(pady=5)
        Button(self.root, text="3. Dificultad de Temas", width=40, command=self.show_critical_points).pack(pady=5)
        Button(self.root, text="4. Correlaciones entre Temas", width=40, command=self.show_topic_correlations).pack(pady=5)
        Button(self.root, text="5. Tasas de Aprobación y Excelencia", width=40, command=self.show_pass_excellence_rates).pack(pady=5)
        Button(self.root, text="6. Rendimiento de Grupos", width=40, command=self.show_group_performance).pack(pady=5)
        Button(self.root, text="Salir", width=20, command=self.root.destroy).pack(pady=20)  # <- AHORA destroy, no quit

    def show_topic_performance(self):
        visualizer.show_topic_performance(self.loader, self.root)

    def show_student_trends(self):
        visualizer.show_student_trends(self.loader, self.root)

    def show_critical_points(self):
        visualizer.show_critical_points(self.loader, self.root)

    def show_topic_correlations(self):
        visualizer.show_topic_correlations(self.loader, self.root)

    def show_pass_excellence_rates(self):
        visualizer.show_pass_excellence_rates(self.loader, self.root)

    def show_group_performance(self):
        visualizer.show_group_performance(self.loader, self.root)

if __name__ == "__main__":
    root = tk.Tk()
    app = StatisticsApp(root)
    root.mainloop()
