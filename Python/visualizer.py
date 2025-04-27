# visualizer.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import Button, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from data_loader import DataLoader

def show_plot(fig, root_menu):
    """Muestra una gráfica y luego vuelve al menú."""
    root_menu.withdraw()

    root = tk.Toplevel()
    root.title("Visualización de Estadísticas")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    def close_window():
        root.destroy()
        root_menu.deiconify()

    btn = Button(root, text="Aceptar", command=close_window)
    btn.pack(pady=10)

    root.mainloop()

def show_topic_performance(loader, root_menu):
    """Muestra un menú de cursos, luego visualiza rendimiento promedio por temas del curso seleccionado."""
    results = loader.get_results()
    topic_perf = results.get('topic_performance', {})

    cursos_unicos = set()
    for key in topic_perf.keys():
        _, course_id = key.split("_")
        cursos_unicos.add(course_id)

    course_id_to_name = {cid: loader.get_course_name(cid) for cid in cursos_unicos}
    course_list = list(cursos_unicos)

    root_menu.withdraw()

    selector = tk.Toplevel()
    selector.title("Seleccione un Curso")

    tk.Label(selector, text="Selecciona un Curso:", font=("Arial", 14)).pack(pady=10)
    course_var = tk.StringVar(value=course_list[0])

    for cid in course_list:
        name = course_id_to_name[cid]
        tk.Radiobutton(selector, text=f"{name} ({cid})", variable=course_var, value=cid, font=("Arial", 12)).pack(anchor="w", padx=20)

    def seleccionar_curso():
        selected_course_id = course_var.get()
        selector.destroy()

        topics = []
        means = []
        for key, stats in topic_perf.items():
            topic_id, course_id = key.split("_")
            if course_id == selected_course_id:
                topic_name = loader.get_topic_name(topic_id)
                topics.append(topic_name)
                means.append(stats['mean'])

        if not topics:
            messagebox.showerror("Error", "No hay datos para este curso.")
            root_menu.deiconify()
            return

        fig, ax = plt.subplots(figsize=(12,8))
        bars = sns.barplot(x=means, y=topics, orient='h', palette='viridis', ax=ax)

        ax.set_title(f"Promedio de Rendimiento por Tema\nCurso: {course_id_to_name[selected_course_id]}", fontsize=16)
        ax.set_xlabel("Promedio (%)")
        ax.set_ylabel("Tema")
        plt.tight_layout()

        for bar in bars.patches:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', ha='left', va='center', fontsize=10, color='black')

        show_plot(fig, root_menu)

    tk.Button(selector, text="Aceptar", command=seleccionar_curso, font=("Arial", 12)).pack(pady=20)
    selector.mainloop()

def show_student_trends(loader, root_menu):
    """Visualiza las tendencias de mejora de estudiantes."""
    results = loader.get_results()
    trends = results.get('student_trends', {})

    students = []
    improvements = []
    for student_id, data in trends.items():
        student_name = loader.get_student_name(student_id)
        students.append(student_name)
        improvements.append(data['improvement'])

    fig, ax = plt.subplots(figsize=(12,8))
    bars = sns.barplot(x=improvements, y=students, orient='h', palette='coolwarm', ax=ax)

    ax.set_title("Mejora Total de los Estudiantes", fontsize=16)
    ax.set_xlabel("Incremento de Puntaje")
    ax.set_ylabel("Estudiante")
    plt.tight_layout()

    for bar in bars.patches:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.1f}', ha='left', va='center', fontsize=10, color='black')

    show_plot(fig, root_menu)

def show_critical_points(loader, root_menu):
    """Muestra un menú de cursos, luego visualiza la dificultad de los temas del curso seleccionado SIN números."""
    results = loader.get_results()
    critical_points = results.get('critical_points', {})
    topic_perf = results.get('topic_performance', {})

    topic_to_course = {}
    for key in topic_perf.keys():
        if "_" in key:
            topic_id, course_id = key.split("_")
            topic_to_course[topic_id] = course_id

    curso_a_temas = {}
    for topic_id, data in critical_points.items():
        course_id = topic_to_course.get(topic_id)
        if course_id:
            curso_a_temas.setdefault(course_id, []).append((topic_id, data['difficulty']))

    if not curso_a_temas:
        messagebox.showerror("Error", "No hay datos de cursos disponibles.")
        return

    course_list = list(curso_a_temas.keys())
    course_id_to_name = {cid: loader.get_course_name(cid) for cid in course_list}

    root_menu.withdraw()

    selector = tk.Toplevel()
    selector.title("Seleccione un Curso")

    tk.Label(selector, text="Selecciona un Curso:", font=("Arial", 14)).pack(pady=10)
    course_var = tk.StringVar(value=course_list[0])

    for cid in course_list:
        name = course_id_to_name.get(cid, cid)
        tk.Radiobutton(selector, text=f"{name} ({cid})", variable=course_var, value=cid, font=("Arial", 12)).pack(anchor="w", padx=20)

    def seleccionar_curso():
        selected_course_id = course_var.get()
        selector.destroy()

        temas_del_curso = curso_a_temas.get(selected_course_id, [])

        if not temas_del_curso:
            messagebox.showerror("Error", "No hay temas para este curso.")
            root_menu.deiconify()
            return

        topics = []
        difficulties = []
        for topic_id, difficulty in temas_del_curso:
            topic_name = loader.get_topic_name(topic_id)
            topics.append(topic_name)
            difficulties.append(difficulty)

        fig, ax = plt.subplots(figsize=(12,8))
        sns.barplot(x=difficulties, y=topics, orient='h', palette='magma', ax=ax)

        ax.set_title(f"Dificultad de Temas\nCurso: {course_id_to_name.get(selected_course_id, selected_course_id)}", fontsize=16)
        ax.set_xlabel("Índice de Dificultad")
        ax.set_ylabel("Tema")
        plt.tight_layout()

        show_plot(fig, root_menu)

    tk.Button(selector, text="Aceptar", command=seleccionar_curso, font=("Arial", 12)).pack(pady=20)
    selector.mainloop()

def show_topic_correlations(loader, root_menu):
    """Visualiza la correlación entre temas como un heatmap (sin números encima)."""
    results = loader.get_results()
    correlations = results.get('topic_correlations', {})

    topic_pairs = list(correlations.keys())
    unique_topics = sorted(set(t for pair in topic_pairs for t in pair.split("_")))
    topic_names = [loader.get_topic_name(tid) for tid in unique_topics]
    name_to_index = {name: idx for idx, name in enumerate(topic_names)}

    corr_matrix = np.zeros((len(topic_names), len(topic_names)))

    for pair, corr_value in correlations.items():
        t1, t2 = pair.split("_")
        n1 = loader.get_topic_name(t1)
        n2 = loader.get_topic_name(t2)
        i = name_to_index[n1]
        j = name_to_index[n2]
        corr_matrix[i, j] = corr_value
        corr_matrix[j, i] = corr_value

    fig, ax = plt.subplots(figsize=(12,10))
    sns.heatmap(
        corr_matrix,
        xticklabels=topic_names,
        yticklabels=topic_names,
        cmap='RdYlGn',
        center=0.5,
        annot=False,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"shrink": 0.75},
        ax=ax
    )
    ax.set_title("Mapa de Correlaciones entre Temas", fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    show_plot(fig, root_menu)

def show_pass_excellence_rates(loader, root_menu):
    """Visualiza tasas de aprobación y excelencia ocultando 100%."""
    results = loader.get_results()
    pass_rates = results.get('pass_excellence_rates', {})

    evaluations = []
    pass_percentages = []
    excellence_percentages = []

    for eval_id, data in pass_rates.items():
        eval_name = loader.get_evaluation_name(eval_id)
        evaluations.append(eval_name)
        pass_percentages.append(data['pass_rate'] * 100)
        excellence_percentages.append(data['excellence_rate'] * 100)

    x = np.arange(len(evaluations))

    fig, ax = plt.subplots(figsize=(16,8))
    width = 0.35
    bar1 = ax.bar(x - width/2, pass_percentages, width=width, label='Tasa de Aprobación (%)', color='royalblue')
    bar2 = ax.bar(x + width/2, excellence_percentages, width=width, label='Tasa de Excelencia (%)', color='darkorange')

    ax.set_xticks(x)
    ax.set_xticklabels(evaluations, rotation=90)
    ax.set_ylabel("Porcentaje (%)")
    ax.set_title("Tasa de Aprobación y Excelencia por Evaluación")
    ax.legend()
    plt.tight_layout()

    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            if height < 99.9:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 2, f'{height:.1f}%', ha='center', va='bottom', fontsize=9, color='black')

    show_plot(fig, root_menu)

def show_group_performance(loader, root_menu):
    """Muestra rendimiento de grupos filtrado por evaluación."""
    results = loader.get_results()
    group_perf = results.get('group_performance', {})

    if not group_perf:
        messagebox.showerror("Error", "No hay datos de grupos disponibles.")
        return

    evaluation_list = list(group_perf.keys())
    eval_id_to_name = {eid: loader.get_evaluation_name(eid) for eid in evaluation_list}

    root_menu.withdraw()

    selector = tk.Toplevel()
    selector.title("Seleccione una Evaluación")

    tk.Label(selector, text="Selecciona una Evaluación:", font=("Arial", 14)).pack(pady=10)
    eval_var = tk.StringVar(value=evaluation_list[0])

    for eid in evaluation_list:
        name = eval_id_to_name.get(eid, eid)
        tk.Radiobutton(selector, text=f"{name} ({eid})", variable=eval_var, value=eid, font=("Arial", 12)).pack(anchor="w", padx=20)

    def seleccionar_evaluacion():
        selected_eval_id = eval_var.get()
        selector.destroy()

        grupos = group_perf.get(selected_eval_id, {})

        if not grupos:
            messagebox.showerror("Error", "No hay grupos para esta evaluación.")
            root_menu.deiconify()
            return

        groups = []
        scores = []
        for group_id, data in grupos.items():
            avg_score = data['avg_score']
            groups.append(group_id)
            scores.append(avg_score)

        fig, ax = plt.subplots(figsize=(12,8))
        bars = sns.barplot(x=scores, y=groups, orient='h', palette='Spectral', ax=ax)

        ax.set_title(f"Promedio de Grupos\nEvaluación: {eval_id_to_name.get(selected_eval_id, selected_eval_id)}", fontsize=16)
        ax.set_xlabel("Promedio de Puntaje (%)")
        ax.set_ylabel("Grupo")
        plt.tight_layout()

        for bar in bars.patches:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%', ha='left', va='center', fontsize=10, color='black')

        show_plot(fig, root_menu)

    tk.Button(selector, text="Aceptar", command=seleccionar_evaluacion, font=("Arial", 12)).pack(pady=20)

    selector.mainloop()



