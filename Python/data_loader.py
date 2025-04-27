import json
import os

class DataLoader:
    def __init__(self):
        self.student_data = None
        self.results_data = None

    def load_data(self):
        """Carga los datos de entrada y resultados desde los JSON."""
        # Directorio donde está este archivo (Python/)
        current_file_dir = os.path.dirname(os.path.abspath(__file__))

        # La raíz del proyecto es el directorio padre de la carpeta Python
        root_dir = os.path.dirname(current_file_dir)

        # Construir las rutas correctas
        student_data_path = os.path.join(root_dir, "entradas", "student-performance-data.json")
        results_data_path = os.path.join(root_dir, "functional", "statistical_results.json")

        # Cargar JSONs
        with open(student_data_path, "r", encoding="utf-8") as f:
            self.student_data = json.load(f)

        with open(results_data_path, "r", encoding="utf-8") as f:
            self.results_data = json.load(f)

    def get_results(self):
        return self.results_data

    def get_student_name(self, student_id):
        if not self.student_data:
            return student_id
        for student in self.student_data.get('students', []):
            if student['id'] == student_id:
                return student.get('name', student_id)
        return student_id

    def get_course_name(self, course_id):
        if not self.student_data:
            return course_id
        for course in self.student_data.get('courses', []):
            if course['id'] == course_id:
                return course.get('name', course_id)
        return course_id

    def get_topic_name(self, topic_id):
        if not self.student_data:
            return topic_id
        for topic in self.student_data.get('topics', []):
            if topic['id'] == topic_id:
                return topic.get('name', topic_id)
        return topic_id

    def get_evaluation_name(self, evaluation_id):
        if not self.student_data:
            return evaluation_id
        for evaluation in self.student_data.get('evaluations', []):
            if evaluation['id'] == evaluation_id:
                return evaluation.get('name', evaluation_id)
        return evaluation_id
