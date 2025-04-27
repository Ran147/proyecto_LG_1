open Yojson.Basic.Util
open Yojson.Basic

(* Type definitions *)
type evaluation = {
  evaluation_id: string;
  eval_type: string;
  weight: float;
  time_taken: string;
  attempts: int;
  score: float;
  topic_scores: (string * float) list;
}

type enrollment = {
  enrollment_id: string;
  student_id: string;
  course_id: string;
  semester: string;
  year: int;
  attendance_percentage: float;
  final_grade: float;
  evaluations: evaluation list;
}

type student = {
  student_id: string;
  name: string;
  cumulative_gpa: float;
}

type course = {
  course_id: string;
  title: string;
  topics: (string * string) list;
  prerequisites: string list;
}

type academic_data = {
  students: student list;
  courses: course list;
  enrollments: enrollment list;
}

(* Analysis result types *)
type regression_result = {
  weighted_score: float;
  attendance: float;
  final_grade: float;
}

type logistic_result = {
  prior_gpa: float;
  avg_eval_score: float;
  attendance: float;
  will_pass: bool;
}

type cluster_result = {
  student_id: string;
  topic_scores: (string * float) list;
  attendance: float;
  final_grade: float;
}

type time_series_result = {
  semester: string;
  year: int;
  avg_grade: float;
}

type correlation_result = {
  topic_id: string;
  avg_topic_score: float;
  avg_final_grade: float;
}

(* JSON output types and functions *)
type analysis_results = {
  regression: regression_result list;
  logistic: logistic_result list;
  clustering: cluster_result list;
  time_series: time_series_result list;
  correlation: correlation_result list;
}

let regression_to_json r =
  `Assoc [
    ("weighted_score", `Float r.weighted_score);
    ("attendance", `Float r.attendance);
    ("final_grade", `Float r.final_grade);
  ]

let logistic_to_json r =
  `Assoc [
    ("prior_gpa", `Float r.prior_gpa);
    ("avg_eval_score", `Float r.avg_eval_score);
    ("attendance", `Float r.attendance);
    ("will_pass", `Bool r.will_pass);
  ]

let cluster_to_json r =
  `Assoc [
    ("student_id", `String r.student_id);
    ("topic_scores", `List (List.map (fun (topic_id, score) ->
      `Assoc [
        ("topic_id", `String topic_id);
        ("score", `Float score);
      ]
    ) r.topic_scores));
    ("attendance", `Float r.attendance);
    ("final_grade", `Float r.final_grade);
  ]

let time_series_to_json r =
  `Assoc [
    ("semester", `String r.semester);
    ("year", `Int r.year);
    ("avg_grade", `Float r.avg_grade);
  ]

let correlation_to_json r =
  `Assoc [
    ("topic_id", `String r.topic_id);
    ("avg_topic_score", `Float r.avg_topic_score);
    ("avg_final_grade", `Float r.avg_final_grade);
  ]

let results_to_json results =
  `Assoc [
    ("regression", `List (List.map regression_to_json results.regression));
    ("logistic", `List (List.map logistic_to_json results.logistic));
    ("clustering", `List (List.map cluster_to_json results.clustering));
    ("time_series", `List (List.map time_series_to_json results.time_series));
    ("correlation", `List (List.map correlation_to_json results.correlation));
  ]

let write_results_to_file results filename =
  let json = results_to_json results in
  let channel = open_out filename in
  Yojson.Basic.to_channel channel json;
  close_out channel

(* Data loading and parsing functions *)
let parse_evaluation json =
  {
    evaluation_id = json |> member "evaluation_id" |> to_string;
    eval_type = json |> member "type" |> to_string;
    weight = json |> member "weight" |> to_float;
    time_taken = json |> member "time_taken" |> to_string;
    attempts = json |> member "attempts" |> to_int;
    score = json |> member "score" |> to_float;
    topic_scores = json |> member "topic_scores" |> to_list |> List.map (fun t ->
      (t |> member "topic_id" |> to_string, t |> member "score" |> to_float)
    );
  }

let parse_enrollment json =
  {
    enrollment_id = json |> member "enrollment_id" |> to_string;
    student_id = json |> member "student_id" |> to_string;
    course_id = json |> member "course_id" |> to_string;
    semester = json |> member "semester" |> to_string;
    year = json |> member "year" |> to_int;
    attendance_percentage = json |> member "attendance_percentage" |> to_float;
    final_grade = json |> member "final_grade" |> to_float;
    evaluations = json |> member "evaluations" |> to_list |> List.map parse_evaluation;
  }

let parse_student json =
  {
    student_id = json |> member "student_id" |> to_string;
    name = json |> member "name" |> to_string;
    cumulative_gpa = json |> member "cumulative_gpa" |> to_float;
  }

let parse_course json =
  {
    course_id = json |> member "course_id" |> to_string;
    title = json |> member "title" |> to_string;
    topics = json |> member "topics" |> to_list |> List.map (fun t ->
      (t |> member "topic_id" |> to_string, t |> member "name" |> to_string)
    );
    prerequisites = json |> member "prerequisites" |> to_list |> List.map to_string;
  }

let load_data filename =
  let json = Yojson.Basic.from_file filename in
  {
    students = json |> member "students" |> to_list |> List.map parse_student;
    courses = json |> member "courses" |> to_list |> List.map parse_course;
    enrollments = json |> member "enrollments" |> to_list |> List.map parse_enrollment;
  }

(* Helper functions *)
let average lst =
  let sum = List.fold_left (+.) 0.0 lst in
  sum /. float_of_int (List.length lst)

let find_student student_id students =
  List.find (fun s -> s.student_id = student_id) students

(* Statistical analysis functions *)
let multiple_linear_regression data =
  let calculate_weighted_scores evaluations =
    evaluations
    |> List.map (fun eval -> eval.score *. eval.weight)
    |> List.fold_left (+.) 0.0
  in
  
  data.enrollments
  |> List.map (fun enrollment ->
    {
      weighted_score = calculate_weighted_scores enrollment.evaluations;
      attendance = enrollment.attendance_percentage;
      final_grade = enrollment.final_grade;
    }
  )

let logistic_regression data =
  let pass_threshold = 70.0 in
  
  data.enrollments
  |> List.map (fun enrollment ->
    let student = find_student enrollment.student_id data.students in
    let avg_eval_score = enrollment.evaluations
      |> List.map (fun eval -> eval.score)
      |> average in
    {
      prior_gpa = student.cumulative_gpa;
      avg_eval_score;
      attendance = enrollment.attendance_percentage;
      will_pass = enrollment.final_grade >= pass_threshold;
    }
  )

let cluster_analysis data =
  let calculate_topic_scores evaluations =
    evaluations
    |> List.fold_left (fun acc eval ->
      List.fold_left (fun acc' (topic_id, score) ->
        let current = try List.assoc topic_id acc' with Not_found -> 0.0 in
        (topic_id, current +. score) :: List.remove_assoc topic_id acc'
      ) acc eval.topic_scores
    ) []
    |> List.map (fun (topic_id, score) ->
      (topic_id, score /. float_of_int (List.length evaluations))
    )
  in
  
  data.enrollments
  |> List.map (fun enrollment ->
    {
      student_id = enrollment.student_id;
      topic_scores = calculate_topic_scores enrollment.evaluations;
      attendance = enrollment.attendance_percentage;
      final_grade = enrollment.final_grade;
    }
  )

let time_series_analysis data =
  let group_by_semester enrollments =
    enrollments
    |> List.fold_left (fun acc enrollment ->
      let key = (enrollment.semester, enrollment.year) in
      let current = try List.assoc key acc with Not_found -> [] in
      (key, enrollment.final_grade :: current) :: List.remove_assoc key acc
    ) []
  in
  
  data.enrollments
  |> group_by_semester
  |> List.map (fun ((semester, year), grades) ->
    {
      semester;
      year;
      avg_grade = average grades;
    }
  )

let correlation_analysis data =
  let collect_topic_scores enrollments =
    enrollments
    |> List.fold_left (fun acc enrollment ->
      enrollment.evaluations
      |> List.fold_left (fun acc' eval ->
        eval.topic_scores
        |> List.fold_left (fun acc'' (topic_id, score) ->
          let current = try List.assoc topic_id acc'' with Not_found -> [] in
          (topic_id, (score, enrollment.final_grade) :: current) :: List.remove_assoc topic_id acc''
        ) acc'
      ) acc
    ) []
  in
  
  data.enrollments
  |> collect_topic_scores
  |> List.map (fun (topic_id, scores) ->
    let topic_scores = List.map fst scores in
    let final_grades = List.map snd scores in
    {
      topic_id;
      avg_topic_score = average topic_scores;
      avg_final_grade = average final_grades;
    }
  )

(* Result formatting functions *)
let format_regression_results results =
  results
  |> List.map (fun r ->
    Printf.sprintf "Weighted Score: %.2f, Attendance: %.2f%%, Final Grade: %.2f"
      r.weighted_score r.attendance r.final_grade
  )

let format_logistic_results results =
  results
  |> List.map (fun r ->
    Printf.sprintf "GPA: %.2f, Avg Score: %.2f, Attendance: %.2f%%, Will Pass: %b"
      r.prior_gpa r.avg_eval_score r.attendance r.will_pass
  )

let format_cluster_results results =
  results
  |> List.map (fun r ->
    let topic_scores = r.topic_scores
      |> List.map (fun (topic_id, score) ->
        Printf.sprintf "  Topic %s: %.2f" topic_id score
      )
      |> String.concat "\n" in
    Printf.sprintf "Student %s:\n%s\n  Attendance: %.2f%%, Final Grade: %.2f"
      r.student_id topic_scores r.attendance r.final_grade
  )

let format_time_series_results results =
  results
  |> List.map (fun r ->
    Printf.sprintf "%s %d: Average Grade = %.2f" r.semester r.year r.avg_grade
  )

let format_correlation_results results =
  results
  |> List.map (fun r ->
    Printf.sprintf "Topic %s: Avg Score = %.2f, Avg Final Grade = %.2f"
      r.topic_id r.avg_topic_score r.avg_final_grade
  )

(* Entry point *)
let () =
  let data = load_data "../entradas/academic-performance-mock-data.json" in
  
  let regression_results = multiple_linear_regression data in
  let logistic_results = logistic_regression data in
  let cluster_results = cluster_analysis data in
  let time_series_results = time_series_analysis data in
  let correlation_results = correlation_analysis data in
  
  let all_results = {
    regression = regression_results;
    logistic = logistic_results;
    clustering = cluster_results;
    time_series = time_series_results;
    correlation = correlation_results;
  } in
  
  (* Write results to JSON file *)
  write_results_to_file all_results "analysis_results.json";
  
  