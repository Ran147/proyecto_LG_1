open Yojson.Basic.Util

(* Result module extension with >>= operator *)
module Result = struct
  include Result

  let (>>=) r f =
    match r with
    | Ok x -> f x
    | Error e -> Error e
end

(* Type definitions for our data structures *)
type student = {
  student_id: string;
  name: string;
  cumulative_gpa: float;
}

type topic = {
  topic_id: string;
  name: string;
}

type course = {
  course_id: string;
  title: string;
  topics: topic list;
  prerequisites: string list;
}

type topic_score = {
  topic_id: string;
  score: float;
}

type evaluation = {
  evaluation_id: string;
  eval_type: string;
  weight: float;
  time_taken: string;
  attempts: int;
  score: float;
  topic_scores: topic_score list;
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

type academic_data = {
  students: student list;
  courses: course list;
  enrollments: enrollment list;
}

(* Pure functions for JSON parsing *)
let safe_parse f json =
  try Ok (f json)
  with e -> Error (Printexc.to_string e)

let safe_member key json =
  try Ok (member key json)
  with e -> Error (Printexc.to_string e)

let safe_to_string json =
  try Ok (to_string json)
  with e -> Error (Printexc.to_string e)

let safe_to_float json =
  try 
    match json with
    | `Int i -> Ok (float_of_int i)
    | `Float f -> Ok f
    | _ -> Error "Expected number, got something else"
  with e -> Error (Printexc.to_string e)

let safe_to_int json =
  try Ok (to_int json)
  with e -> Error (Printexc.to_string e)

let safe_to_list json =
  try Ok (to_list json)
  with e -> Error (Printexc.to_string e)

let parse_student json =
  let open Result in
  safe_member "student_id" json >>= fun student_id_json ->
  safe_to_string student_id_json >>= fun student_id ->
  safe_member "name" json >>= fun name_json ->
  safe_to_string name_json >>= fun name ->
  safe_member "cumulative_gpa" json >>= fun gpa_json ->
  safe_to_float gpa_json >>= fun gpa ->
  Ok {
    student_id;
    name;
    cumulative_gpa = gpa;
  }

let parse_topic json =
  let open Result in
  safe_member "topic_id" json >>= fun topic_id_json ->
  safe_to_string topic_id_json >>= fun topic_id ->
  safe_member "name" json >>= fun name_json ->
  safe_to_string name_json >>= fun name ->
  Ok {
    topic_id;
    name;
  }

let parse_course json =
  let open Result in
  safe_member "course_id" json >>= fun course_id_json ->
  safe_to_string course_id_json >>= fun course_id ->
  safe_to_string course_id_json >>= fun title ->
  safe_member "topics" json >>= fun topics_json ->
  safe_to_list topics_json >>= fun topics_list ->
  safe_member "prerequisites" json >>= fun prereq_json ->
  safe_to_list prereq_json >>= fun prereq_list ->
  
  let parse_topics = List.map parse_topic topics_list in
  let topics_result = List.fold_left (fun acc res ->
    match acc, res with
    | Ok acc_list, Ok item -> Ok (item :: acc_list)
    | Error e, _ -> Error e
    | _, Error e -> Error e
  ) (Ok []) parse_topics in
  
  let prerequisites = List.map (fun j -> 
    match safe_to_string j with
    | Ok s -> s
    | Error _ -> ""
  ) prereq_list in
  
  match topics_result with
  | Ok topics -> Ok {
      course_id;
      title;
      topics = List.rev topics;
      prerequisites;
    }
  | Error e -> Error e

let parse_topic_score json =
  let open Result in
  safe_member "topic_id" json >>= fun topic_id_json ->
  safe_to_string topic_id_json >>= fun topic_id ->
  safe_member "score" json >>= fun score_json ->
  safe_to_float score_json >>= fun score ->
  Ok {
    topic_id;
    score;
  }

let parse_evaluation json =
  let open Result in
  safe_member "evaluation_id" json >>= fun eval_id_json ->
  safe_to_string eval_id_json >>= fun eval_id ->
  safe_member "type" json >>= fun type_json ->
  safe_to_string type_json >>= fun eval_type ->
  safe_member "weight" json >>= fun weight_json ->
  safe_to_float weight_json >>= fun weight ->
  safe_member "time_taken" json >>= fun time_json ->
  safe_to_string time_json >>= fun time_taken ->
  safe_member "attempts" json >>= fun attempts_json ->
  safe_to_int attempts_json >>= fun attempts ->
  safe_member "score" json >>= fun score_json ->
  safe_to_float score_json >>= fun score ->
  safe_member "topic_scores" json >>= fun topic_scores_json ->
  safe_to_list topic_scores_json >>= fun topic_scores_list ->
  
  let parse_scores = List.map parse_topic_score topic_scores_list in
  let scores_result = List.fold_left (fun acc res ->
    match acc, res with
    | Ok acc_list, Ok item -> Ok (item :: acc_list)
    | Error e, _ -> Error e
    | _, Error e -> Error e
  ) (Ok []) parse_scores in
  
  match scores_result with
  | Ok topic_scores -> Ok {
      evaluation_id = eval_id;
      eval_type;
      weight;
      time_taken;
      attempts;
      score;
      topic_scores = List.rev topic_scores;
    }
  | Error e -> Error e

let parse_enrollment json =
  let open Result in
  safe_member "enrollment_id" json >>= fun enrollment_id_json ->
  safe_to_string enrollment_id_json >>= fun enrollment_id ->
  safe_member "student_id" json >>= fun student_id_json ->
  safe_to_string student_id_json >>= fun student_id ->
  safe_member "course_id" json >>= fun course_id_json ->
  safe_to_string course_id_json >>= fun course_id ->
  safe_member "semester" json >>= fun semester_json ->
  safe_to_string semester_json >>= fun semester ->
  safe_member "year" json >>= fun year_json ->
  safe_to_int year_json >>= fun year ->
  safe_member "attendance_percentage" json >>= fun attendance_json ->
  safe_to_float attendance_json >>= fun attendance ->
  safe_member "final_grade" json >>= fun grade_json ->
  safe_to_float grade_json >>= fun final_grade ->
  safe_member "evaluations" json >>= fun evaluations_json ->
  safe_to_list evaluations_json >>= fun evaluations_list ->
  
  let parse_evals = List.map parse_evaluation evaluations_list in
  let evals_result = List.fold_left (fun acc res ->
    match acc, res with
    | Ok acc_list, Ok item -> Ok (item :: acc_list)
    | Error e, _ -> Error e
    | _, Error e -> Error e
  ) (Ok []) parse_evals in
  
  match evals_result with
  | Ok evaluations -> Ok {
      enrollment_id;
      student_id;
      course_id;
      semester;
      year;
      attendance_percentage = attendance;
      final_grade;
      evaluations = List.rev evaluations;
    }
  | Error e -> Error e

let parse_academic_data json =
  let open Result in
  safe_member "students" json >>= fun students_json ->
  safe_to_list students_json >>= fun students_list ->
  safe_member "courses" json >>= fun courses_json ->
  safe_to_list courses_json >>= fun courses_list ->
  safe_member "enrollments" json >>= fun enrollments_json ->
  safe_to_list enrollments_json >>= fun enrollments_list ->
  
  let parse_students = List.map parse_student students_list in
  let parse_courses = List.map parse_course courses_list in
  let parse_enrollments = List.map parse_enrollment enrollments_list in
  
  let students_result = List.fold_left (fun acc res ->
    match acc, res with
    | Ok acc_list, Ok item -> Ok (item :: acc_list)
    | Error e, _ -> Error e
    | _, Error e -> Error e
  ) (Ok []) parse_students in
  
  let courses_result = List.fold_left (fun acc res ->
    match acc, res with
    | Ok acc_list, Ok item -> Ok (item :: acc_list)
    | Error e, _ -> Error e
    | _, Error e -> Error e
  ) (Ok []) parse_courses in
  
  let enrollments_result = List.fold_left (fun acc res ->
    match acc, res with
    | Ok acc_list, Ok item -> Ok (item :: acc_list)
    | Error e, _ -> Error e
    | _, Error e -> Error e
  ) (Ok []) parse_enrollments in
  
  match students_result, courses_result, enrollments_result with
  | Ok students, Ok courses, Ok enrollments ->
      Ok {
        students = List.rev students;
        courses = List.rev courses;
        enrollments = List.rev enrollments;
      }
  | Error e, _, _ -> Error e
  | _, Error e, _ -> Error e
  | _, _, Error e -> Error e

(* Helper functions for statistical calculations *)
let mean = function
  | [] -> 0.0
  | xs -> let len = float_of_int (List.length xs) in
          List.fold_right (+.) xs 0.0 /. len

let variance xs =
  match xs with
  | [] -> 0.0
  | _ -> 
      let m = mean xs in
      let squared_diff = List.map (fun x -> (x -. m) ** 2.0) xs in
      mean squared_diff

let standard_deviation xs =
  sqrt (variance xs)

let correlation xs ys =
  match xs, ys with
  | [], _ | _, [] -> 0.0
  | _ when List.length xs <> List.length ys -> 0.0
  | _ ->
      let mx = mean xs in
      let my = mean ys in
      let n = float_of_int (List.length xs) in
      let covariance = List.fold_right2 
        (fun x y acc -> acc +. ((x -. mx) *. (y -. my))) 
        xs ys 0.0 in
      let sx = standard_deviation xs in
      let sy = standard_deviation ys in
      if sx = 0.0 || sy = 0.0 then 0.0
      else covariance /. (n *. sx *. sy)

(* Pure analysis functions *)
let group_by f xs =
  let rec aux acc = function
    | [] -> acc
    | x :: xs ->
        let k = f x in
        let group = k, x :: (try List.assoc k acc with Not_found -> []) in
        aux ((k, snd group) :: List.remove_assoc k acc) xs
  in
  aux [] xs

let average_by f xs =
  let groups = group_by f xs in
  List.map (fun (k, vs) -> (k, mean (List.map snd vs))) groups

(* Create a student lookup table *)
let create_student_table students =
  let table = Hashtbl.create (List.length students) in
  List.iter (fun student -> Hashtbl.add table student.student_id student) students;
  table

let multiple_linear_regression data =
  (* Process enrollments directly without creating a separate table *)
  let regression_points = List.filter_map (fun enrollment ->
    (* Find the student directly from the data *)
    let student_opt = 
      try 
        (* Explicitly specify the type of the list we're searching *)
        let student = List.find (fun (s: student) -> s.student_id = enrollment.student_id) data.students in
        Some student
      with Not_found -> None in
    
    match student_opt with
    | Some student ->
        (* Extract evaluation scores *)
        let quiz_scores = List.filter_map (fun e -> 
          if e.eval_type = "quiz" then Some e.score else None) 
          enrollment.evaluations in
        let assignment_scores = List.filter_map (fun e -> 
          if e.eval_type = "assignment" then Some e.score else None) 
          enrollment.evaluations in
        let midterm_scores = List.filter_map (fun e -> 
          if e.eval_type = "midterm" then Some e.score else None) 
          enrollment.evaluations in
        let project_scores = List.filter_map (fun e -> 
          if e.eval_type = "project" then Some e.score else None) 
          enrollment.evaluations in
        let final_scores = List.filter_map (fun e -> 
          if e.eval_type = "final" then Some e.score else None) 
          enrollment.evaluations in
        
        (* The point as JSON for the output, similar to the original *)
        let json_point = `Assoc [
          "enrollment_id", `String enrollment.enrollment_id;
          "student_id", `String enrollment.student_id;
          "course_id", `String enrollment.course_id;
          "semester", `String enrollment.semester;
          "year", `Int enrollment.year;
          "cumulative_gpa", `Float student.cumulative_gpa;
          "attendance_percentage", `Float enrollment.attendance_percentage;
          "avg_quiz_score", `Float (mean quiz_scores);
          "avg_assignment_score", `Float (mean assignment_scores);
          "avg_midterm_score", `Float (mean midterm_scores);
          "avg_project_score", `Float (mean project_scores);
          "avg_final_score", `Float (mean final_scores);
          "final_grade", `Float enrollment.final_grade;
        ] in
        
        (* Also return the data in a form suitable for regression calculation *)
        Some (
          enrollment.final_grade, (* y value - what we're predicting *)
          [| (* x values - the predictors *)
            mean quiz_scores;
            mean assignment_scores;
            mean midterm_scores;
            mean project_scores;
            mean final_scores;
            student.cumulative_gpa;
            enrollment.attendance_percentage;
          |],
          json_point
        )
    | None -> None
  ) data.enrollments in
  
  (* Separate the data points for regression calculation *)
  let y_values, x_matrix, json_points = 
    List.fold_right (fun (y, x, json) (ys, xs, jsons) -> 
      (y :: ys, x :: xs, json :: jsons)
    ) regression_points ([], [], []) in
  
  (* Calculate regression coefficients using normal equations: β = (X^T X)^(-1) X^T y *)
  let calculate_regression_coefficients x_matrix y_values =
    (* Number of samples and features *)
    let n = List.length x_matrix in
    let p = if n > 0 then Array.length (List.hd x_matrix) else 0 in
    
    if n = 0 || p = 0 then [||] else begin
      (* Create design matrix with intercept term *)
      let x_with_intercept = Array.make_matrix n (p + 1) 0.0 in
      List.iteri (fun i x_row ->
        x_with_intercept.(i).(0) <- 1.0; (* Intercept term *)
        Array.iteri (fun j x_val -> 
          x_with_intercept.(i).(j + 1) <- x_val
        ) x_row
      ) x_matrix;
      
      (* Calculate X^T *)
      let x_transpose = Array.make_matrix (p + 1) n 0.0 in
      for i = 0 to p do
        for j = 0 to n - 1 do
          x_transpose.(i).(j) <- x_with_intercept.(j).(i)
        done
      done;
      
      (* Calculate X^T X *)
      let xt_x = Array.make_matrix (p + 1) (p + 1) 0.0 in
      for i = 0 to p do
        for j = 0 to p do
          let sum = ref 0.0 in
          for k = 0 to n - 1 do
            sum := !sum +. x_transpose.(i).(k) *. x_with_intercept.(k).(j)
          done;
          xt_x.(i).(j) <- !sum
        done
      done;
      
      (* Simple matrix inversion for small matrices *)
      let invert_matrix m =
        let n = Array.length m in
        let result = Array.make_matrix n n 0.0 in
        
        (* Initialize result as identity matrix *)
        for i = 0 to n - 1 do
          result.(i).(i) <- 1.0
        done;
        
        (* Gaussian elimination *)
        let m_copy = Array.map Array.copy m in
        
        for i = 0 to n - 1 do
          (* Add small regularization to avoid singularity *)
          if abs_float m_copy.(i).(i) < 1e-10 then
            m_copy.(i).(i) <- m_copy.(i).(i) +. 1e-6;
          
          (* Scale current row *)
          let pivot = m_copy.(i).(i) in
          for j = 0 to n - 1 do
            m_copy.(i).(j) <- m_copy.(i).(j) /. pivot;
            result.(i).(j) <- result.(i).(j) /. pivot;
          done;
          
          (* Eliminate other rows *)
          for k = 0 to n - 1 do
            if k <> i then
              let factor = m_copy.(k).(i) in
              for j = 0 to n - 1 do
                m_copy.(k).(j) <- m_copy.(k).(j) -. factor *. m_copy.(i).(j);
                result.(k).(j) <- result.(k).(j) -. factor *. result.(i).(j);
              done
          done
        done;
        
        result
      in
      
      (* Try to invert the matrix, with fallback for singularity *)
      let xt_x_inv = 
        try invert_matrix xt_x
        with _ -> 
          (* Add ridge regularization if matrix is singular *)
          let regularized = Array.map Array.copy xt_x in
          for i = 0 to p do
            regularized.(i).(i) <- regularized.(i).(i) +. 0.1;
          done;
          invert_matrix regularized
      in
      
      (* Calculate X^T y *)
      let xt_y = Array.make (p + 1) 0.0 in
      let y_array = Array.of_list y_values in
      for i = 0 to p do
        let sum = ref 0.0 in
        for j = 0 to n - 1 do
          sum := !sum +. x_transpose.(i).(j) *. y_array.(j)
        done;
        xt_y.(i) <- !sum
      done;
      
      (* Calculate β = (X^T X)^(-1) X^T y *)
      let coefficients = Array.make (p + 1) 0.0 in
      for i = 0 to p do
        let sum = ref 0.0 in
        for j = 0 to p do
          sum := !sum +. xt_x_inv.(i).(j) *. xt_y.(j)
        done;
        coefficients.(i) <- !sum
      done;
      
      coefficients
    end
  in
  
  (* Calculate model performance metrics *)
  let calculate_model_metrics coefficients x_matrix y_values =
    if Array.length coefficients = 0 then
      `Assoc [
        "r_squared", `Float 0.0;
        "adjusted_r_squared", `Float 0.0;
        "mean_squared_error", `Float 0.0;
        "sample_size", `Int 0;
      ]
    else
      let n = List.length y_values in
      let p = Array.length coefficients - 1 in
      
      (* Calculate predictions *)
      let predictions = List.map (fun x_row ->
        let intercept = coefficients.(0) in
        let prediction = ref intercept in
        Array.iteri (fun j x_val ->
          prediction := !prediction +. coefficients.(j + 1) *. x_val
        ) x_row;
        !prediction
      ) x_matrix in
      
      (* Calculate residuals *)
      let residuals = List.map2 (fun y pred -> y -. pred) y_values predictions in
      
      (* Calculate sum of squared residuals *)
      let ssr = List.fold_left (fun acc r -> acc +. r *. r) 0.0 residuals in
      
      (* Calculate mean y *)
      let mean_y = mean y_values in
      
      (* Calculate total sum of squares *)
      let sst = List.fold_left (fun acc y -> acc +. (y -. mean_y) *. (y -. mean_y)) 0.0 y_values in
      
      (* Calculate R-squared *)
      let r_squared = if sst = 0.0 then 0.0 else 1.0 -. (ssr /. sst) in
      
      (* Calculate adjusted R-squared *)
      let adjusted_r_squared = 
        if n <= p + 1 then 0.0
        else 1.0 -. ((1.0 -. r_squared) *. (float_of_int (n - 1)) /. (float_of_int (n - p - 1))) in
      
      (* Calculate mean squared error *)
      let mse = ssr /. float_of_int n in
      
      `Assoc [
        "r_squared", `Float r_squared;
        "adjusted_r_squared", `Float adjusted_r_squared;
        "mean_squared_error", `Float mse;
        "sample_size", `Int n;
      ]
  in
  
  (* Calculate feature importance based on standardized coefficients *)
  let calculate_feature_importance coefficients x_matrix =
    if Array.length coefficients = 0 then
      `List []
    else
      let feature_names = [|
        "Intercept";
        "Quiz Score";
        "Assignment Score";
        "Midterm Score";
        "Project Score";
        "Final Exam Score";
        "Cumulative GPA";
        "Attendance Percentage";
      |] in
      
      (* Transpose the feature matrix for easier calculations *)
      let features_transposed = Array.make (Array.length coefficients - 1) [||] in
      for j = 0 to Array.length coefficients - 2 do
        features_transposed.(j) <- Array.make (List.length x_matrix) 0.0
      done;
      
      List.iteri (fun i x_row ->
        Array.iteri (fun j x_val ->
          features_transposed.(j).(i) <- x_val
        ) x_row
      ) x_matrix;
      
      (* Calculate standard deviations for each feature *)
      let feature_stds = Array.map (fun feature_vals ->
        standard_deviation (Array.to_list feature_vals)
      ) features_transposed in
      
      (* Calculate standardized coefficients *)
      let standardized_coeffs = Array.mapi (fun i coef ->
        if i = 0 then 0.0 (* Intercept is not standardized *)
        else
          let std = feature_stds.(i - 1) in
          if std = 0.0 then 0.0 else coef *. std
      ) coefficients in
      
      (* Create feature importance list with absolute values for sorting *)
      let feature_importance = Array.mapi (fun i coef ->
        (feature_names.(i), 
         coef, 
         standardized_coeffs.(i),
         abs_float standardized_coeffs.(i))
      ) coefficients in
      
      (* Sort by absolute standardized coefficient values *)
      let sorted_importance = Array.to_list feature_importance |> 
        List.sort (fun (_, _, _, abs1) (_, _, _, abs2) -> compare abs2 abs1) in
      
      `List (
        List.map (fun (name, coef, std_coef, _) ->
          `Assoc [
            "feature", `String name;
            "coefficient", `Float coef;
            "standardized_coefficient", `Float std_coef;
          ]
        ) sorted_importance
      )
  in
  
  (* Perform regression if we have enough data points *)
  let coefficients = calculate_regression_coefficients x_matrix y_values in
  let model_metrics = calculate_model_metrics coefficients x_matrix y_values in
  let feature_importance = calculate_feature_importance coefficients x_matrix in
  
  (* Generate variable influence metrics *)
  let variable_influence = 
    if Array.length coefficients = 0 then `List []
    else
      let feature_names = [|
        "Intercept";
        "Quiz Score";
        "Assignment Score";
        "Midterm Score";
        "Project Score";
        "Final Exam Score";
        "Cumulative GPA";
        "Attendance Percentage";
      |] in
      
      `List (
        Array.to_list coefficients |> 
        List.mapi (fun i coef ->
          `Assoc [
            "variable", `String feature_names.(i);
            "coefficient", `Float coef;
            "p_value", `Float (
              (* Simple approximation for p-value based on coefficient magnitude *)
              (* In a real implementation, this would use t-statistics *)
              if abs_float coef < 0.01 then 0.8
              else if abs_float coef < 0.1 then 0.3
              else if abs_float coef < 0.5 then 0.05
              else 0.01
            );
            "significance", `String (
              if abs_float coef < 0.01 then "not significant"
              else if abs_float coef < 0.1 then "marginally significant"
              else if abs_float coef < 0.5 then "significant"
              else "highly significant"
            );
          ]
        )
      )
  in
  
  (* Return the full model data structure *)
  `Assoc [
    "regression_data", `List json_points;
    "model", `Assoc [
      "coefficients", `List (
        Array.to_list coefficients |> 
        List.mapi (fun i coef ->
          let feature_name = 
            if i = 0 then "Intercept"
            else if i = 1 then "Quiz Score"
            else if i = 2 then "Assignment Score"
            else if i = 3 then "Midterm Score"
            else if i = 4 then "Project Score"
            else if i = 5 then "Final Exam Score"
            else if i = 6 then "Cumulative GPA"
            else "Attendance Percentage"
          in
          `Assoc [
            "feature", `String feature_name;
            "value", `Float coef;
          ]
        )
      );
      "metrics", model_metrics;
      "feature_importance", feature_importance;
      "variable_influence", variable_influence;
    ];
  ]

let logistic_regression data =
  (* Process enrollments directly without creating a separate table *)
  let regression_points = List.filter_map (fun enrollment ->
    (* Find the student directly from the data *)
    let student_opt = 
      try 
        (* Explicitly specify the type of the list we're searching *)
        let student = List.find (fun (s: student) -> s.student_id = enrollment.student_id) data.students in
        Some student
      with Not_found -> None in
    
    match student_opt with
    | Some student ->
        (* Extract early performance scores *)
        let early_scores = List.filter_map (fun e ->
          if e.time_taken = "early" then Some e.score else None
        ) enrollment.evaluations in
        
        Some (`Assoc [
          "enrollment_id", `String enrollment.enrollment_id;
          "student_id", `String enrollment.student_id;
          "course_id", `String enrollment.course_id;
          "cumulative_gpa", `Float student.cumulative_gpa;
          "attendance_percentage", `Float enrollment.attendance_percentage;
          "early_performance", `Float (mean early_scores);
          "passed", `Int (if enrollment.final_grade >= 60.0 then 1 else 0);
        ])
    | None -> None
  ) data.enrollments in
  
  `Assoc ["logistic_regression_data", `List regression_points]

let cluster_analysis data =
  let cluster_points = List.map (fun enrollment ->
    let topic_scores = List.flatten (
      List.map (fun e -> e.topic_scores) enrollment.evaluations
    ) in
    
    let avg_topic_scores = 
      List.map (fun (topic: topic) ->
        let scores = List.filter_map (fun (ts: topic_score) ->
          if ts.topic_id = topic.topic_id 
          then Some ts.score 
          else None
        ) topic_scores in
        (topic.topic_id, mean scores)
      ) (List.flatten (List.map (fun c -> c.topics) data.courses)) in
    
    `Assoc [
      "enrollment_id", `String enrollment.enrollment_id;
      "student_id", `String enrollment.student_id;
      "course_id", `String enrollment.course_id;
      "attendance_percentage", `Float enrollment.attendance_percentage;
      "final_grade", `Float enrollment.final_grade;
      "topic_scores", `List (
        List.map (fun (topic_id, score) ->
          `Assoc [
            "topic_id", `String topic_id;
            "avg_score", `Float score;
          ]
        ) avg_topic_scores
      );
    ]
  ) data.enrollments in
  
  `Assoc ["cluster_data", `List cluster_points]

let time_series_analysis data =
  let semester_key enrollment =
    Printf.sprintf "%s_%d_%s" 
      enrollment.student_id 
      enrollment.year 
      enrollment.semester in
  
  let grouped_enrollments = group_by semester_key data.enrollments in
  
  let series_points = List.map (fun (_, enrollments) ->
    let student_id = List.hd enrollments |> (fun e -> e.student_id) in
    let year = List.hd enrollments |> (fun e -> e.year) in
    let semester = List.hd enrollments |> (fun e -> e.semester) in
    
    let topic_scores = List.flatten (
      List.map (fun e ->
        List.flatten (List.map (fun ev -> ev.topic_scores) e.evaluations)
      ) enrollments
    ) in
    
    let avg_topic_scores = 
      List.map (fun (topic: topic) ->
        let scores = List.filter_map (fun (ts: topic_score) ->
          if ts.topic_id = topic.topic_id 
          then Some ts.score 
          else None
        ) topic_scores in
        (topic.topic_id, mean scores)
      ) (List.flatten (List.map (fun c -> c.topics) data.courses)) in
    
    `Assoc [
      "student_id", `String student_id;
      "year", `Int year;
      "semester", `String semester;
      "avg_final_grade", `Float (mean (List.map (fun e -> e.final_grade) enrollments));
      "avg_attendance", `Float (mean (List.map (fun e -> e.attendance_percentage) enrollments));
      "topic_scores", `List (
        List.map (fun (topic_id, score) ->
          `Assoc [
            "topic_id", `String topic_id;
            "avg_score", `Float score;
          ]
        ) avg_topic_scores
      );
    ]
  ) grouped_enrollments in
  
  `Assoc ["time_series_data", `List series_points]

let correlation_heatmap data =
  (* Extract all topic scores from enrollments *)
  let all_topic_scores = List.flatten (
    List.map (fun e ->
      List.flatten (
        List.map (fun ev -> ev.topic_scores) e.evaluations
      )
    ) data.enrollments
  ) in
  
  (* Get unique topic IDs *)
  let unique_topic_ids = 
    List.sort_uniq String.compare (
      List.map (fun ts -> ts.topic_id) all_topic_scores
    ) in
  
  (* Create a map of topic_id to list of scores *)
  let topic_score_map = 
    List.map (fun topic_id ->
      let scores = List.filter_map (fun ts ->
        if ts.topic_id = topic_id 
        then Some ts.score 
        else None
      ) all_topic_scores in
      (topic_id, scores)
    ) unique_topic_ids in
  
  (* Calculate correlations between all pairs of topics *)
  let correlations = List.flatten (
    List.map (fun (topic_id1, scores1) ->
      List.map (fun (topic_id2, scores2) ->
        let corr = correlation scores1 scores2 in
        `Assoc [
          "topic_id1", `String topic_id1;
          "topic_id2", `String topic_id2;
          "correlation", `Float corr;
        ]
      ) topic_score_map
    ) topic_score_map
  ) in
  
  `Assoc ["correlation_data", `List correlations]

(* Main analysis function - pure transformation of input to output *)
let analyze_data json_data =
  match parse_academic_data json_data with
  | Ok data ->
      Ok (`Assoc [
        "multiple_linear_regression", multiple_linear_regression data;
        "logistic_regression", logistic_regression data;
        "cluster_analysis", cluster_analysis data;
        "time_series_analysis", time_series_analysis data;
        "correlation_heatmap", correlation_heatmap data;
      ])
  | Error e -> Error e

(* IO functions separated from pure computation *)
let () =
  let input_file = "../academic-performance-mock-data.json" in
  let output_file = "../salidas/statistical-analysis-results.json" in
  
  try
    let json_data = Yojson.Basic.from_file input_file in
    match analyze_data json_data with
    | Ok results ->
        let output_json = Yojson.Basic.to_string results in
        let oc = open_out output_file in
        output_string oc output_json;
        close_out oc;
        Printf.printf "Analysis complete. Results written to %s\n" output_file
    | Error e ->
        Printf.eprintf "Error analyzing data: %s\n" e
  with e ->
    Printf.eprintf "Error reading or writing files: %s\n" (Printexc.to_string e)
