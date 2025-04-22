open Yojson.Basic.Util

(* Type definitions for our data structures *)
type student = {
  id: string;
  name: string;
  program: string;
  academic_year: int;
  gpa: float;
  courses_taken: string list;
}

type topic = {
  id: string;
  name: string;
  evaluations: string list;
}

type course = {
  id: string;
  name: string;
  semester: string;
  academic_year: int;
  credits: int;
  professor: string;
  department: string;
  topics: topic list;
  student_grades: (string * float) list;
}

type evaluation = {
  id: string;
  name: string option;
  date: string;
  weight: float option;
  topic_ids: string list;
  course_id: string;
  student_scores: (string * (string * float) list) list; (* student_id * (topic_id * score) list *)
}

type dataset = {
  students: student list;
  courses: course list;
  evaluations: evaluation list;
}

module StringPair = struct
  type t = string * string
  let compare = compare
end

module TopicCourseMap = Map.Make(StringPair)
module StringMap = Map.Make(String)

(* Helper functions *)
let list_take n lst =
  let rec aux acc n = function
    | [] -> List.rev acc
    | hd::tl when n > 0 -> aux (hd::acc) (n-1) tl
    | _ -> List.rev acc
  in
  aux [] n lst

let list_filter_map f lst =
  let rec aux acc = function
    | [] -> List.rev acc
    | hd::tl ->
        match f hd with
        | Some x -> aux (x::acc) tl
        | None -> aux acc tl
  in
  aux [] lst

(* Helper function to convert JSON numbers to float *)
let to_number_as_float = function
  | `Int i -> float_of_int i
  | `Float f -> f
  | json -> raise (Type_error ("Expected number (int/float)", json))

(* JSON parsing functions *)
let parse_student json =
  {
    id = json |> member "id" |> to_string;
    name = json |> member "name" |> to_string;
    program = json |> member "program" |> to_string;
    academic_year = json |> member "academic_year" |> to_int;
    gpa = json |> member "gpa" |> to_number_as_float;
    courses_taken = json |> member "courses_taken" |> to_list |> List.map to_string;
  }

let parse_topic json =
  {
    id = json |> member "id" |> to_string;
    name = json |> member "name" |> to_string;
    evaluations = json |> member "evaluations" |> to_list |> List.map to_string;
  }

let parse_course json =
  {
    id = json |> member "id" |> to_string;
    name = json |> member "name" |> to_string;
    semester = json |> member "semester" |> to_string;
    academic_year = json |> member "academic_year" |> to_int;
    credits = json |> member "credits" |> to_int;
    professor = json |> member "professor" |> to_string;
    department = json |> member "department" |> to_string;
    topics = json |> member "topics" |> to_list |> List.map parse_topic;
    student_grades = json |> member "student_grades" |> to_assoc |> List.map (fun (k, v) -> (k, to_number_as_float v));
  }

let parse_evaluation json =
  let student_scores =
    json |> member "student_scores" |> to_assoc |> List.map (fun (student_id, v) ->
      let by_topic = v |> member "by_topic" |> to_assoc |> List.map (fun (tid, score) -> (tid, to_number_as_float score)) in
      (student_id, by_topic)
    )
  in
  {
    id = json |> member "id" |> to_string;
    name = json |> member "type" |> to_string_option;
    date = json |> member "date" |> to_string;
    weight = json |> member "weight" |> to_float_option;
    topic_ids = json |> member "topics_covered" |> to_list |> List.map to_string;
    course_id = json |> member "course_id" |> to_string;
    student_scores;
  }

let parse_dataset json =
  {
    students = json |> member "students" |> to_list |> List.map parse_student;
    courses = json |> member "courses" |> to_list |> List.map parse_course;
    evaluations = json |> member "evaluations" |> to_list |> List.map parse_evaluation;
  }

(* Helper: collect all scores for each (topic, course) *)
let collect_topic_course_scores dataset =
  List.fold_left (fun acc eval ->
    List.fold_left (fun acc topic_id ->
      let scores =
        list_filter_map (fun (_, by_topic) ->  (* Removed unused student_id *)
          match List.assoc_opt topic_id by_topic with Some s -> Some s | None -> None
        ) eval.student_scores
      in
      let key = (topic_id, eval.course_id) in
      let prev = TopicCourseMap.find_opt key acc |> Option.value ~default:[] in
      TopicCourseMap.add key (prev @ scores) acc
    ) acc eval.topic_ids
  ) TopicCourseMap.empty dataset.evaluations

(* 1. Rendimiento por temas por cursos*)
let calculate_topic_performance dataset =
  let topic_scores = collect_topic_course_scores dataset in
  TopicCourseMap.mapi (fun _ scores ->  (* Removed unused topic_id and course_id *)
    let n = float_of_int (List.length scores) in
    let mean = List.fold_left (+.) 0. scores /. n in
    let variance = List.fold_left (fun acc x -> acc +. (x -. mean) ** 2.) 0. scores /. n in
    let std_dev = sqrt variance in
    let min_score = List.fold_left min infinity scores in
    let max_score = List.fold_left max neg_infinity scores in
    `Assoc [
      ("mean", `Float mean);
      ("std_dev", `Float std_dev);
      ("min_score", `Float min_score);
      ("max_score", `Float max_score)
    ]
  ) topic_scores

(* 2. Tendencias de rendimiento de estudiantes (sobre el tiempo, por fecha) *)
let calculate_student_trends dataset =
  let student_scores_over_time =
    List.fold_left (fun acc eval ->
      List.fold_left (fun acc (student_id, by_topic) ->
        let total = List.fold_left (fun s (_, v) -> s +. v) 0. by_topic in
        let prev = StringMap.find_opt student_id acc |> Option.value ~default:[] in
        StringMap.add student_id ((eval.date, total) :: prev) acc
      ) acc eval.student_scores
    ) StringMap.empty dataset.evaluations
  in
  StringMap.mapi (fun _ scores ->  (* Removed unused student_id *)
    let sorted = List.sort (fun (d1, _) (d2, _) -> String.compare d1 d2) scores in
    let n = float_of_int (List.length sorted) in
    let xs = List.mapi (fun i _ -> float_of_int i) sorted in
    let ys = List.map snd sorted in
    let sum_x = List.fold_left (+.) 0. xs in
    let sum_y = List.fold_left (+.) 0. ys in
    let sum_xy = List.fold_left2 (fun acc x y -> acc +. x *. y) 0. xs ys in
    let sum_x2 = List.fold_left (fun acc x -> acc +. x *. x) 0. xs in
    let slope =
      if n > 1. then
        (n *. sum_xy -. sum_x *. sum_y) /. (n *. sum_x2 -. sum_x *. sum_x)
      else 0.
    in
    let initial_score = List.hd ys in
    let final_score = List.hd (List.rev ys) in
    let improvement = final_score -. initial_score in
    `Assoc [
      ("slope", `Float slope);
      ("initial_score", `Float initial_score);
      ("final_score", `Float final_score);
      ("improvement", `Float improvement)
    ]
  ) student_scores_over_time

(* 3. Puntos de rendimiento críticos (dificultad/tasa de fracaso por tema) *)
let calculate_critical_points dataset =
  let topic_scores =
    List.fold_left (fun acc eval ->
      List.fold_left (fun acc topic_id ->
        let scores =
          list_filter_map (fun (_, by_topic) ->
            match List.assoc_opt topic_id by_topic with Some s -> Some s | None -> None
          ) eval.student_scores
        in
        let prev = StringMap.find_opt topic_id acc |> Option.value ~default:[] in
        StringMap.add topic_id (prev @ scores) acc
      ) acc eval.topic_ids
    ) StringMap.empty dataset.evaluations
  in
  StringMap.mapi (fun _ scores ->  (* Removed unused topic_id *)
    let n = float_of_int (List.length scores) in
    let mean = List.fold_left (+.) 0. scores /. n in
    let difficulty = 1. -. (mean /. 100.) in
    let failure_rate = float_of_int (List.length (List.filter (fun x -> x < 60.) scores)) /. n in
    `Assoc [
      ("difficulty", `Float difficulty);
      ("failure_rate", `Float failure_rate)
    ]
  ) topic_scores 

(* 4. Correlaciones entre temas (correlación entre temas) *)
let calculate_topic_correlations dataset =
  let topic_scores =
    List.fold_left (fun acc eval ->
      List.fold_left (fun acc topic_id ->
        let scores =
          list_filter_map (fun (_, by_topic) ->
            match List.assoc_opt topic_id by_topic with Some s -> Some s | None -> None
          ) eval.student_scores
        in
        let prev = StringMap.find_opt topic_id acc |> Option.value ~default:[] in
        StringMap.add topic_id (prev @ scores) acc
      ) acc eval.topic_ids
    ) StringMap.empty dataset.evaluations
  in
  let topic_ids = StringMap.bindings topic_scores |> List.map fst in
  let correlation xs ys =
    let n = float_of_int (List.length xs) in
    let mean_x = List.fold_left (+.) 0. xs /. n in
    let mean_y = List.fold_left (+.) 0. ys /. n in
    let cov = List.fold_left2 (fun acc x y -> acc +. (x -. mean_x) *. (y -. mean_y)) 0. xs ys /. n in
    let std_x = sqrt (List.fold_left (fun acc x -> acc +. (x -. mean_x) ** 2.) 0. xs /. n) in
    let std_y = sqrt (List.fold_left (fun acc y -> acc +. (y -. mean_y) ** 2.) 0. ys /. n) in
    if std_x = 0. || std_y = 0. then 0. else cov /. (std_x *. std_y)
  in
  List.fold_left (fun acc t1 ->
    List.fold_left (fun acc t2 ->
      if t1 < t2 then
        let xs = StringMap.find t1 topic_scores in
        let ys = StringMap.find t2 topic_scores in
        let min_len = min (List.length xs) (List.length ys) in
        let xs = list_take min_len xs in
        let ys = list_take min_len ys in
        let corr = correlation xs ys in
        StringMap.add (t1 ^ "_" ^ t2) (`Float corr) acc
      else acc
    ) acc topic_ids
  ) StringMap.empty topic_ids

(* Main function to run all analyses *)
let analyze_data json_file =
  let json = Yojson.Basic.from_file json_file in
  let dataset = parse_dataset json in
  let topic_performance = calculate_topic_performance dataset in
  let student_trends = calculate_student_trends dataset in
  let critical_points = calculate_critical_points dataset in
  let topic_correlations = calculate_topic_correlations dataset in
  let output_json = `Assoc [
    ("topic_performance", `Assoc (
      TopicCourseMap.bindings topic_performance |> List.map (fun ((topic_id, course_id), v) ->
        (topic_id ^ "_" ^ course_id, v)
      )
    ));
    ("student_trends", `Assoc (StringMap.bindings student_trends));
    ("critical_points", `Assoc (StringMap.bindings critical_points));
    ("topic_correlations", `Assoc (StringMap.bindings topic_correlations));
  ] in
  Yojson.Basic.to_file "statistical_results.json" output_json 

(* Entry point *)
let () =
  analyze_data "../entradas/student-performance-data.json" 