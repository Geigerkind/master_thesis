export PYTHONPATH="/home/shino/Uni/master_thesis"
export TF_CPP_MIN_LOG_LEVEL=2

NUM_ALLOWED_JOBS=1

function eval_model {
  if [[ ! -f "${PYTHONPATH}/bin/eval_${1}_DT_${2}_${3}_KNN_${4}_${5}_${6}_DS_${8}/evaluation_knn_model.h5" ]]; then
    python3 sources/build/eval.py ${1} ${2} ${3} ${4} ${5} ${6} ${7} 1
  fi
  if [[ ! -f "${PYTHONPATH}/bin/eval_${1}_DT_${2}_${3}_KNN_${4}_${5}_${6}_DS_${8}/evaluation_knn_anomaly_model.h5" ]]; then
    python3 sources/build/eval_anomaly.py ${1} ${2} ${3} ${4} ${5} ${6} ${7} 1
  fi
  if [[ ! -f "${PYTHONPATH}/bin/eval_${1}_DT_${2}_${3}_KNN_${4}_${5}_${6}_DS_${8}/combined_test_route/evaluation_continued_knn/log_general_metrics.csv" ]]; then
    python3 sources/build/eval_graphs.py ${1} ${2} ${3} ${4} ${5} ${6} ${7} 1
  fi
}

function exec_model_in_parallel {
  eval_model ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8} &

  # Block until more jobs may be started
  runningJobs=$(jobs | wc -l | xargs)
  while (( runningJobs >= NUM_ALLOWED_JOBS )); do
    sleep 0.1s
    runningJobs=$(jobs | grep -v "Done" | wc -l  | xargs)
  done
}

function eval_data_sets {
  bool_arr=( 0 1 )
  for encode_path_as_locations in "${bool_arr[@]}"; do
    # Max height and num neurons
    exec_model_in_parallel ${encode_path_as_locations} 16 8 1 16 75 ${1} ${2}
    exec_model_in_parallel ${encode_path_as_locations} 16 16 1 32 75 ${1} ${2}
    exec_model_in_parallel ${encode_path_as_locations} 16 32 1 64 75 ${1} ${2}
    exec_model_in_parallel ${encode_path_as_locations} 16 64 1 128 75 ${1} ${2}

    # forest size and num hidden layer
    exec_model_in_parallel ${encode_path_as_locations} 8 32 1 32 75 ${1} ${2}
    exec_model_in_parallel ${encode_path_as_locations} 16 32 2 32 75 ${1} ${2}
    exec_model_in_parallel ${encode_path_as_locations} 32 32 4 32 75 ${1} ${2}
    exec_model_in_parallel ${encode_path_as_locations} 64 32 8 32 75 ${1} ${2}

    # Useful combinations
    exec_model_in_parallel ${encode_path_as_locations} 32 64 4 64 75 ${1} ${2}
  done
}

function do_evaluation {
  eval_data_sets 1 1
  eval_data_sets 1,2 12
  eval_data_sets 1,2,3 123
  eval_data_sets 1,2,3,4 1234
}

do_evaluation