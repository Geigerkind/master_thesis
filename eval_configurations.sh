export PYTHONPATH="/home/shino/Uni/master_thesis"

bool_arr=(0 1)
# Simple Square only
for encode_path_as_locations in "${bool_arr}"; do
  # Max height and num neurons
  python3 sources/build/eval.py ${encode_path_as_locations} 16 8 1 16 75 1
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_8_KNN_1_16_75_DS_1

  python3 sources/build/eval.py ${encode_path_as_locations} 16 16 1 32 75 1
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_16_KNN_1_32_75_DS_1

  python3 sources/build/eval.py ${encode_path_as_locations} 16 32 1 64 75 1
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_32_KNN_1_64_75_DS_1

  python3 sources/build/eval.py ${encode_path_as_locations} 16 64 1 128 75 1
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_64_KNN_1_128_75_DS_1

  # forest size and num hidden layer
  python3 sources/build/eval.py ${encode_path_as_locations} 8 32 1 32 75 1
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_8_32_KNN_1_32_75_DS_1

  python3 sources/build/eval.py ${encode_path_as_locations} 16 32 2 32 75 1
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_32_32_KNN_2_32_75_DS_1

  python3 sources/build/eval.py ${encode_path_as_locations} 32 32 4 32 75 1
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_32_32_KNN_4_32_75_DS_1

  python3 sources/build/eval.py ${encode_path_as_locations} 64 32 8 32 75 1
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_64_32_KNN_8_32_75_DS_1

  # Useful combinations
  python3 sources/build/eval.py ${encode_path_as_locations} 32 64 4 64 100 1
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_32_64_KNN_4_64_100_DS_1
done

# Simple Square and Long Rectangle
for encode_path_as_locations in "${bool_arr}"; do
  # Max height and num neurons
  python3 sources/build/eval.py ${encode_path_as_locations} 16 8 1 16 75 1,2
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_8_KNN_1_16_75_DS_12

  python3 sources/build/eval.py ${encode_path_as_locations} 16 16 1 32 75 1,2
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_16_KNN_1_32_75_DS_12

  python3 sources/build/eval.py ${encode_path_as_locations} 16 32 1 64 75 1,2
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_32_KNN_1_64_75_DS_12

  python3 sources/build/eval.py ${encode_path_as_locations} 16 64 1 128 75 1,2
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_64_KNN_1_128_75_DS_12

  # forest size and num hidden layer
  python3 sources/build/eval.py ${encode_path_as_locations} 8 32 1 32 75 1,2
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_8_32_KNN_1_32_75_DS_12

  python3 sources/build/eval.py ${encode_path_as_locations} 16 32 2 32 75 1,2
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_32_32_KNN_2_32_75_DS_12

  python3 sources/build/eval.py ${encode_path_as_locations} 32 32 4 32 75 1,2
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_32_32_KNN_4_32_75_DS_12

  python3 sources/build/eval.py ${encode_path_as_locations} 64 32 8 32 75 1,2
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_64_32_KNN_8_32_75_DS_12

  # Useful combinations
  python3 sources/build/eval.py ${encode_path_as_locations} 32 64 4 64 100 1,2
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_32_64_KNN_4_64_100_DS_12
done

# Simple Square and Long Rectangle and Rectangle with Ramp
for encode_path_as_locations in "${bool_arr}"; do
  # Max height and num neurons
  python3 sources/build/eval.py ${encode_path_as_locations} 16 8 1 16 75 1,2,3
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_8_KNN_1_16_75_DS_123

  python3 sources/build/eval.py ${encode_path_as_locations} 16 16 1 32 75 1,2,3
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_16_KNN_1_32_75_DS_123

  python3 sources/build/eval.py ${encode_path_as_locations} 16 32 1 64 75 1,2,3
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_32_KNN_1_64_75_DS_123

  python3 sources/build/eval.py ${encode_path_as_locations} 16 64 1 128 75 1,2,3
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_64_KNN_1_128_75_DS_123

  # forest size and num hidden layer
  python3 sources/build/eval.py ${encode_path_as_locations} 8 32 1 32 75 1,2,3
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_8_32_KNN_1_32_75_DS_123

  python3 sources/build/eval.py ${encode_path_as_locations} 16 32 2 32 75 1,2,3
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_32_32_KNN_2_32_75_DS_123

  python3 sources/build/eval.py ${encode_path_as_locations} 32 32 4 32 75 1,2,3
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_32_32_KNN_4_32_75_DS_123

  python3 sources/build/eval.py ${encode_path_as_locations} 64 32 8 32 75 1,2,3
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_64_32_KNN_8_32_75_DS_123

  # Useful combinations
  python3 sources/build/eval.py ${encode_path_as_locations} 32 64 4 64 100 1,2,3
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_32_64_KNN_4_64_100_DS_123
done

# Simple Square and Long Rectangle and Rectangle with Ramp and Many Corners
for encode_path_as_locations in "${bool_arr}"; do
  # Max height and num neurons
  python3 sources/build/eval.py ${encode_path_as_locations} 16 8 1 16 75 1,2,3,4
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_8_KNN_1_16_75_DS_1234

  python3 sources/build/eval.py ${encode_path_as_locations} 16 16 1 32 75 1,2,3,4
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_16_KNN_1_32_75_DS_1234

  python3 sources/build/eval.py ${encode_path_as_locations} 16 32 1 64 75 1,2,3,4
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_32_KNN_1_64_75_DS_1234

  python3 sources/build/eval.py ${encode_path_as_locations} 16 64 1 128 75 1,2,3,4
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_16_64_KNN_1_128_75_DS_1234

  # forest size and num hidden layer
  python3 sources/build/eval.py ${encode_path_as_locations} 8 32 1 32 75 1,2,3,4
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_8_32_KNN_1_32_75_DS_1234

  python3 sources/build/eval.py ${encode_path_as_locations} 16 32 2 32 75 1,2,3,4
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_32_32_KNN_2_32_75_DS_1234

  python3 sources/build/eval.py ${encode_path_as_locations} 32 32 4 32 75 1,2,3,4
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_32_32_KNN_4_32_75_DS_1234

  python3 sources/build/eval.py ${encode_path_as_locations} 64 32 8 32 75 1,2,3,4
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_64_32_KNN_8_32_75_DS_1234

  # Useful combinations
  python3 sources/build/eval.py ${encode_path_as_locations} 32 64 4 64 100 1,2,3,4
  python3 sources/build/eval_graphs.py eval_${encode_path_as_locations}_DT_32_64_KNN_4_64_100_DS_1234
done