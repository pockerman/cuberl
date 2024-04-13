echo "Running tests..."
cd build/tests/test_maths/test_vector_math
./test_vector_math

cd ..
cd ..
cd test_policies
cd test_epsilon_greedy_policy
./test_epsilon_greedy_policy
cd ..
cd test_max_tabular_policy
./test_max_tabular_policy
cd ..
cd test_random_tabular_policy
./test_random_tabular_policy
cd ..
cd test_softmax_policy
./test_softmax_policy
