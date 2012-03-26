#!/bin/bash
# Script for testing basic functionaltity of matxvec_sparse

mkdir -p tests/logs
executable=build/matxvec_sparse/matxvec_sparse
logs=tests/logs
n_tests=6
n_passed=0
n_failed=0

echo -e "Test 1: Basic test ..."
$executable > "${logs}/test1.log"
returned=$?
if [ $returned -ne 0 ]; then
  n_failed=$((n_failed + 1))
  echo -e "\tFAILED! (${returned})"
else
  n_passed=$((n_passed + 1))
  echo -e "\tPASSED!"
fi

echo -e "Test 2: Higher dimension (50x40), 60 nnz ..."
$executable 50 40 60 > "${logs}/test2.log"
returned=$?
if [ $returned -ne 0 ]; then
  n_failed=$((n_failed + 1))
  echo -e "\tFAILED! (${returned})"
else
  n_passed=$((n_passed + 1))
  echo -e "\tPASSED!"
fi

echo -e "Test 3: Even higher dimension (500x400), 600 nnz ..."
$executable 500 400 600 > "${logs}/test3.log"
returned=$?
if [ $returned -ne 0 ]; then
  n_failed=$((n_failed + 1))
  echo -e "\tFAILED! (${returned})"
else
  n_passed=$((n_passed + 1))
  echo -e "\tPASSED!"
fi

echo -e "Test 4: Extreme high dimensionality (50000x40000), 60000 nnz ..."
$executable 50000 40000 60000 > "${logs}/test4.log"
returned=$?
if [ $returned -ne 0 ]; then
  n_failed=$((n_failed + 1))
  echo -e "\tFAILED! (${returned})"
else
  n_passed=$((n_passed + 1))
  echo -e "\tPASSED!"
fi

echo -e "Test 5: Same as Test 3, but more threads (8) ..."
$executable 500 400 600 8 > "${logs}/test5.log"
returned=$?
if [ $returned -ne 0 ]; then
  n_failed=$((n_failed + 1))
  echo -e "\tFAILED! (${returned})"
else
  n_passed=$((n_passed + 1))
  echo -e "\tPASSED!"
fi

echo -e "Test 6: Same as Test 4, but more threads (8) ..."
$executable 50000 40000 60000 8 > "${logs}/test6.log"
returned=$?
if [ $returned -ne 0 ]; then
  n_failed=$((n_failed + 1))
  echo -e "\tFAILED! (${returned})"
else
  n_passed=$((n_passed + 1))
  echo -e "\tPASSED!"
fi

# return number failures
exit $n_failed

