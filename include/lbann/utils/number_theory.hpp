////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_UTILS_NUMBER_THEORY_HPP
#define LBANN_UTILS_NUMBER_THEORY_HPP

#include <vector>

namespace lbann {
namespace number_theory {

/** Get prime number.
 *  Indices are zero-indexed, so prime(0) is 2, prime(1) is 3, and so
 *  on. Results are cached for future function calls.
 */
int prime(int n);

/** Get prime factorization of n.
 *  Prime factors are sorted in ascending order.
 */
std::vector<int> prime_factors(int n);

/** Get balanced factorization of n.
 *  Factors are sorted in ascending order. The factors should be as
 *  close as possible.
 */
std::vector<int> balanced_factors(int n, int num_factors);

} // namespace number_theory
} // namespace lbann

#endif // LBANN_UTILS_NUMBER_THEORY_HPP
