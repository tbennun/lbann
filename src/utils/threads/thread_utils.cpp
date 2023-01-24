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

#include "lbann/utils/threads/thread_utils.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/lbann_library.hpp"
#include <omp.h>
#include <thread>

namespace lbann {

int num_free_cores_per_process(const lbann_comm* comm)
{
  auto hw_cc = std::thread::hardware_concurrency();
  auto max_threads = std::max(hw_cc, decltype(hw_cc){1});

  auto omp_threads = omp_get_max_threads();
  auto processes_on_node = comm->get_procs_per_node();

  auto aluminum_threads = 0;
#ifdef LBANN_HAS_ALUMINUM
  aluminum_threads = 1;
#endif // LBANN_HAS_ALUMINUM

  auto max_cores_per_process =
    static_cast<int>(max_threads / processes_on_node);

  auto io_threads_per_process =
    std::max(1, (max_cores_per_process - omp_threads - aluminum_threads));

  return io_threads_per_process;
}

int free_core_offset(const lbann_comm* comm)
{
  auto hw_cc = std::thread::hardware_concurrency();
  auto max_threads = std::max(hw_cc, decltype(hw_cc){1});

  auto omp_threads = omp_get_max_threads();

  auto aluminum_threads = 0;
#ifdef LBANN_HAS_ALUMINUM
  aluminum_threads = 1;
#endif // LBANN_HAS_ALUMINUM

  // Offset into the CPUMASK of each process
  auto io_threads_offset = (omp_threads + aluminum_threads) % max_threads;

  return io_threads_offset;
}

} // namespace lbann
