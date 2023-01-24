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
#ifndef LBANN_EXECUTION_ALGORITHMS_LTFB_TERMINATION_CRITERIA_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_LTFB_TERMINATION_CRITERIA_HPP_INCLUDED

#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/execution_algorithms/ltfb/execution_context.hpp"

namespace lbann {
namespace ltfb {

/** @class TerminationCriteria
 *  @brief The stopping criteria for an LTFB-type algorithm
 *
 *  An object here needs to manage
 */
class LTFBTerminationCriteria final : public lbann::TerminationCriteria
{
public:
  LTFBTerminationCriteria(size_t max_metalearning_steps)
    : m_max_metalearning_steps{max_metalearning_steps}
  {}
  ~LTFBTerminationCriteria() = default;
  bool operator()(ExecutionContext const& c) const final
  {
    return this->operator()(dynamic_cast<LTFBExecutionContext const&>(c));
  }
  /** @brief Decide if the criteria are fulfilled. */
  bool operator()(LTFBExecutionContext const& exe_state) const noexcept
  {
    return exe_state.get_step() >= m_max_metalearning_steps;
  }

private:
  size_t m_max_metalearning_steps;
}; // class TerminationCriteria

} // namespace ltfb
} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_TERMINATION_CRITERIA_HPP_INCLUDED
