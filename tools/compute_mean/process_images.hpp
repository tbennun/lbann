////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
//
////////////////////////////////////////////////////////////////////////////////

#ifndef _TOOLS_COMPUTE_MEAN_PROCESS_IMAGES_
#define _TOOLS_COMPUTE_MEAN_PROCESS_IMAGES_
#include "image_list.hpp"
#include "mpi_states.hpp"
#include "params.hpp"
#include "walltimes.hpp"

namespace tools_compute_mean {

/**
 * Crop images and use them to compute mean. At the same time, store the cropped
 * images.
 */
bool process_images(const image_list& img_list,
                    const params& mp,
                    const mpi_states& ms,
                    walltimes& wt);

} // end of namespace tools_compute_mean
#endif // _TOOLS_COMPUTE_MEAN_PROCESS_IMAGES_
