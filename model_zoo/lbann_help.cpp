////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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
// lbann_proto.cpp - prototext application
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/lbann_library.hpp"
#include <lbann/proto/proto_common.hpp>

#include <iostream>

using namespace lbann;

int main(int argc, char* argv[])
{
  auto& arg_parser = global_argument_parser();
  construct_all_options();

  try {
    arg_parser.parse(argc, argv);
  }
  catch (std::exception const& e) {
    std::cerr << "Error during argument parsing:\n\ne.what():\n\n  " << e.what()
              << "\n\nProcess terminating." << std::endl;
    std::terminate();
  }
  // Output help message from the arg_parser
  std::cout << arg_parser << std::endl;
  return EXIT_SUCCESS;
}
