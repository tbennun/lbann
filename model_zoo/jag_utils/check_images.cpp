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
////////////////////////////////////////////////////////////////////////////////

#include "lbann_config.hpp"

#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "lbann/lbann.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

using namespace lbann;

#define NUM_OUTPUT_DIRS 100
#define LBANN_OPTION_NUM_SAMPLES_PER_FILE 1000

//==========================================================================
int main(int argc, char* argv[])
{
  world_comm_ptr comm = initialize(argc, argv);
  bool master = comm->am_world_master();
  const int rank = comm->get_rank_in_world();
  const int np = comm->get_procs_in_world();

  try {
    auto& arg_parser = global_argument_parser();
    construct_std_options();
    construct_jag_options();
    try {
      arg_parser.parse(argc, argv);
    }
    catch (std::exception const& e) {
      auto guessed_rank = guess_global_rank();
      if (guessed_rank <= 0)
        // Cannot call `El::ReportException` because MPI hasn't been
        // initialized yet.
        std::cerr << "Error during argument parsing:\n\ne.what():\n\n  "
                  << e.what() << "\n\nProcess terminating." << std::endl;
      std::terminate();
    }

    if (arg_parser.get<std::string>(LBANN_OPTION_FILELIST)) {
      if (master) {
        throw lbann_exception(std::string{} + __FILE__ + " " +
                              std::to_string(__LINE__) +
                              " :: usage: " + argv[0] + " --filelist");
      }
    }

    std::vector<std::string> files;
    std::ifstream in(
      arg_parser.get<std::string>(LBANN_OPTION_FILELIST).c_str());
    if (!in) {
      throw lbann_exception(std::string{} + __FILE__ + " " +
                            std::to_string(__LINE__) + " :: failed to open " +
                            arg_parser.get<std::string>(LBANN_OPTION_FILELIST) +
                            " for reading");
    }
    std::string line;
    while (getline(in, line)) {
      if (line.size()) {
        files.push_back(line);
      }
    }
    in.close();

    hid_t hdf5_file_hnd;
    std::string key;
    conduit::Node n_ok;

    size_t h = 0;
    for (size_t j = rank; j < files.size(); j += np) {
      h += 1;
      if (h % 10 == 0)
        std::cout << rank << " :: processed " << h << " files\n";
      try {
        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read(files[j]);
      }
      catch (...) {
        std::cerr << rank
                  << " :: exception hdf5_open_file_for_read: " << files[j]
                  << "\n";
        continue;
      }

      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd,
                                                        "/",
                                                        cnames);
      }
      catch (const std::exception&) {
        std::cerr << rank
                  << " :: exception hdf5_group_list_child_names: " << files[j]
                  << "\n";
        continue;
      }

      for (size_t i = 0; i < cnames.size(); i++) {
        // is the next sample valid?
        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        }
        catch (const exception& e) {
          throw lbann_exception(
            std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
            " :: caught exception reading success flag for child " +
            std::to_string(i) + " of " + std::to_string(cnames.size()) + "; " +
            e.what());
        }
        int success = n_ok.to_int64();

        if (success == 1) {
          key = "/" + cnames[i] + "/outputs/images";
          std::vector<std::string> image_names;
          try {
            conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd,
                                                            key,
                                                            image_names);
          }
          catch (const std::exception&) {
            std::cerr
              << rank
              << " :: exception :hdf5_group_list_child_names for images: "
              << files[j] << "\n";
            continue;
          }
        }
      }
    }
  }
  catch (const std::exception& e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  // Clean up
  return EXIT_SUCCESS;
}
