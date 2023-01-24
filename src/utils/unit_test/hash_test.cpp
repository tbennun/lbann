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
// MUST include this
#include "Catch2BasicSupport.hpp"

// File being tested
#include <lbann/utils/hash.hpp>

#include <unordered_set>

TEST_CASE("Testing convenience functions for hashing", "[hash][utilities]")
{

  SECTION("hash_combine")
  {
    std::unordered_set<size_t> hashes;
    for (size_t seed = 0; seed <= 16; seed += 2) {
      hashes.insert(seed);
    }
    for (size_t seed = 0; seed <= 16; seed += 2) {
      for (int val = -49; val <= 49; val += 7) {
        const auto hash = lbann::hash_combine(seed, val);
        CHECK_FALSE(hashes.count(hash));
        hashes.insert(hash);
      }
    }
  }

  SECTION("enum_hash")
  {
    enum class Humor
    {
      PHLEGMATIC,
      CHOLERIC,
      SANGUINE,
      MELANCHOLIC
    };
    std::vector<Humor> enum_list = {Humor::MELANCHOLIC,
                                    Humor::SANGUINE,
                                    Humor::CHOLERIC,
                                    Humor::PHLEGMATIC};
    std::unordered_set<size_t> hashes;
    for (const auto val : enum_list) {
      const auto hash = lbann::enum_hash<Humor>()(val);
      CHECK_FALSE(hashes.count(hash));
      hashes.insert(hash);
    }
  }

  SECTION("pair_hash")
  {
    const std::vector<unsigned long> i_list =
      {1, 2, 1018, 1019, 11209, 543210, 4294967295};
    const std::vector<float> j_list =
      {-12.34f, -8.76f, -4.56f, 0.f, 4.56f, 8.76f, 12.34f};
    std::unordered_set<size_t> hashes;
    for (const auto i : i_list) {
      for (const auto j : j_list) {
        std::pair<unsigned long, float> val1(i, j);
        const auto hash1 = lbann::pair_hash<unsigned long, float>()(val1);
        CHECK_FALSE(hashes.count(hash1));
        hashes.insert(hash1);
        std::pair<float, unsigned long> val2(j, i);
        const auto hash2 = lbann::pair_hash<float, unsigned long>()(val2);
        CHECK_FALSE(hashes.count(hash2));
        hashes.insert(hash2);
      }
    }
  }
}
