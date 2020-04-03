//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SAMPLING_FACTORY_H
#define SAMPLING_FACTORY_H

#include <algorithm>
#include <memory>
#include "sampling_method.h"
#include "gaussian_sampling.h"
#include "cached_gaussian_sampling.h"
#include "deepes.pb.h"
#include <glog/logging.h>

namespace DeepES{
/* @brief: create an sampling_method according to the configuration"
 * @args: 
 *    config: configuration for the DeepES
 * 
 */
std::shared_ptr<SamplingMethod> create_sampling_method(const DeepESConfig& Config);

}//namespace

#endif
