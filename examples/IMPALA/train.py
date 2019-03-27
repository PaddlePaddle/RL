#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from learner import Learner


def main(config):
    learner = Learner(config)

    try:
        while True:
            start = time.time()
            learner.step()
            while time.time() - start < config['log_metrics_interval_s']:
                learner.step()
            learner.log_metrics()

    except KeyboardInterrupt:
        learner.close()


if __name__ == '__main__':
    from impala_config import config
    main(config)
