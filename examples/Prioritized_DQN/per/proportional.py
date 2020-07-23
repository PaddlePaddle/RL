#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

from .utils import SumTree


class ProportionalPER(object):
    """Proportional Prioritized Experience Replay.
    """

    def __init__(self,
                 alpha,
                 seg_num,
                 size=1e6,
                 eps=0.01,
                 init_mem=None,
                 framestack=4):
        self.alpha = alpha
        self.seg_num = seg_num
        self.size = int(size)
        self.elements = SumTree(self.size)
        if init_mem:
            self.elements.from_list(init_mem)
        self.framestack = framestack
        self._max_priority = 1.0
        self.eps = eps

    def _get_stacked_item(self, idx):
        """ For atari environment, we use a 4-frame-stack as input
        """
        obs, act, reward, next_obs, done = self.elements.elements[idx]
        stacked_obs = np.zeros((self.framestack, ) + obs.shape)
        stacked_obs[-1] = obs
        for i in range(self.framestack - 2, -1, -1):
            elem_idx = (self.size + idx + i - self.framestack + 1) % self.size
            obs, _, _, _, d = self.elements.elements[elem_idx]
            if d:
                break
            stacked_obs[i] = obs
        return (stacked_obs, act, reward, next_obs, done)

    def store(self, item, delta=None):
        assert len(item) == 5  # (s, a, r, s', terminal)
        if not delta:
            delta = self._max_priority
        assert delta >= 0
        ps = np.power(delta + self.eps, self.alpha)
        self.elements.add(item, ps)

    def update(self, indices, priorities):
        priorities = np.array(priorities) + self.eps
        priorities_alpha = np.power(priorities, self.alpha)
        for idx, priority in zip(indices, priorities_alpha):
            self.elements.update(idx, priority)
            self._max_priority = max(priority, self._max_priority)

    def sample_one(self):
        assert self.elements.full(), "The replay memory is not full!"
        sample_val = np.random.uniform(0, self.elements.total_p)
        item, tree_idx, _ = self.elements.retrieve(sample_val)
        return item, tree_idx

    def sample(self):
        """ sample a batch of `seg_num` transitions

        Return:
            items: 
            indices: 
            probs: `N * P(i)`, for later calculating sampling weights
        """
        assert self.elements.full(), "The replay memory is not full!"
        seg_size = self.elements.total_p / self.seg_num
        seg_bound = [(seg_size * i, seg_size * (i + 1))
                     for i in range(self.seg_num)]
        items, indices, priorities = [], [], []
        for low, high in seg_bound:
            sample_val = np.random.uniform(low, high)
            _, tree_idx, priority = self.elements.retrieve(sample_val)
            elem_idx = tree_idx - self.elements.capacity + 1
            item = self._get_stacked_item(elem_idx)
            items.append(item)
            indices.append(tree_idx)
            priorities.append(priority)

        probs = self.size * np.array(priorities) / self.elements.total_p
        return np.array(items), np.array(indices), np.array(probs)
