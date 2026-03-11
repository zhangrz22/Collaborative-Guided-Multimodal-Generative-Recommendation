import json
import os
import numpy as np
import random
from collections import defaultdict
from multiprocessing import Process, Queue


def load_data(args):
    """Load interaction data from JSON file"""
    data_path = os.path.join(args.data_path, args.dataset, f"{args.dataset}.inter.json")

    with open(data_path, 'r') as f:
        inters = json.load(f)

    # Convert to format: user_id -> [item_list]
    # Re-index users and items starting from 1
    user_map = {}
    item_map = {}
    user_idx = 1
    item_idx = 1

    User = defaultdict(list)

    for user_id, item_list in inters.items():
        if user_id not in user_map:
            user_map[user_id] = user_idx
            user_idx += 1

        mapped_items = []
        for item_id in item_list:
            if item_id not in item_map:
                item_map[item_id] = item_idx
                item_idx += 1
            mapped_items.append(item_map[item_id])

        User[user_map[user_id]] = mapped_items

    usernum = user_idx - 1
    itemnum = item_idx - 1

    # Split into train/valid/test
    user_train = {}
    user_valid = {}
    user_test = {}

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 4:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]

    print(f"Dataset: {args.dataset}")
    print(f"Users: {usernum}, Items: {itemnum}")
    print(f"Interactions: {sum(len(v) for v in User.values())}")

    avg_len = sum(len(user_train[u]) for u in user_train) / len(user_train)
    print(f"Average sequence length: {avg_len:.2f}")

    return [user_train, user_valid, user_test, usernum, itemnum]


def random_neq(l, r, s):
    """Sample a random number in [l, r) that is not in set s"""
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, num_neg, result_queue, SEED):
    """Background process for sampling training batches with multiple negative samples"""
    def sample(uid):
        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen, num_neg], dtype=np.int32)  # Multiple negatives per position
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            # Sample multiple negative items for this position
            for n in range(num_neg):
                neg[idx, n] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0

    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    """Multi-process batch sampler"""
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, num_neg=1, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function,
                       args=(User, usernum, itemnum, batch_size, maxlen, num_neg, self.result_queue, np.random.randint(2e9)))
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
