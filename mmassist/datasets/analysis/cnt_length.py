import os
import json
from tqdm import tqdm
from collections import Counter
from mmassist.model.tokenization_proact import (
    build_tokenizer_and_update_config,
    ProActConfig,
)
from mmassist.configs.arguments import DATA_ROOT_DIR

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
config = ProActConfig(
    llm_pretrained=model_name,
    img_patch_token_size=(3, 3),
    use_img_cls_token=True,
)
tokenizer = build_tokenizer_and_update_config(config)
chat_formatter = tokenizer.chat_formatter


ann_dir = os.path.join(DATA_ROOT_DIR, "ego4d/annotations")


def cnt_num_tokens(ann_file):
    with open(os.path.join(ann_dir, ann_file), "r") as f:
        ann = json.load(f)

    conv = ann["conversation"]
    tknzed = tokenizer.encode(chat_formatter.apply_chat_template(conv))
    return len(tknzed)


import multiprocessing as mp

num_proc = mp.cpu_count()
pool = mp.Pool(num_proc)
jobs = [pool.apply_async(cnt_num_tokens, args=(af,)) for af in os.listdir(ann_dir)]
pool.close()

counter = Counter()
for job in tqdm(jobs):
    counter[job.get()] += 1

# plot histogram
import matplotlib.pyplot as plt
import json

with open("num_tokens.json", "w") as f:
    json.dump(dict(counter), f)

num_tokens = list(counter.keys())
plt.hist(num_tokens, bins=range(0, 80000, 500))
plt.xlabel("Number of tokens")
plt.ylabel("Number of videos")
# save
plt.savefig("num_tokens.png")
