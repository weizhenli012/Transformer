from tqdm import tqdm

pinyin_list = []
hanzi_list = []
vocab = set()
max_length = 64

with open("./zh.tsv", errors='ignore', encoding='utf-8') as f:
    contexts = f.readlines()
    for line in contexts:
        line = line .strip().split(" ")
        pinyin = line[1].split(" ")
        hanzi = line[2].split(" ")
        for p,h in zip(pinyin,hanzi):
            vocab.add(p)
            vocab.add(h)
        pinyin = pinyin + ["PAD"]*(max_length-len(pinyin))
        hanzi = hanzi + ["PAD"]*(max_length-len(hanzi))
        if len(pinyin) <= max_length:
            pinyin_list.append(pinyin)
            hanzi_list.append(hanzi)

vocab = ["PAD"] + list(sorted(vocab))
vocab_size = len(vocab)

pinyin_list = pinyin_list[:3000]
hanzi_list = hanzi_list[:3000]

def get_token_ids():
    pinyin_ids = []
    hanzi_ids = []
    for pinyin,hanzi in zip(tqdm(pinyin_list,hanzi_list)):
        pinyin_ids.append([vocab.index(p) for p in pinyin])
        hanzi_ids.append([vocab.index(h) for h in hanzi])
    return pinyin_ids,hanzi_ids



if __name__ == '__main__':

    pinyin_tokens_ids,hanzi_tokens_ids = get_token_ids()

    for i in range(1024):
        pinyin = (pinyin_tokens_ids[i])
        hanzi = (hanzi_tokens_ids[i])
        print([vocab[py] for py in pinyin])
        print([vocab[hz] for hz in hanzi])
        print("------------------")