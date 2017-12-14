import numpy as np
import pandas as pd
import jieba
import matplotlib.pyplot as plt
import collections
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
def transexel2txt(org,target):
    df = pd.read_excel(org)
    df = df.values
    df = df[:,2]
    print(df.shape)
    with open(target,"w",encoding="utf-8") as f:
        for i in df:
            words = [k for k in jieba.cut(i)]
            words = " ".join(words)
            f.write(words+"\n")

def plot(org,max_num = 35):
    df = pd.read_excel(org)
    df = df.values
    df = df[:, 2]
    all_words  = np.zeros(shape = [max_num,])
    print(all_words)
    for i in df:
        words = [k for k in jieba.cut(i)]
        all_words[len(words)] += 1
    # for i in df:
    #     words = [k for k in jieba.cut(i)]
    #     all_words.append(len(words))
    # m = collections.Counter(all_words)
    # print(m)
    # print(m.most_common(3))


    X = range(max_num)
    Y = all_words
    fig = plt.figure()
    plt.bar(X, Y, 0.4, color="green")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("bar chart")

    plt.show()
    plt.savefig("../../data/len.jpg")
if __name__ == "__main__":
    # transexel2txt("../../data/org.xlsx","../../data/corpus.txt")
    plot("../../data/org.xlsx")