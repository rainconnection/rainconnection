from models import fm

if __name__ == '__main__':
    model = fm.FactorizationMachine(['test_1', 'test_2'])

    print(model)
    print(model.LogisticEmbeddingDict.item2idx)
    print(model.LogisticEmbeddingDict.idx2item)
