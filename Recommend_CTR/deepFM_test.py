from models.deepFM import DeepFM

if __name__ == '__main__':
    model = DeepFM(['test_1', 'test_2'])

    print(model)
    print(model.fm_layer)
    print(model.fm_layer.LogisticEmbeddingDict.idx2item)
    print(model.fm_layer.LogisticEmbeddingDict.item2idx)
