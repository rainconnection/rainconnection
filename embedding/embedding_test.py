from feature_embedding import FeatureEmbeddingModel


if __name__ == '__main__':
    model = FeatureEmbeddingModel()

    print(model.embedding_dict.item2idx)
    print(model.embedding_dict.idx2item)

    print(model.embedding_dict.items)
    print(model.embedding_dict(['test_item_1', 'test_item_1', 'test_item_1', 'test_item_1']))