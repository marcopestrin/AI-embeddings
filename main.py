import numpy as np
from fastembed import TextEmbedding, ImageEmbedding


# for x in TextEmbedding.list_supported_models():
#     print(x['model'])
# for x in ImageEmbedding.list_supported_models():
#     print(x['model'])

doc = [
    "My new home is really beautiful and I love it.",
    "Home is a club thet I don't like. Too many people and too much noise.",
    "Radika is a really fancy club in town. It's ok because is quiet and not crowded.",
    "My old house was really bad. It was too small and too dark."
]

embedding_model_for_text = TextEmbedding('mixedbread-ai/mxbai-embed-large-v1')
embeddings_text = np.array(list(embedding_model_for_text.embed(doc)))

embedding_model_for_image = ImageEmbedding('Qdrant/resnet50-onnx')
embeddings_image = list(embedding_model_for_image.embed(['./images/bangkok.webp', './images/cairo.webp', './images/brisbane.webp']))
# print(embeddings_image)


diff = embeddings_text[:, None, :] - embeddings_text[None, :, :]
distances = np.linalg.norm(diff, axis=-1)
# print(distances)

for  i in range(len(doc)):
    for j in range(i + 1, len(doc)):
        print(f"Distance between doc {i} and doc {j}: {distances[i, j]:.4f}")