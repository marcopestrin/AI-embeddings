from functools import lru_cache
from fastembed import TextEmbedding

def _cosine_distance(v1: list[float], v2: list[float]) -> float:
    # Calcola la distanza coseno (1 - similarità) tra due vettori.
    # Metrica: distanza coseno (0 = identici, 1 = opposti)
    if not v1 or not v2:
        return 1.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(x ** 2 for x in v1) ** 0.5
    norm2 = sum(x ** 2 for x in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return 1 - (dot / (norm1 * norm2))

def print_embedding_distances(airport_embeddings: dict, aircraft_embeddings: dict) -> None:
    # Stampa la distanza coseno tra tutti gli embeddings (airports e aircraft)
    if not airport_embeddings and not aircraft_embeddings:
        print("\n📊 Embeddings Distances: (non disponibili)")
        return

    print("\n" + "=" * 80)
    print("DISTANZE COSENO TRA EMBEDDINGS")
    print("=" * 80)

    # Distanze tra aeroporti (mostra i più distanti/simili)
    if airport_embeddings:
        print("\n TOP 10 COPPIE DI AEROPORTI PIÙ DISTANTI (meno simili):")
        distances = []
        iatas = sorted(airport_embeddings.keys())
        for i, iata1 in enumerate(iatas):
            for iata2 in iatas[i + 1 :]:
                dist = _cosine_distance(airport_embeddings[iata1], airport_embeddings[iata2])
                distances.append((iata1, iata2, dist))

        distances.sort(key=lambda x: x[2], reverse=True)
        for iata1, iata2, dist in distances[:10]:
            print(f"  {iata1} ↔ {iata2}: {dist:.4f}")

        print("\n TOP 10 COPPIE DI AEROPORTI PIÙ SIMILI (meno distanti):")
        distances.sort(key=lambda x: x[2])
        for iata1, iata2, dist in distances[:10]:
            print(f"  {iata1} ↔ {iata2}: {dist:.4f}")

    # Distanze tra velivoli
    if aircraft_embeddings:
        print("\n DISTANZE TRA VELIVOLI:")
        distances = []
        codes = sorted(aircraft_embeddings.keys())
        for i, code1 in enumerate(codes):
            for code2 in codes[i + 1 :]:
                dist = _cosine_distance(aircraft_embeddings[code1], aircraft_embeddings[code2])
                distances.append((code1, code2, dist))

        distances.sort(key=lambda x: x[2])
        for code1, code2, dist in distances[:10]:
            print(f"  {code1} ↔ {code2}: {dist:.4f}")

    print("=" * 80 + "\n")

@lru_cache(maxsize=1)
def _get_embedding_model():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def generate_airport_embeddings(airports: dict) -> dict[str, list[float]]:
    model = _get_embedding_model()
    embeddings = {}

    for iata, info in airports.items():
        text = f"{info['name']} in {info['city']}, {info['country']}. Tier {info['tier']} airport."
        vec = list(model.query_embed([text]))[0]
        embeddings[iata] = vec.tolist() if hasattr(vec, 'tolist') else list(vec)

    return embeddings

def generate_aircraft_embeddings(aircraft_data: dict) -> dict[str, list[float]]:
    model = _get_embedding_model()
    embeddings = {}

    for code, specs in aircraft_data.items():
        text = (
            f"{code}: {specs['manufacturer']} {specs['category']} aircraft. "
            f"{specs['seats']} seats, {specs['range_km']} km range, "
            f"{specs['cruise_kmh']} km/h cruise speed."
        )
        vec = list(model.query_embed([text]))[0]
        embeddings[code] = vec.tolist() if hasattr(vec, 'tolist') else list(vec)

    return embeddings
