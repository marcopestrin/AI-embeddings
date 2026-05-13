from agentspan.agents import tool
from network import (
    AIRPORTS,
    AIRCRAFT_DATA,
    ROUTE_NETWORK,
    # Getter lazy: generano gli embedding solo alla prima chiamata (vedi network.py)
    get_airport_embeddings_lazy,
    get_aircraft_embeddings_lazy,
)


def _parse_arrival_to_minutes(arrival_local: str) -> int:
    if "+" in arrival_local:
        base, offset = arrival_local.split("+")
        day_offset = int(offset)
    else:
        base, day_offset = arrival_local, 0

    h, m = map(int, base.split(":"))
    return day_offset * 24 * 60 + h * 60 + m


def _parse_departure_to_minutes(departure_local: str) -> int:
    # Le partenze sono sempre nel formato 'HH:MM' senza offset di giorno.
    h, m = map(int, departure_local.split(":"))
    return h * 60 + m


@tool
def list_routes_from(iata_code: str) -> list[dict]:
    # tutte le destinazioni dirette servite da un aeroporto, ordinate per traffico.
    code = iata_code.upper()
    if code not in AIRPORTS:
        raise ValueError(f"Aeroporto '{code}' non trovato nel database")

    # filtriamo le rotte che PARTONO dall'aeroporto richiesto
    result = []
    for (origin, destination), flights in ROUTE_NETWORK.items():
        if origin != code:
            continue  # Salta tutte le rotte che non partono da `code`

        result.append({
            "destination_iata":      destination,
            "destination_city":      AIRPORTS[destination]["city"],
            "daily_flights":         len(flights),
            # set() elimina i duplicati. sorted() per output stabile.
            "carriers":              sorted({f["carrier"]  for f in flights}),
            "aircraft_types":        sorted({f["aircraft"] for f in flights}),
        })

    # ordinamento finale per traffico decrescente
    result.sort(key=lambda x: x["daily_flights"], reverse=True)
    return result


@tool
def get_route_schedule(origin: str, destination: str) -> dict:
    o, d = origin.upper(), destination.upper()
    if (o, d) not in ROUTE_NETWORK:
        return {
            "origin":           o,
            "destination":      d,
            "found":            False,
            "message":          f"Nessun volo diretto da {o} a {d}",
            "flights":          [],
        }

    # ritorniamo nuovi dict, non i riferimenti interni: l'agente non può corrompere ROUTE_NETWORK
    return {
        "origin":           o,
        "destination":      d,
        "found":            True,
        "flights":          [dict(f) for f in ROUTE_NETWORK[(o, d)]],
    }


@tool
def get_airport_centrality(iata_code: str) -> dict:
    # Misura la centralità di un aeroporto nella rete:
    # - quanti aeroporti si raggiungono direttamente (out)
    # - da quanti si è raggiunti direttamente (in)
    # - voli totali al giorno (in + out)
    # - se può essere classificato come "hub" (>= 70% di copertura)
    code = iata_code.upper()
    if code not in AIRPORTS:
        raise ValueError(f"Aeroporto '{code}' non trovato nel database")

    outbound_dests = set()
    inbound_origs = set()
    out_flights = 0
    in_flights = 0

    # scansione completa della rete
    for (origin, destination), flights in ROUTE_NETWORK.items():
        if origin == code:
            outbound_dests.add(destination)
            out_flights += len(flights)
        if destination == code:
            inbound_origs.add(origin)
            in_flights += len(flights)

    # numero massimo di partner possibili (tutti gli altri aeroporti)
    max_partners = len(AIRPORTS) - 1
    coverage_pct = round(100 * len(outbound_dests) / max_partners, 1)

    return {
        "iata":                    code,
        "city":                    AIRPORTS[code]["city"],
        "tier":                    AIRPORTS[code]["tier"],
        "outbound_destinations":   sorted(outbound_dests),
        "outbound_count":          len(outbound_dests),
        "outbound_flights_per_day": out_flights,
        "inbound_origins":         sorted(inbound_origs),
        "inbound_count":           len(inbound_origs),
        "inbound_flights_per_day": in_flights,
        "total_daily_flights":     out_flights + in_flights,
        "coverage_pct":            coverage_pct,
        "is_hub":                  coverage_pct >= 70.0,
    }


@tool
def find_connections(origin: str, destination: str) -> list[dict]:
    # Trova voli con UNO scalo intermedio fra due aeroporti. Vincoli realistici:
    # - tempo di transito minimo: 90 minuti
    # - tempo di transito massimo: 8 ore (480 minuti)

    origin, destination = origin.upper(), destination.upper()
    MIN_LAYOVER = 90
    MAX_LAYOVER = 480
    results = []

    # Per ogni aeroporto candidato come scalo intermedio
    for layover in AIRPORTS:
        if layover in (origin, destination):
            continue # escludi origine e destinazione dalla lista degli scali possibili
        if (origin, layover) not in ROUTE_NETWORK or (layover, destination) not in ROUTE_NETWORK:
            continue

        # Combinazioni di voli: ogni volo del 1° tratto x ogni volo del 2° tratto
        for f1 in ROUTE_NETWORK[(origin, layover)]:
            arr1_min = _parse_arrival_to_minutes(f1["arrival_local"])

            for f2 in ROUTE_NETWORK[(layover, destination)]:
                dep2_min = _parse_departure_to_minutes(f2["departure_local"])

                # Se la partenza del 2° volo è prima dell'arrivo del 1° volo significa che si parte il giorno dopo (es. arrivo 23:00, ripartenza 06:00 = +1 giorno)
                if dep2_min < arr1_min:
                    dep2_min += 24 * 60

                layover_min = dep2_min - arr1_min

                # Filtra solo connessioni realistiche
                if not (MIN_LAYOVER <= layover_min <= MAX_LAYOVER):
                    continue

                total_min = f1["duration_min"] + layover_min + f2["duration_min"]

                results.append({
                    "leg1": {
                        "flight_no":     f1["flight_no"],
                        "carrier":       f1["carrier"],
                        "from":          origin,
                        "to":            layover,
                        "departure":     f1["departure_local"],
                        "arrival":       f1["arrival_local"],
                        "duration_min":  f1["duration_min"],
                        "aircraft":      f1["aircraft"],
                    },
                    "layover_iata":      layover,
                    "layover_city":      AIRPORTS[layover]["city"],
                    "layover_min":       layover_min,
                    "leg2": {
                        "flight_no":     f2["flight_no"],
                        "carrier":       f2["carrier"],
                        "from":          layover,
                        "to":            destination,
                        "departure":     f2["departure_local"],
                        "arrival":       f2["arrival_local"],
                        "duration_min":  f2["duration_min"],
                        "aircraft":      f2["aircraft"],
                    },
                    "total_duration_min": total_min,
                })

    # ordina dal viaggio più rapido al più lento
    results.sort(key=lambda x: x["total_duration_min"])
    return results


@tool
def get_airline_network(carrier: str) -> dict:
    code = carrier.upper()

    routes_operated = []
    airports_served = set()
    aircraft_used = set()
    flights_per_airport: dict[str, int] = {}  # iata -> conteggio voli (per trovare l'hub)

    # scansione completa: per ogni rotta della rete, isolo solo i voli di questa compagnia
    for (origin, destination), flights in ROUTE_NETWORK.items():
        carrier_flights = [f for f in flights if f["carrier"] == code]
        if not carrier_flights:
            continue

        airports_served.add(origin)
        airports_served.add(destination)
        flights_per_airport[origin] = flights_per_airport.get(origin, 0) + len(carrier_flights)
        flights_per_airport[destination] = flights_per_airport.get(destination, 0) + len(carrier_flights)

        for f in carrier_flights:
            aircraft_used.add(f["aircraft"])

        routes_operated.append({
            "origin":           origin,
            "destination":      destination,
            "daily_flights":    len(carrier_flights),
        })

    if not routes_operated:
        # Ritorniamo un risultato "vuoto" con la lista dei carrier validi, così il LLM può ricalibrare la query senza far fallire il tool.
        all_carriers = sorted({f["carrier"] for flights in ROUTE_NETWORK.values() for f in flights})
        return {
            "carrier_code":         code,
            "found":                False,
            "message":              f"Compagnia '{code}' non opera sulla rete monitorata.",
            "available_carriers":   all_carriers,
        }

    # Hub principale = aeroporto con il maggior numero di voli (in + out) della compagnia
    hub_iata, hub_count = max(flights_per_airport.items(), key=lambda x: x[1])

    # rotte più trafficate prima
    routes_operated.sort(key=lambda r: r["daily_flights"], reverse=True)

    return {
        "carrier_code":         code,
        "found":                True,
        "airports_served":      sorted(airports_served),
        "total_routes":         len(routes_operated),
        "total_daily_flights":  sum(r["daily_flights"] for r in routes_operated),
        "hub_iata":             hub_iata,
        "hub_city":             AIRPORTS[hub_iata]["city"],
        "hub_daily_flights":    hub_count,
        "aircraft_types":       sorted(aircraft_used),
        "routes":               routes_operated,
    }


@tool
def get_aircraft_info(aircraft_code: str) -> dict:
    if aircraft_code not in AIRCRAFT_DATA:
        available = sorted(AIRCRAFT_DATA.keys())
        raise ValueError(f"Velivolo '{aircraft_code}' non riconosciuto. Disponibili: {available}")

    info = AIRCRAFT_DATA[aircraft_code]

    # cerca tutte le rotte sulla rete dove questo velivolo è effettivamente operato
    routes_using = []
    for (origin, destination), flights in ROUTE_NETWORK.items():
        count = sum(1 for f in flights if f["aircraft"] == aircraft_code)
        if count > 0:
            routes_using.append({
                "origin":                       origin,
                "destination":                  destination,
                "flights_using_this_aircraft":  count,
            })

    # rotte con più voli per questo velivolo prima
    routes_using.sort(key=lambda r: r["flights_using_this_aircraft"], reverse=True)

    return {
        "aircraft_code":    aircraft_code,
        "manufacturer":     info["manufacturer"],
        "category":         info["category"],
        "typical_seats":    info["seats"],
        "range_km":         info["range_km"],
        "cruise_speed_kmh": info["cruise_kmh"],
        "routes_in_use":    len(routes_using),
        "top_routes":       routes_using[:5],
    }


@tool
def get_airport_embeddings(iata_code: str) -> dict:
    # Ritorna l'embedding vettoriale di un aeroporto (rappresentazione numerica per ML/similarity search).
    code = iata_code.upper()
    if code not in AIRPORTS:
        raise ValueError(f"Aeroporto '{code}' non trovato nel database")

    # Lazy: genera gli embedding solo alla prima chiamata di QUESTO processo worker
    embeddings_map = get_airport_embeddings_lazy()

    if not embeddings_map:
        return {
            "iata":      code,
            "city":      AIRPORTS[code]["city"],
            "message":   "Embeddings non disponibili (fastembed non installato)",
            "embedding": None,
        }

    embedding = embeddings_map.get(code)
    if embedding is None:
        return {"iata": code, "message": "Embedding non generato", "embedding": None}

    return {
        "iata":              code,
        "city":              AIRPORTS[code]["city"],
        "country":           AIRPORTS[code]["country"],
        "tier":              AIRPORTS[code]["tier"],
        "embedding_dim":     len(embedding),
        "embedding_sample":  embedding[:5],  # primi 5 valori per leggibilità
        "embedding_full":    embedding,      # vettore completo (384 dimensioni)
    }


@tool
def get_aircraft_embeddings(aircraft_code: str) -> dict:
    # Ritorna l'embedding vettoriale di un velivolo (rappresentazione numerica per ML/similarity search).
    code = aircraft_code
    if code not in AIRCRAFT_DATA:
        available = sorted(AIRCRAFT_DATA.keys())
        raise ValueError(f"Velivolo '{code}' non riconosciuto. Disponibili: {available}")

    # Lazy: genera gli embedding solo alla prima chiamata di QUESTO processo worker
    embeddings_map = get_aircraft_embeddings_lazy()

    if not embeddings_map:
        return {
            "aircraft_code":    code,
            "message":          "Embeddings non disponibili (fastembed non installato)",
            "embedding":        None,
        }

    embedding = embeddings_map.get(code)
    if embedding is None:
        return {"aircraft_code": code, "message": "Embedding non generato", "embedding": None}

    info = AIRCRAFT_DATA[code]
    return {
        "aircraft_code":     code,
        "manufacturer":      info["manufacturer"],
        "category":          info["category"],
        "embedding_dim":     len(embedding),
        "embedding_sample":  embedding[:5],  # primi 5 valori per leggibilità
        "embedding_full":    embedding,      # vettore completo (384 dimensioni)
    }
