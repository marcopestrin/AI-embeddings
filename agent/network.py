import math
import random
from datetime import datetime, timedelta
from embeddings import generate_airport_embeddings, generate_aircraft_embeddings

# Cache vuote all'avvio: verranno popolate al primo accesso
_AIRPORT_EMBEDDINGS_CACHE: dict[str, list[float]] = {}
_AIRCRAFT_EMBEDDINGS_CACHE: dict[str, list[float]] = {}

def get_airport_embeddings_lazy() -> dict[str, list[float]]:
    if not _AIRPORT_EMBEDDINGS_CACHE:
        _AIRPORT_EMBEDDINGS_CACHE.update(generate_airport_embeddings(AIRPORTS))
    return _AIRPORT_EMBEDDINGS_CACHE


def get_aircraft_embeddings_lazy() -> dict[str, list[float]]:
    if not _AIRCRAFT_EMBEDDINGS_CACHE:
        _AIRCRAFT_EMBEDDINGS_CACHE.update(generate_aircraft_embeddings(AIRCRAFT_DATA))
    return _AIRCRAFT_EMBEDDINGS_CACHE


# aeroporti monitorati. Tier 1 = mega-hub, 2 = major, 3 = regionale.
AIRPORTS = {
    "FCO": {"name": "Roma Fiumicino",                  "city": "Roma",              "country": "IT", "lat": 41.8003,  "lon": 12.2389,   "tier": 2},
    "LHR": {"name": "London Heathrow",                 "city": "Londra",            "country": "UK", "lat": 51.4700,  "lon": -0.4543,   "tier": 1},
    "BNE": {"name": "Brisbane Airport",                "city": "Brisbane",          "country": "AU", "lat": -27.3842, "lon": 153.1175,  "tier": 3},
    "CPT": {"name": "Cape Town International",         "city": "Città del Capo",    "country": "ZA", "lat": -33.9648, "lon": 18.6017,   "tier": 3},
    "DXB": {"name": "Dubai International",             "city": "Dubai",             "country": "AE", "lat": 25.2532,  "lon": 55.3657,   "tier": 1},
    "BKK": {"name": "Suvarnabhumi",                    "city": "Bangkok",           "country": "TH", "lat": 13.6900,  "lon": 100.7501,  "tier": 3},
    # "MEX": {"name": "Benito Juárez",                   "city": "Città del Messico", "country": "MX", "lat": 19.4363,  "lon": -99.0721,  "tier": 3},
    # "YVR": {"name": "Vancouver International",         "city": "Vancouver",         "country": "CA", "lat": 49.1939,  "lon": -123.1844, "tier": 3},
    # "IST": {"name": "Istanbul Airport",                "city": "Istanbul",          "country": "TR", "lat": 41.2753,  "lon": 28.7519,   "tier": 1},
    "DFW": {"name": "Dallas/Fort Worth International", "city": "Dallas",            "country": "US", "lat": 32.8998,  "lon": -97.0403,  "tier": 2},
    # "CAI": {"name": "Cairo International",             "city": "Cairo",             "country": "EG", "lat": 30.1219,  "lon": 31.4055,   "tier": 2},
    # "SIN": {"name": "Singapore Changi",                "city": "Singapore",         "country": "SG", "lat": 1.3644,   "lon": 103.9915,  "tier": 1},
    "KIX": {"name": "Kansai International",            "city": "Osaka",             "country": "JP", "lat": 34.4547,  "lon": 135.2306,  "tier": 2},
    "MEL": {"name": "Tullamarine Airport",             "city": "Melbourne",         "country": "AU", "lat": -37.6733, "lon": 144.8412,  "tier": 2},
    "ICN": {"name": "Incheon International",           "city": "Seoul",             "country": "KR", "lat": 37.4602,  "lon": 126.4407,  "tier": 1},
    "AMS": {"name": "Amsterdam Airport Schiphol",      "city": "Amsterdam",         "country": "NL", "lat": 52.3086,  "lon": 4.7639,    "tier": 1},
    "PHX": {"name": "Phoenix Sky Harbor",              "city": "Phoenix",           "country": "US", "lat": 33.4342,  "lon": -112.0015, "tier": 2},
}

# Codici IATA delle compagnie aeree.
HOME_CARRIERS = {
    "LHR": ["BA"], # British Airways
    "CPT": ["SA"], # South African Airways
    "DXB": ["EK"], # Emirates
    "BKK": ["TG"], # Thai Airways
    # "MEX": ["AM"], # Aeroméxico
    # "YVR": ["AC"], # Air Canada
    # "IST": ["TK"], # Turkish Airlines
    # "CAI": ["MS"], # EgyptAir
    # "SIN": ["SQ"], # Singapore Airlines
    "KIX": ["NH"], # ANA
    "MEL": ["QF"], # Qantas
    "BNE": ["QF"], # Qantas
    "FCO": ["AZ"], # ITA Airways
    "ICN": ["KE"], # Korean Air
    "AMS": ["KL"], # KLM
    "DFW": ["AA"], # American Airlines
    "PHX": ["AA"], # American Airlines
}

AIRCRAFT_DATA = {
    "A320":                 {"manufacturer": "Airbus", "category": "narrow-body", "seats": 165, "range_km": 6500,  "cruise_kmh": 833},
    "A321neo":              {"manufacturer": "Airbus", "category": "narrow-body", "seats": 200, "range_km": 7400,  "cruise_kmh": 833},
    "A330-300":             {"manufacturer": "Airbus", "category": "wide-body",   "seats": 295, "range_km": 11750, "cruise_kmh": 871},
    "A350-900":             {"manufacturer": "Airbus", "category": "wide-body",   "seats": 315, "range_km": 15000, "cruise_kmh": 903},
    "A380-800":             {"manufacturer": "Airbus", "category": "wide-body",   "seats": 555, "range_km": 14800, "cruise_kmh": 903},
    "B737 MAX 8":           {"manufacturer": "Boeing", "category": "narrow-body", "seats": 178, "range_km": 6570,  "cruise_kmh": 839},
    "B777-300ER":           {"manufacturer": "Boeing", "category": "wide-body",   "seats": 396, "range_km": 13649, "cruise_kmh": 905},
    "B787-9 Dreamliner":    {"manufacturer": "Boeing", "category": "wide-body",   "seats": 296, "range_km": 14140, "cruise_kmh": 903},
}

CARRIER_FLEETS = {
    "AZ": ["A320", "A321neo", "A330-300", "A350-900"],
    "BA": ["A320", "A321neo", "A350-900", "A380-800", "B777-300ER", "B787-9 Dreamliner"],
    "QF": ["A330-300", "A380-800", "B787-9 Dreamliner"],
    "SA": ["A320", "A330-300"],
    "EK": ["A350-900", "A380-800", "B777-300ER"],
    "TG": ["A320", "A330-300", "A350-900", "A380-800", "B777-300ER", "B787-9 Dreamliner"],
    "AM": ["B737 MAX 8", "B787-9 Dreamliner"],
    "AC": ["A320", "A321neo", "A330-300", "B737 MAX 8", "B777-300ER", "B787-9 Dreamliner"],
    "TK": ["A320", "A321neo", "A330-300", "A350-900", "B737 MAX 8", "B777-300ER", "B787-9 Dreamliner"],
    "AA": ["A320", "A321neo", "B737 MAX 8", "B777-300ER", "B787-9 Dreamliner"],
    "MS": ["A320", "A321neo", "A330-300", "A350-900", "B737 MAX 8", "B777-300ER", "B787-9 Dreamliner"],
    "SQ": ["A350-900", "A380-800", "B737 MAX 8", "B777-300ER"],
    "NH": ["A320", "A321neo", "A380-800", "B737 MAX 8", "B777-300ER", "B787-9 Dreamliner"],
    "KE": ["A330-300", "A350-900", "A380-800", "B737 MAX 8", "B777-300ER", "B787-9 Dreamliner"],
    "KL": ["A330-300", "B737 MAX 8", "B777-300ER", "B787-9 Dreamliner"],
}


def _pick_aircraft_for_route(distance_km: float, carrier: str, rng: random.Random) -> str:
    if distance_km > 9000:
        distance_candidates = ["A350-900", "B777-300ER", "B787-9 Dreamliner", "A380-800"]
    elif distance_km > 4000:
        distance_candidates = ["A330-300", "B787-9 Dreamliner", "A350-900"]
    else:
        distance_candidates = ["A320", "A321neo", "B737 MAX 8"]
    
    fleet = CARRIER_FLEETS.get(carrier, [])
    compatible = [a for a in distance_candidates if a in fleet]

    if compatible:
        return rng.choice(compatible)
    
    if fleet:
        return rng.choice(fleet)
    
    return rng.choice(distance_candidates)


def _haversine_km(o: str, d: str) -> float:
    # Distanza ortodromica in km tra due aeroporti. Serve a stimare la durata del volo (km / velocità di crociera).
    a, b = AIRPORTS[o], AIRPORTS[d]

    # le funzioni trig di math vogliono RADIANTI, le coordinate sono in gradi
    lat1, lon1, lat2, lon2 = map(math.radians, [a["lat"], a["lon"], b["lat"], b["lon"]])

    # differenze tra le due posizioni
    dlat, dlon = lat2 - lat1, lon2 - lon1

    # formula haversine: h = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2

    # distanza = 2·R·asin(√h), con R = raggio terrestre medio = 6371 km
    return 2 * 6371 * math.asin(math.sqrt(h))


def _build_network() -> dict[tuple[str, str], list[dict]]:
    rng = random.Random(42)  # generatore deterministico. Random seed fisso (42) -> la rete è riproducibile a ogni esecuzione.
    routes: dict[tuple[str, str], list[dict]] = {}  # accumulator
    iatas = list(AIRPORTS.keys())  # i 17 codici IATA

    # Iteriamo su tutte le 272 coppie ordinate (17 × 17 - 17 diagonali)
    for o in iatas:
        for d in iatas:
            if o == d:
                continue  # un aeroporto non vola "verso se stesso"

            # simulazione di frequenze giornaliere
            # tier 1 (mega-hub) -> si collegano a tutti
            # tier 2 (major)    -> collegamenti agli hub
            # tier 3 (regionale)-> quasi solo verso hub
            t_o, t_d = AIRPORTS[o]["tier"], AIRPORTS[d]["tier"]
            t_min, t_max = min(t_o, t_d), max(t_o, t_d)
            if t_min == 1 and t_max == 1:
                freq = rng.randint(4, 8)        # hub ↔ hub
            elif t_min == 1 and t_max == 2:
                freq = rng.randint(3, 5)        # hub ↔ major
            elif t_min == 1 and t_max == 3:
                freq = rng.randint(1, 3)        # hub ↔ regional
            elif t_min == 2 and t_max == 2:
                freq = rng.randint(1, 2)        # major ↔ major
            elif t_min == 2 and t_max == 3:
                freq = rng.choice([0, 0, 1])    # major ↔ regional: raro
            else:
                freq = 0                         # regional ↔ regional: nessuno

            if freq == 0:
                continue  # questa coppia non ha collegamenti diretti

            # stima durata di volo
            distance = _haversine_km(o, d)
            # ~800 km/h velocità di crociera + 40-50 min di taxi/holding
            duration_min = int(distance / 800 * 60) + rng.randint(40, 50)

            # generazione dei singoli voli del giorno
            flights = []
            for i in range(freq):
                # Distribuiamo le partenze nelle 24h: con freq=4 → 03:xx, 09:xx, 15:xx, 21:xx
                base_hour = (3 + i * (24 // max(freq, 1))) % 24
                minute = rng.choice([0, 5, 15, 30, 45, 50])
                dep_dt = datetime(2026, 1, 1, base_hour, minute) # data fissa, conta solo l'ora
                arr_dt = dep_dt + timedelta(minutes=duration_min)

                # Vettore: 60% home dell'origine, 40% home della destinazione
                carrier = rng.choice(HOME_CARRIERS[o] if rng.random() < 0.6 else HOME_CARRIERS[d])

                # Velivolo: scelta intersecando candidati per distanza + flotta reale del carrier
                aircraft = _pick_aircraft_for_route(distance, carrier, rng)

                # Atterraggio dopo mezzanotte Marchiamolo con "+1" (o "+2" per voli ultra-long)
                next_day_offset = arr_dt.day - dep_dt.day
                arr_str = arr_dt.strftime("%H:%M") + (f"+{next_day_offset}" if next_day_offset else "")

                flights.append({
                    "flight_no": f"{carrier}{rng.randint(100, 9999)}",
                    "carrier": carrier,
                    "aircraft": aircraft,
                    "departure_local": dep_dt.strftime("%H:%M"),
                    "arrival_local": arr_str,
                    "duration_min": duration_min,
                })

            # Ordiniamo i voli per orario di partenza crescente
            flights.sort(key=lambda f: f["departure_local"])
            routes[(o, d)] = flights

    # Output: dict { (origin, dest) -> [list di voli] }.
    return routes


# La rete viene costruita 1 sola volta, al caricamento del modulo.
ROUTE_NETWORK = _build_network()



def print_route_network(network: dict[tuple[str, str], list[dict]]) -> None:
    # Stampa tutte le rotte della rete in formato tabellare compatto. Una riga per rotta: origin → destination | voli/giorno | carriers | velivoli.
   
    # Header
    print("\n" + "=" * 90)
    print(f"ROTTE GENERATE NELLA RETE ({len(network)} totali)")
    print("=" * 90)
    print(f"{'Rotta':<14}{'Voli':>5}  {'Carriers':<20}{'Velivoli'}")
    print("-" * 90)

    # Ordiniamo per origin, poi per destination — output stabile e leggibile
    for (origin, destination) in sorted(network.keys()):
        flights = network[(origin, destination)]
        carriers = sorted({f["carrier"] for f in flights})
        aircraft = sorted({f["aircraft"] for f in flights})
        route_label = f"{origin} → {destination}"
        carriers_str = ", ".join(carriers)
        aircraft_str = ", ".join(aircraft)
        print(f"{route_label:<14}{len(flights):>5}  {carriers_str:<20}{aircraft_str}")

    # Riepilogo finale
    total_flights = sum(len(flights) for flights in network.values())
    print("-" * 90)
    print(f"Totale: {len(network)} rotte, {total_flights} voli giornalieri")
    print("=" * 90 + "\n")


print_route_network(ROUTE_NETWORK)
