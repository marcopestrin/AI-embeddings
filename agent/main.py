import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
from agentspan.agents import Agent, AgentRuntime
# Import per stampa distanze embeddings (vedi sotto, dentro __main__)
from network import get_airport_embeddings_lazy, get_aircraft_embeddings_lazy, ROUTE_NETWORK, print_route_network
from embeddings import print_embedding_distances
from tools import (list_routes_from, get_route_schedule, get_airport_centrality, find_connections, get_airline_network, get_aircraft_info, get_airport_embeddings, get_aircraft_embeddings)


flight_network_agent = Agent(
    name="flight_network_assistant",
    model=os.getenv("AGENTSPAN_MODEL", "anthropic/claude-haiku-4-5-20251001"),
    tools=[
        list_routes_from,
        get_route_schedule,
        get_airport_centrality,
        find_connections,
        get_airline_network,
        get_aircraft_info,
        get_airport_embeddings,
        get_aircraft_embeddings,
    ],
    instructions=(
        # "Sei un analista della rete di rotte aeree fra 17 aeroporti monitorati: FCO, LHR, DXB, IST, DFW, BKK, BNE, CPT, MEX, YVR, CAI, SIN, KIX, MEL, ICN, AMS, PHX."
        "Sei un analista della rete di rotte aeree fra 10 aeroporti monitorati: FCO, LHR, DXB, IST, DFW, BKK, BNE, CPT, MEX, YVR"
        "Ogni volta che si fa rierimento agli aeroporti, si intendono quelli appena citati. Non ci sono altri aeroporti. "
        "Sei esperto di: collegamenti diretti e indiretti, frequenze, vettori, velivoli, orari, "
        "centralità degli aeroporti, statistiche delle compagnie aeree, specifiche dei velivoli, e embedding vettoriali per similarity search. "
        "Quando elenchi rotte o aeroporti, ordina SEMPRE per traffico decrescente salvo richiesta esplicita di altro ordine. "
        "La durata di volo va espressa in minuti come numero intero se sotto i 60, altrimenti in ore e minuti. "
        "Se un orario di arrivo termina con '+1' o '+2', l'atterraggio è il giorno successivo. "
        "Per rotte indirette usa find_connections. Per analisi semantiche su aeroporti/velivoli usa gli embedding tool."
    ),
)


if __name__ == "__main__":
    # Stampa tutte le rotte generate (solo nel main process, non nei worker Conductor)
    print_route_network(ROUTE_NETWORK)
    print_embedding_distances(get_airport_embeddings_lazy(), get_aircraft_embeddings_lazy())

    with AgentRuntime() as runtime:
        # esempio di prompt che esercita più tool
        handle = runtime.start(
            flight_network_agent,
            "Dimmi le 3 rotte con scali che durano di più, specifica quanto durano i voli (inclusa durata scalo), e quali sono i velivoli usati su queste rotte. "
        )
        print(f"Run ID: {handle.run_id}")
        result = handle.join()

        if str(result.status).upper() == "FAILED":
            print(f"Errore: {result.error}")
        else:
            print(result.output.get("result") if isinstance(result.output, dict) else result.output)
            if result.token_usage:
                print(f"\nToken usati: {result.token_usage}")

