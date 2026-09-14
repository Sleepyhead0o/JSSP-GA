import random
import time
import os
import statistics
from collections import Counter

import matplotlib.pyplot as plt


def les_benchmark_fil(filsti):
    with open(filsti, "r") as fil:
        linjer = [
            linje.strip()
            for linje in fil
            if linje.strip()
        ]

    antall_jobber, antall_maskiner = map(
        int,
        linjer[0].split()
    )

    jobber = []

    for linje in linjer[1:]:
        verdier = list(map(int, linje.split()))
        operasjoner = []

        for i in range(0, len(verdier), 2):
            maskin = verdier[i]
            prosesstid = verdier[i + 1]

            operasjoner.append(
                (maskin, prosesstid)
            )

        jobber.append(operasjoner)

    return jobber, antall_maskiner

# Operation-based representation
def opprett_kromosom(jobber):
    kromosom = []

    for jobb_id, operasjoner in enumerate(jobber):
        kromosom.extend([jobb_id] * len(operasjoner))

    random.shuffle(kromosom)

    return kromosom


# Initial Population
def opprett_populasjon(jobber, populasjonsstorrelse):
    return [
        opprett_kromosom(jobber)
        for _ in range(populasjonsstorrelse)
    ]


# Semi-Active Schedule Building Algorithm (SBA) / Decoder
def dekod_kromosom(kromosom, jobber, antall_maskiner):
    antall_jobber = len(jobber)

    jobb_klar_tid = [0] * antall_jobber
    maskin_klar_tid = [0] * antall_maskiner
    neste_operasjon = [0] * antall_jobber

    tidsplan = []

    for jobb_id in kromosom:
        operasjon_id = neste_operasjon[jobb_id]

        maskin_id, prosesstid = jobber[jobb_id][operasjon_id]

        starttid = max(
            jobb_klar_tid[jobb_id],
            maskin_klar_tid[maskin_id]
        )

        sluttid = starttid + prosesstid

        tidsplan.append({
            "jobb": jobb_id,
            "operasjon": operasjon_id,
            "maskin": maskin_id,
            "start": starttid,
            "slutt": sluttid,
            "prosesstid": prosesstid
        })

        jobb_klar_tid[jobb_id] = sluttid
        maskin_klar_tid[maskin_id] = sluttid
        neste_operasjon[jobb_id] += 1

    makespan = max(jobb_klar_tid)

    return tidsplan, makespan


# Fitness Function / Makespan Calculation
def beregn_makespan(kromosom, jobber, antall_maskiner):
    _, makespan = dekod_kromosom(
        kromosom,
        jobber,
        antall_maskiner
    )

    return makespan


# Tournament Selection
def turneringsseleksjon(
    populasjon,
    fitnessverdier,
    turneringsstorrelse=3
):
    kandidater = random.sample(
        range(len(populasjon)),
        turneringsstorrelse
    )

    beste_indeks = min(
        kandidater,
        key=lambda indeks: fitnessverdier[indeks]
    )

    return populasjon[beste_indeks].copy()


# Operation Identification for PPX
def kromosom_til_operasjoner(kromosom):
    tellere = {}
    operasjoner = []

    for jobb_id in kromosom:
        operasjon_id = tellere.get(jobb_id, 0)

        operasjoner.append(
            (jobb_id, operasjon_id)
        )

        tellere[jobb_id] = operasjon_id + 1

    return operasjoner


# Precedence Preservative Crossover (PPX)
def ppx_kryssing(forelder_1, forelder_2):
    operasjoner_1 = kromosom_til_operasjoner(forelder_1)
    operasjoner_2 = kromosom_til_operasjoner(forelder_2)

    liste_1 = operasjoner_1.copy()
    liste_2 = operasjoner_2.copy()

    barn_operasjoner = []

    while liste_1:
        velg_forelder = random.randint(0, 1)

        if velg_forelder == 0:
            valgt_operasjon = liste_1[0]
        else:
            valgt_operasjon = liste_2[0]

        barn_operasjoner.append(valgt_operasjon)

        liste_1.remove(valgt_operasjon)
        liste_2.remove(valgt_operasjon)

    barn = [
        jobb_id
        for jobb_id, operasjon_id in barn_operasjoner
    ]

    return barn


# Swap Mutation
def byttemutasjon(kromosom):
    barn = kromosom.copy()

    if len(barn) < 2:
        return barn

    indeks_1, indeks_2 = random.sample(
        range(len(barn)),
        2
    )

    barn[indeks_1], barn[indeks_2] = (
        barn[indeks_2],
        barn[indeks_1]
    )

    return barn


# Elitism
def hent_elite(
    populasjon,
    fitnessverdier,
    elite_antall
):
    rangerte_indekser = sorted(
        range(len(populasjon)),
        key=lambda indeks: fitnessverdier[indeks]
    )

    elite = [
        populasjon[indeks].copy()
        for indeks in rangerte_indekser[:elite_antall]
    ]

    return elite


# Replacement / New Generation
def lag_neste_generasjon(
    populasjon,
    jobber,
    antall_maskiner,
    kryssingssannsynlighet,
    mutasjonssannsynlighet,
    turneringsstorrelse,
    elite_antall
):
    fitnessverdier = [
        beregn_makespan(
            individ,
            jobber,
            antall_maskiner
        )
        for individ in populasjon
    ]

    ny_populasjon = hent_elite(
        populasjon,
        fitnessverdier,
        elite_antall
    )

    while len(ny_populasjon) < len(populasjon):
        forelder_1 = turneringsseleksjon(
            populasjon,
            fitnessverdier,
            turneringsstorrelse
        )

        forelder_2 = turneringsseleksjon(
            populasjon,
            fitnessverdier,
            turneringsstorrelse
        )

        if random.random() < kryssingssannsynlighet:
            barn = ppx_kryssing(
                forelder_1,
                forelder_2
            )
        else:
            barn = forelder_1.copy()

        if random.random() < mutasjonssannsynlighet:
            barn = byttemutasjon(barn)

        assert Counter(barn) == Counter(forelder_1)

        ny_populasjon.append(barn)

    return ny_populasjon


# Genetic Algorithm
def genetisk_algoritme(
    jobber,
    antall_maskiner,
    populasjonsstorrelse=100,
    antall_generasjoner=500,
    kryssingssannsynlighet=0.8,
    mutasjonssannsynlighet=0.1,
    turneringsstorrelse=3,
    elite_antall=2,
    vis_fremdrift=False
):
    populasjon = opprett_populasjon(
        jobber,
        populasjonsstorrelse
    )

    beste_kromosom = None
    beste_makespan = float("inf")

    historikk = []
    generasjon_for_beste = 0

    for generasjon in range(antall_generasjoner):
        fitnessverdier = [
            beregn_makespan(
                individ,
                jobber,
                antall_maskiner
            )
            for individ in populasjon
        ]

        generasjon_beste_makespan = min(
            fitnessverdier
        )

        beste_indeks = fitnessverdier.index(
            generasjon_beste_makespan
        )

        if generasjon_beste_makespan < beste_makespan:
            beste_makespan = generasjon_beste_makespan
            beste_kromosom = populasjon[beste_indeks].copy()
            generasjon_for_beste = generasjon

        historikk.append(beste_makespan)

        if vis_fremdrift:
            print(
                f"Generasjon {generasjon + 1}: "
                f"beste makespan = {beste_makespan}"
            )

        populasjon = lag_neste_generasjon(
            populasjon=populasjon,
            jobber=jobber,
            antall_maskiner=antall_maskiner,
            kryssingssannsynlighet=kryssingssannsynlighet,
            mutasjonssannsynlighet=mutasjonssannsynlighet,
            turneringsstorrelse=turneringsstorrelse,
            elite_antall=elite_antall
        )

    beste_tidsplan, _ = dekod_kromosom(
        beste_kromosom,
        jobber,
        antall_maskiner
    )

    return {
        "beste_kromosom": beste_kromosom,
        "beste_makespan": beste_makespan,
        "beste_tidsplan": beste_tidsplan,
        "historikk": historikk,
        "generasjon_for_beste": generasjon_for_beste
    }


# Independent Runs / Statistical Evaluation
def kjor_eksperiment(
    jobber,
    antall_maskiner,
    antall_kjoringer=20,
    **ga_parametere
):
    makespan_resultater = []
    kjoretider = []
    konvergens_generasjoner = []

    beste_resultat = None

    for kjoring in range(antall_kjoringer):
        starttid = time.perf_counter()

        resultat = genetisk_algoritme(
            jobber,
            antall_maskiner,
            **ga_parametere
        )

        sluttid = time.perf_counter()

        kjoretid = sluttid - starttid

        makespan_resultater.append(
            resultat["beste_makespan"]
        )

        kjoretider.append(kjoretid)

        konvergens_generasjoner.append(
            resultat["generasjon_for_beste"]
        )

        if (
            beste_resultat is None
            or resultat["beste_makespan"]
            < beste_resultat["beste_makespan"]
        ):
            beste_resultat = resultat

        print(
            f"Kjøring {kjoring + 1}/{antall_kjoringer}: "
            f"Makespan = {resultat['beste_makespan']}, "
            f"Tid = {kjoretid:.3f} sek"
        )

    statistikk = {
        "beste": min(makespan_resultater),
        "verste": max(makespan_resultater),
        "gjennomsnitt": statistics.mean(
            makespan_resultater
        ),
        "standardavvik": (
            statistics.stdev(makespan_resultater)
            if len(makespan_resultater) > 1
            else 0
        ),
        "gjennomsnittlig_kjoretid": statistics.mean(
            kjoretider
        ),
        "gjennomsnittlig_konvergens": statistics.mean(
            konvergens_generasjoner
        )
    }

    return statistikk, beste_resultat


# Schedule Output
def skriv_ut_tidsplan(tidsplan):
    print()

    print(
        f"{'Jobb':<8}"
        f"{'Operasjon':<12}"
        f"{'Maskin':<10}"
        f"{'Start':<10}"
        f"{'Slutt':<10}"
    )

    print("-" * 50)

    for operasjon in tidsplan:
        print(
            f"J{operasjon['jobb'] + 1:<7}"
            f"O{operasjon['operasjon'] + 1:<11}"
            f"M{operasjon['maskin'] + 1:<9}"
            f"{operasjon['start']:<10}"
            f"{operasjon['slutt']:<10}"
        )


# Gantt Chart
def tegn_gantt(
    tidsplan,
    antall_maskiner,
    tittel="JSSP Gantt-diagram"
):
    figur, akse = plt.subplots()

    for operasjon in tidsplan:
        maskin = operasjon["maskin"]
        start = operasjon["start"]

        varighet = (
            operasjon["slutt"]
            - operasjon["start"]
        )

        jobb = operasjon["jobb"]

        akse.barh(
            y=maskin,
            width=varighet,
            left=start
        )

        akse.text(
            start + varighet / 2,
            maskin,
            f"J{jobb + 1}",
            ha="center",
            va="center"
        )

    akse.set_yticks(
        range(antall_maskiner)
    )

    akse.set_yticklabels(
        [
            f"M{maskin + 1}"
            for maskin in range(antall_maskiner)
        ]
    )

    akse.set_xlabel("Tid")
    akse.set_ylabel("Maskin")
    akse.set_title(tittel)

    akse.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# Convergence Curve
def tegn_konvergens(historikk):
    plt.figure()

    plt.plot(
        range(1, len(historikk) + 1),
        historikk
    )

    plt.xlabel("Generasjon")
    plt.ylabel("Beste makespan")

    plt.title(
        "Konvergens for genetisk algoritme"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    resultater = []

    # Problem kategorier
    kategorier = {
        "Small": [
            "data/la01.txt",
            "data/la02.txt"
        ],
        "Medium": [
            "data/la16.txt",
            "data/la17.txt"
        ],
        "Large": [
            "data/la31.txt",
            "data/la32.txt"
        ]
    }

    for kategori, filer in kategorier.items():
        for filsti in filer:
            jobber, antall_maskiner = les_benchmark_fil(filsti)
            antall_jobber = len(jobber)

            resultat = genetisk_algoritme(
                jobber,
                antall_maskiner,
                vis_fremdrift=False
            )

            beste_makespan = resultat["beste_makespan"]

            instans = filsti.split("/")[-1].replace(".txt", "")

            resultater.append([
                kategori,
                instans,
                antall_jobber,
                antall_maskiner,
                beste_makespan
            ])

    print(
        f"{'Kategori':<15}"
        f"{'Instans':<15}"
        f"{'Jobber':<15}"
        f"{'Maskiner':<15}"
        f"{'Beste makespan':<15}"
    )

    print("-" * 85)

    for kategori, instans, jobber, maskiner, makespan in resultater:
        print(
            f"{kategori:<15}"
            f"{instans:<15}"
            f"{jobber:<15}"
            f"{maskiner:<15}"
            f"{makespan:<15}"
        )