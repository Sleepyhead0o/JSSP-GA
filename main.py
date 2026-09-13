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


jobber, antall_maskiner = les_benchmark_fil(
    "data/la01.txt"
)

print("Antall jobber:", len(jobber))
print("Antall maskiner:", antall_maskiner)
print("Første jobb:", jobber[0])