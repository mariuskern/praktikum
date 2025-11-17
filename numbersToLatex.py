import json

filename = input("Dateiname: ")

with open(filename, "r") as f:
    data = json.load(f)

for method, d1 in data.items():
    for dataset, d2 in d1.items():
        # Accuracy
        a1 = round(d2["Accuracy@1"], 2)
        a5 = round(d2["Accuracy@5"], 2)
        # a10 = round(d2["Accuracy@10"], 2)

        # Precision mean
        p1 = round(d2["per_class"]["mean"]["Precision@1"], 2)
        p5 = round(d2["per_class"]["mean"]["Precision@5"], 2)
        # p10 = round(d2["per_class"]["mean"]["Precision@10"], 2)

        # Precision std
        ps1 = round(d2["per_class"]["std"]["Precision@1"], 2)
        ps5 = round(d2["per_class"]["std"]["Precision@5"], 2)
        # ps10 = round(d2["per_class"]["std"]["Precision@10"], 2)

        # Recall mean
        r1 = round(d2["per_class"]["mean"]["Recall@1"], 2)
        r5 = round(d2["per_class"]["mean"]["Recall@5"], 2)
        # r10 = round(d2["per_class"]["mean"]["Recall@10"], 2)

        # Recall std
        rs1 = round(d2["per_class"]["std"]["Recall@1"], 2)
        rs5 = round(d2["per_class"]["std"]["Recall@5"], 2)
        # rs10 = round(d2["per_class"]["std"]["Recall@10"], 2)

        # Ausgabe
        print(
            method + " " + dataset + ": "
            + f"{a1} & {a5} & " # {a10} & "
            + f"{p1} $\\pm$ {ps1} & {p5} $\\pm$ {ps5} & " # {p10}$\\pm${ps10} & "
            + f"{r1} $\\pm$ {rs1} & {r5} $\\pm$ {rs5}" # & {r10}$\\pm${rs10}"
        )
    
    print()