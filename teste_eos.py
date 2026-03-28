from EOSpython import EOS
import pandas as pd
import numpy as np

sat_TLEs = [38755, 40053]  # Satélites SPOT 6 e 7
criteria_w = [0.05, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.05]
qpv = [[0, 30, 1000], [0, 2, 40], [0, 10, 40], [0, 2, 15],
       [0, 1, 4], [0, 100, 20000], [0, 4, 10], [0, 0.5, 1]]

# Parâmetros de tempo
schedule_start = [2025, 8, 31, 0, 0]
hours_horizon = 2

# Geração da base de clientes
database, map_file = EOS.customer_db(number_of_requests_0=250)

print(database)
print(database.info())

# Geração do cenário com tempo e granularidade específicos
x_data = EOS.scenario(
    customer_database=database,
    m=map_file,
    seconds_gran=10,
    NORAD_ids=sat_TLEs,
    weather_real=False,
    simplify=False,
    schedule_start=schedule_start,
    hours_horizon=hours_horizon
)

# Caminho base do arquivo
base_path = "eoss_instance_20250831"

# Salvar df (tentativas válidas)
x_data.df.to_csv(base_path + "_df.csv", index=False)

# Salvar pf_df (dados adicionais, se aplicável)
x_data.pf_df.to_csv(base_path + "_pf_df.csv", index=False)

# Salvar lista de satélites
with open(base_path + "_sats.txt", "w", encoding="utf-8") as f:
    for sat in x_data.sats:
        f.write(str(sat) + "\n")

# Apenas para referência visual: salvar metadados como texto
with open(base_path + "_info.txt", "w", encoding="utf-8") as f:
    f.write("Cenário de simulação\n")
    f.write(f"Início: {schedule_start}\n")
    f.write(f"Horizonte: {hours_horizon} horas\n")
    f.write(f"NORADs usados: {sat_TLEs}\n")
    f.write(f"Tamanho de df: {len(x_data.df)} linhas\n")

# Solução heurística simples
print("Running heuristic...")
x_res1 = EOS.solve(
    x_data,
    scoring_method=2,
    solution_method="DAG",
    criteria_weights_l=criteria_w,
    threshold_parameters_l=qpv
)

# Visualização e avaliação
EOS.visualize(x_data, x_res1, 'EOS_example')
df = EOS.evaluate(x_data, x_res1)

# Impressão dos resultados
print(df.solution)
print(df.scenario)

# GLPK
# print("Running GLPK...")
# x_res2 = EOS.solve(x_data, scoring_method=3, solution_method = "GLPK",  #3=WSA
#                    criteria_weights_l = criteria_w, 
#                    threshold_parameters_l= qpv)

# # EOS.visualize(x_data, x_res2, 'EOS_example') #output is an interactive map called EOS_example.html saved in the wd

# df2 = EOS.evaluate(x_data, x_res2)
# print(df2.solution)
# print(df2.scenario)


#gurobi
# print("Running Gurobi...")
# x_res3 = EOS.solve(x_data, scoring_method=2, solution_method = "gurobi",
#                    criteria_weights_l = criteria_w,
#                    threshold_parameters_l= qpv)

# # EOS.visualize(x_data, x_res2, 'EOS_example') #output is an interactive map called EOS_example.html saved in the wd

# df3 = EOS.evaluate(x_data, x_res3)
# print(df3.solution)
# print(df3.scenario)

#PuLP
print("Running PuLP...")
x_res4 = EOS.solve(x_data, scoring_method=2, solution_method = "PuLP",
                   criteria_weights_l = criteria_w,
                   threshold_parameters_l= qpv)

# EOS.visualize(x_data, x_res2, 'EOS_example') #output is an interactive map called EOS_example.html saved in the wd

df4 = EOS.evaluate(x_data, x_res4)
print(df4.solution)
print(df4.scenario)

# import json

# def export_eos_scenario(x_data, output_path):
#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write("### EOSS Scenario Dump ###\n\n")
#         for key in dir(x_data):
#             if key.startswith("_"):
#                 continue
#             try:
#                 value = getattr(x_data, key)
#                 f.write(f"\n--- {key.upper()} ---\n")
#                 if isinstance(value, (list, tuple)):
#                     for item in value:
#                         f.write(json.dumps(item, default=str) + "\n")
#                 elif isinstance(value, dict):
#                     f.write(json.dumps(value, indent=2, default=str) + "\n")
#                 else:
#                     f.write(str(value) + "\n")
#             except Exception:
#                 continue

# # Salva o conteúdo do cenário em um arquivo TXT
# export_eos_scenario(x_data, "eoss_instance_20250613_1805.txt")


