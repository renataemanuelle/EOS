from EOSpython import EOS
import pandas as pd
import numpy as np
import os

from types import SimpleNamespace

def save_instance(base_path: str, x_data):
    # --- salva tabelas sem perda (CSV é opcional) ---
    # Recomendado: pickle do pandas preserva listas/datetimes
    x_data.df.to_pickle(base_path + "_df.pkl")
    x_data.pf_df.to_pickle(base_path + "_pf_df.pkl")

    # (opcional) CSVs só para inspeção humana
    x_data.df.to_csv(base_path + "_df.csv", index=False)
    x_data.pf_df.to_csv(base_path + "_pf_df.csv", index=False)

    with open(base_path + "_sats.txt", "w", encoding="utf-8") as f:
        for sat in x_data.sats:
            f.write(str(sat) + "\n")

    # --- salva estado do LPP ---
    LPP = x_data.LPP

    # garante que os 9 atributos existem
    keys = ["B","F","LHS","RHS","eLHS","eRHS","performance_df","stereo","strips"]
    missing = [k for k in keys if not hasattr(LPP, k)]
    if missing:
        raise AttributeError(f"LPP sem atributos esperados: {missing}")

    # salva performance_df do LPP (por segurança; é o que o solve usa)
    LPP.performance_df.to_pickle(base_path + "_lpp_performance_df.pkl")

    # LHS (cvxopt.spmatrix) -> COO + shape
    LHS = LPP.LHS
    I = np.array(LHS.I, dtype=np.int64)
    J = np.array(LHS.J, dtype=np.int64)
    V = np.array(LHS.V, dtype=np.float64)
    shape = np.array(LHS.size, dtype=np.int64)  # (rows, cols)

    np.savez_compressed(
        base_path + "_lpp_state.npz",
        B=np.asarray(LPP.B),
        F=np.asarray(LPP.F),
        RHS=np.asarray(LPP.RHS),
        eLHS=np.asarray(LPP.eLHS),
        eRHS=np.asarray(LPP.eRHS),
        stereo=np.asarray(LPP.stereo),
        strips=np.asarray(LPP.strips),
        LHS_I=I, LHS_J=J, LHS_V=V, LHS_shape=shape
    )

    print("[OK] Salvo:",
          base_path + "_df.pkl,",
          base_path + "_pf_df.pkl,",
          base_path + "_sats.txt,",
          base_path + "_lpp_performance_df.pkl,",
          base_path + "_lpp_state.npz")



# Gerar cenário

sat_TLEs = [38755, 40053]

horizon_start = [2026,6,13,9,40]
horizon = 8
n_requests = 250

# Prefixo eoss_instance_ (padrão do repo / RKO / eos_read_scenario).
# Formato: eoss_instance_{requisições}_{horizonte}h_{n_sats}sat
n_sats = len(sat_TLEs)
instance_tag = f"{n_requests}_{horizon}h_{n_sats}sat"
base_path = os.path.join("Instances", f"eoss_instance_{instance_tag}")
os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
print(f"[INFO] base_path = {base_path}")

database, map_file = EOS.customer_db(number_of_requests_0=n_requests, map_generation=False)

x_data = EOS.scenario(
    customer_database=database,
    m=map_file,
    seconds_gran=10,
    NORAD_ids=sat_TLEs,
    weather_real=False,
    simplify=False,
    schedule_start=horizon_start,
    hours_horizon=horizon,
    map_generation=False,
)

# Calcular score do cenário (ELECTRE-III) e salvar no pf_df

score_scenario = EOS.score_requests(
    x_data,
    scoring_method=2,                 # 2 = ELECTRE
)

x_data.pf_df["score_scenario"] = score_scenario

# (Opcional, mas recomendado) salvar também metadados do scoring
x_data.pf_df["score_method"] = "ELECTRE-III"
x_data.pf_df["score_alpha"] = 1

x_data.df.to_csv(base_path + "_df.csv", index=False)
x_data.pf_df.to_csv(base_path + "_pf_df.csv", index=False)

with open(base_path + "_sats.txt", "w", encoding="utf-8") as f:
    for sat in x_data.sats:
        f.write(str(sat) + "\n")

with open(base_path + "_info.txt", "w", encoding="utf-8") as f:
    f.write("Cenário de simulação\n")
    f.write(f"Início: {horizon_start}\n")
    f.write(f"Horizonte: {horizon} horas\n")
    f.write(f"NORADs usados: {sat_TLEs}\n")
    f.write(f"Tamanho de df: {len(x_data.df)} linhas\n")
    f.write(f"Instância: eoss_instance_{instance_tag}\n")


save_instance(base_path, x_data)

# -------------------------------------------------------------
# Teste de leitura e solução ()


res = EOS.solve(
    x_data,
    solution_method="DAG",
    use_existing_score=True,
    score_column="score_scenario"
)

print("Score usado (primeiros 5):", res.score[:5])
print("x (qtde selecionados):", int(np.sum(res.x)))
print("obj:", float(-res.score @ res.x))
print("time:", res.time)

res_cached = EOS.solve(
    x_data,
    solution_method="DAG",
    scoring_method=2,                 # 2 = ELECTRE
    use_existing_score=True,
    score_column="score_scenario"
)

print("Obj cached:", float(-res_cached.score @ res_cached.x))

res_recalc = EOS.solve(
    x_data,
    solution_method="DAG",
    scoring_method=2,                 # 2 = ELECTRE
    use_existing_score=False,
)

print("Obj recalc:", float(-res_recalc.score @ res_recalc.x))
