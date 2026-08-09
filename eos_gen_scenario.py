from EOSpython import EOS
import pandas as pd
import numpy as np
import os
import argparse

from types import SimpleNamespace

# Frota offline (mesma ordem/IDs do TLE_CATALOG em EOS.py).
# n_sats=2 → [38755, 40053]; n_sats=3 → +38012; n_sats=4 → +39019
DEFAULT_FLEET = [38755, 40053, 38012, 39019]

DEFAULT_N_REQUESTS = 250
DEFAULT_HORIZON = 8
DEFAULT_N_SATS = 2
DEFAULT_HORIZON_START = [2026, 6, 13, 9, 40]


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
    keys = ["B", "F", "LHS", "RHS", "eLHS", "eRHS", "performance_df", "stereo", "strips"]
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Gera instância EOS e salva em Instances/eoss_instance_{N}_{H}h_{K}sat_*"
    )
    p.add_argument("n_requests", type=int, nargs="?", default=None,
                   help="Nº de requisições (ex.: 250)")
    p.add_argument("horizon", type=int, nargs="?", default=None,
                   help="Horizonte em horas (ex.: 8)")
    p.add_argument("n_sats", type=int, nargs="?", default=None,
                   help="Nº de satélites (ex.: 2). Usa os K primeiros de DEFAULT_FLEET "
                        "ou --norads.")
    p.add_argument("--norads", type=int, nargs="+", default=None,
                   help="Lista explícita de NORADs (ex.: --norads 38755 40053). "
                        "Sobrescreve n_sats/DEFAULT_FLEET.")
    p.add_argument("--start", type=int, nargs=5, default=None,
                   metavar=("Y", "M", "D", "H", "MIN"),
                   help="Início do horizonte [Y M D H MIN] (default: 2026 6 13 9 40)")
    p.add_argument("--instances-dir", type=str, default="Instances",
                   help="Pasta de saída (default: Instances)")
    p.add_argument("--map", action="store_true",
                   help="Gera mapas HTML (default: desligado)")
    p.add_argument("--test-solve", action="store_true",
                   help="Após gerar, roda smoke-tests DAG (mais lento; default: off)")
    return p.parse_args()


def generate_instance(n_requests: int,
                      horizon: int,
                      sat_TLEs: list[int],
                      horizon_start: list[int],
                      instances_dir: str = "Instances",
                      map_generation: bool = False,
                      test_solve: bool = False):
    n_sats = len(sat_TLEs)
    instance_tag = f"{n_requests}_{horizon}h_{n_sats}sat"
    base_path = os.path.join(instances_dir, f"eoss_instance_{instance_tag}")
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)

    print(f"[INFO] base_path     = {base_path}")
    print(f"[INFO] n_requests    = {n_requests}")
    print(f"[INFO] horizon       = {horizon}h")
    print(f"[INFO] sat_TLEs      = {sat_TLEs}")
    print(f"[INFO] horizon_start = {horizon_start}")
    print(f"[INFO] map_generation= {map_generation}")

    database, map_file = EOS.customer_db(
        number_of_requests_0=n_requests,
        map_generation=map_generation,
    )

    x_data = EOS.scenario(
        customer_database=database,
        m=map_file,
        seconds_gran=10,
        NORAD_ids=sat_TLEs,
        weather_real=False,
        simplify=False,
        schedule_start=horizon_start,
        hours_horizon=horizon,
        map_generation=map_generation,
    )

    # Calcular score do cenário (ELECTRE-III) e salvar no pf_df
    score_scenario = EOS.score_requests(
        x_data,
        scoring_method=2,  # 2 = ELECTRE
    )
    x_data.pf_df["score_scenario"] = score_scenario
    x_data.pf_df["score_method"] = "ELECTRE-III"
    x_data.pf_df["score_alpha"] = 1

    with open(base_path + "_info.txt", "w", encoding="utf-8") as f:
        f.write("Cenário de simulação\n")
        f.write(f"Início: {horizon_start}\n")
        f.write(f"Horizonte: {horizon} horas\n")
        f.write(f"NORADs usados: {sat_TLEs}\n")
        f.write(f"Tamanho de df: {len(x_data.df)} linhas\n")
        f.write(f"Instância: eoss_instance_{instance_tag}\n")

    save_instance(base_path, x_data)

    if test_solve:
        print("[INFO] Rodando smoke-test DAG (use_existing_score=True)...")
        res = EOS.solve(
            x_data,
            solution_method="DAG",
            use_existing_score=True,
            score_column="score_scenario",
        )
        print("Score usado (primeiros 5):", res.score[:5])
        print("x (qtde selecionados):", int(np.sum(res.x)))
        print("obj:", float(-res.score @ res.x))
        print("time:", res.time)

    print(f"[DONE] Instância pronta: eoss_instance_{instance_tag}")
    return base_path, x_data


if __name__ == "__main__":
    args = parse_args()

    n_requests = args.n_requests if args.n_requests is not None else DEFAULT_N_REQUESTS
    horizon = args.horizon if args.horizon is not None else DEFAULT_HORIZON
    horizon_start = list(args.start) if args.start is not None else list(DEFAULT_HORIZON_START)

    if args.norads is not None:
        sat_TLEs = list(args.norads)
    else:
        n_sats = args.n_sats if args.n_sats is not None else DEFAULT_N_SATS
        if n_sats < 1 or n_sats > len(DEFAULT_FLEET):
            raise SystemExit(
                f"n_sats={n_sats} inválido. Use 1..{len(DEFAULT_FLEET)} "
                f"ou passe --norads."
            )
        sat_TLEs = DEFAULT_FLEET[:n_sats]

    generate_instance(
        n_requests=n_requests,
        horizon=horizon,
        sat_TLEs=sat_TLEs,
        horizon_start=horizon_start,
        instances_dir=args.instances_dir,
        map_generation=args.map,
        test_solve=args.test_solve,
    )
