import numpy as np
import pandas as pd
import time
import os
import argparse
from types import SimpleNamespace
from EOSpython import EOS


def extract_instance_tag(base_path: str) -> str:
    """Extrai o tag da instância a partir do base_path.
    Ex: 'Instances/eoss_instance_1000_24h'      -> '1000_24h'
        'Instances/eoss_instance_250_8h_2sat'   -> '250_8h_2sat'
        'eoss_instance_20251121'                -> '20251121'
    """
    name = os.path.basename(base_path)
    prefix = "eoss_instance_"
    idx = name.find(prefix)
    if idx != -1:
        tag = name[idx + len(prefix):]
        suffixes = ("_info", "_pf_df", "_df", "_lpp_state",
                    "_lpp_performance_df", "_sats")
        for suf in suffixes:
            suf_pos = tag.find(suf)
            if suf_pos != -1:
                return tag[:suf_pos]
        dot = tag.rfind(".")
        if dot != -1:
            return tag[:dot]
        return tag
    dot = name.rfind(".")
    return name[:dot] if dot != -1 else name


def build_instance_base_path(n_requests: int,
                             horizon: int,
                             n_sats: int | None = None,
                             instances_dir: str = "Instances") -> str:
    """Resolve o base_path a partir de N, H e (opcional) nº de satélites.

    Ordem de busca:
      1) Instances/eoss_instance_{N}_{H}h_{K}sat   (formato novo)
      2) Instances/eoss_instance_{N}_{H}h           (legado SBPO, 1 sat)
      3) mesmos nomes no cwd (compatibilidade)
    """
    names = []
    if n_sats is not None:
        names.append(f"eoss_instance_{n_requests}_{horizon}h_{n_sats}sat")
    names.append(f"eoss_instance_{n_requests}_{horizon}h")

    candidates = []
    for name in names:
        candidates.append(os.path.join(instances_dir, name))
        candidates.append(name)

    for base in candidates:
        if os.path.isfile(base + "_pf_df.pkl") and os.path.isfile(base + "_lpp_state.npz"):
            return base

    tried = "\n  - ".join(c + "_*" for c in candidates)
    raise FileNotFoundError(
        f"Instância não encontrada para N={n_requests}, H={horizon}h"
        + (f", sats={n_sats}" if n_sats is not None else "")
        + f".\nTentou:\n  - {tried}"
    )


def load_instance(base_path: str):
    df = pd.read_pickle(base_path + "_df.pkl")
    pf_df = pd.read_pickle(base_path + "_pf_df.pkl")

    with open(base_path + "_sats.txt", "r", encoding="utf-8") as f:
        sats = [int(line.strip()) for line in f if line.strip()]

    st = np.load(base_path + "_lpp_state.npz", allow_pickle=True)
    perf_df = pd.read_pickle(base_path + "_lpp_performance_df.pkl")

    from cvxopt import spmatrix
    I = st["LHS_I"]; J = st["LHS_J"]; V = st["LHS_V"]
    shape = tuple(st["LHS_shape"].tolist())
    LHS = spmatrix(V, I, J, size=shape)

    LPP_obj = SimpleNamespace()
    LPP_obj.B = st["B"]
    LPP_obj.F = st["F"]
    LPP_obj.LHS = LHS
    LPP_obj.RHS = st["RHS"]
    LPP_obj.eLHS = st["eLHS"]
    LPP_obj.eRHS = st["eRHS"]
    LPP_obj.stereo = st["stereo"]
    LPP_obj.strips = st["strips"]
    LPP_obj.performance_df = perf_df

    x_data = SimpleNamespace()
    x_data.df = df
    x_data.pf_df = pf_df
    x_data.sats = sats
    x_data.LPP = LPP_obj

    return x_data


def export_evaluation_csv(x_data, res, t_solver, t_total, solution_method,
                          output_path="Results/Evaluation_EOS.csv"):
    """Exporta métricas no formato section,metric,value (compatível com Evaluation_RKO.csv)."""
    pf = x_data.pf_df
    sel = np.where(res.x)[0]
    sel_df = pf.iloc[sel]

    score = res.score
    total_score = float(score @ res.x)
    ofv = float(-score @ res.x)

    rows = []

    # --- scenario (sobre pf_df completo, equivalente a scenario_raw do RKO) ---
    rows.append(("scenario", "requests",           int(pf["ID"].nunique())))
    rows.append(("scenario", "attempts",            len(pf)))
    rows.append(("scenario", "n_sats",              len(x_data.sats)))
    rows.append(("scenario", "constraints",
                 int(x_data.LPP.eRHS.shape[0] + x_data.LPP.RHS.shape[0])))
    rows.append(("scenario", "avg_angle",           f"{pf['angle'].mean():.6f}"))
    rows.append(("scenario", "avg_area",            f"{pf['area'].mean():.6f}"))
    rows.append(("scenario", "avg_price",           f"{pf['price'].mean():.6f}"))
    rows.append(("scenario", "avg_sun_elevation",   f"{pf['sun elevation'].mean():.6f}"))
    rows.append(("scenario", "avg_cloud_cover",     f"{pf['cloud cover real'].mean():.6f}"))
    rows.append(("scenario", "avg_priority",        f"{pf['priority'].mean():.6f}"))

    # --- solution (sobre aquisições selecionadas) ---
    rows.append(("solution", "acquisitions",            int(np.sum(res.x))))
    rows.append(("solution", "unique_requests_served",  int(sel_df["ID"].nunique())))
    rows.append(("solution", "total_score",             f"{total_score:.16f}"))
    rows.append(("solution", "ofv",                     f"{ofv:.16f}"))
    rows.append(("solution", "total_profit",            f"{sel_df['price'].sum():.6f}"))
    rows.append(("solution", "total_area",              f"{sel_df['area'].sum():.6f}"))
    rows.append(("solution", "avg_cloud_cover",         f"{sel_df['cloud cover real'].mean():.6f}"))
    rows.append(("solution", "cloud_good_lt10",
                 int((sel_df["cloud cover real"] < 10).sum())))
    rows.append(("solution", "cloud_bad_gt30",
                 int((sel_df["cloud cover real"] > 30).sum())))
    rows.append(("solution", "avg_angle",               f"{sel_df['angle'].mean():.6f}"))
    rows.append(("solution", "angle_good_le10",
                 int((sel_df["angle"] <= 10).sum())))
    rows.append(("solution", "angle_bad_ge30",
                 int((sel_df["angle"] >= 30).sum())))
    rows.append(("solution", "avg_priority",            f"{sel_df['priority'].mean():.6f}"))
    rows.append(("solution", "priority_1",  int((sel_df["priority"] == 1).sum())))
    rows.append(("solution", "priority_2",  int((sel_df["priority"] == 2).sum())))
    rows.append(("solution", "priority_3",  int((sel_df["priority"] == 3).sum())))
    rows.append(("solution", "priority_4",  int((sel_df["priority"] == 4).sum())))
    rows.append(("solution", "avg_sun_elevation",
                 f"{sel_df['sun elevation'].mean():.6f}"))

    # --- eos (métricas específicas do framework) ---
    rows.append(("eos", "solution_method",  solution_method))
    rows.append(("eos", "solver_time",      f"{t_solver:.3f}"))
    rows.append(("eos", "total_time",       f"{t_total:.3f}"))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("section,metric,value\n")
        for section, metric, value in rows:
            f.write(f"{section},{metric},{value}\n")

    print(f"[Evaluation] Exported to {output_path} ({len(rows)} metrics)")


def export_solution_vector_csv(x_data, res,
                               output_path="Results/Solution_Vector_EOS.csv"):
    """Exporta vetor de solução por aquisição (compatível com Solution_Vector_RKO.csv)."""
    pf = x_data.pf_df
    n = len(pf)

    score_col = "score_scenario" if "score_scenario" in pf.columns else None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("index,ID,satellite,time,score_scenario,selected\n")
        for i in range(n):
            row = pf.iloc[i]
            score_val = float(row[score_col]) if score_col else float(res.score[i])
            selected = int(res.x[i])
            f.write(f"{i},{row['ID']},{row['satellite']},{row['time']},"
                    f"{score_val:.16f},{selected}\n")

    print(f"[SolutionVector] Exported to {output_path} ({n} rows)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Carrega instância EOS, resolve com DAG e exporta Evaluation/Solution_Vector."
    )
    p.add_argument("n_requests", type=int, nargs="?", default=None,
                   help="Nº de requisições (ex.: 250)")
    p.add_argument("horizon", type=int, nargs="?", default=None,
                   help="Horizonte em horas (ex.: 8)")
    p.add_argument("n_sats", type=int, nargs="?", default=None,
                   help="Nº de satélites (ex.: 2). Se omitido, tenta formato legado.")
    p.add_argument("--base", type=str, default=None,
                   help="base_path completo (ex.: Instances/eoss_instance_250_8h_2sat). "
                        "Sobrescreve N/H/sats.")
    p.add_argument("--instances-dir", type=str, default="Instances",
                   help="Pasta das instâncias (default: Instances)")
    return p.parse_args()


# ============================================================================
#  Execução principal
# ============================================================================
# Defaults alinhados ao gen (edite aqui ou passe por CLI):
#   python eos_read_scenario.py
#   python eos_read_scenario.py 250 8 2
#   python eos_read_scenario.py --base Instances/eoss_instance_250_8h_2sat
DEFAULT_N_REQUESTS = 250
DEFAULT_HORIZON = 8
DEFAULT_N_SATS = 2

if __name__ == "__main__":
    args = parse_args()

    if args.base:
        base_path = args.base
        # aceita path com sufixo acidental
        for suf in ("_info.txt", "_pf_df.pkl", "_df.pkl", "_sats.txt"):
            if base_path.endswith(suf):
                base_path = base_path[: -len(suf)]
                break
    else:
        n_requests = args.n_requests if args.n_requests is not None else DEFAULT_N_REQUESTS
        horizon = args.horizon if args.horizon is not None else DEFAULT_HORIZON
        n_sats = args.n_sats if args.n_sats is not None else DEFAULT_N_SATS
        base_path = build_instance_base_path(
            n_requests, horizon, n_sats, instances_dir=args.instances_dir
        )

    tag = extract_instance_tag(base_path)
    print(f"[INFO] base_path = {base_path}")
    print(f"[INFO] tag       = {tag}")

    x_data = load_instance(base_path)
    print(f"[INFO] sats={x_data.sats} (|S|={len(x_data.sats)}), "
          f"|A|={len(x_data.pf_df)}, requests={x_data.pf_df['ID'].nunique()}")

    # --- Solve: score_scenario (ELECTRE-III pré-calculado) ---
    t_total_start = time.time()
    res = EOS.solve(
        x_data,
        solution_method="DAG",
        use_existing_score=True,
        score_column="score_scenario",
    )
    t_total_no_io = time.time() - t_total_start

    print("=" * 60)
    print("  SOLVE — score_scenario (ELECTRE-III)")
    print("=" * 60)
    print("x (qtde selecionados):", int(np.sum(res.x)))
    print("obj:", float(-res.score @ res.x))
    print("T_solver:", res.time)
    print("T_total_no_io:", t_total_no_io)

    eval_res = EOS.evaluate(x_data, res)
    print("obj avaliado (scenario):")
    print(eval_res.scenario.to_string(index=False))
    print("obj avaliado (solution):")
    print(eval_res.solution.to_string(index=False))

    export_evaluation_csv(x_data, res, t_solver=res.time, t_total=t_total_no_io,
                          solution_method="DAG",
                          output_path=f"Results/Evaluation_EOS_{tag}.csv")
    export_solution_vector_csv(x_data, res,
                               output_path=f"Results/Solution_Vector_EOS_{tag}.csv")
