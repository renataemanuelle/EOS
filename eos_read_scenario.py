import numpy as np
import pandas as pd
import time
from types import SimpleNamespace
from EOSpython import EOS

def load_instance(base_path: str):
    # carrega df/pf_df preservando tipos
    df = pd.read_pickle(base_path + "_df.pkl")
    pf_df = pd.read_pickle(base_path + "_pf_df.pkl")

    with open(base_path + "_sats.txt", "r", encoding="utf-8") as f:
        sats = [int(line.strip()) for line in f if line.strip()]

    # carrega LPP state
    st = np.load(base_path + "_lpp_state.npz", allow_pickle=True)
    perf_df = pd.read_pickle(base_path + "_lpp_performance_df.pkl")

    # reconstrói LHS (cvxopt.spmatrix)
    from cvxopt import spmatrix
    I = st["LHS_I"]; J = st["LHS_J"]; V = st["LHS_V"]
    shape = tuple(st["LHS_shape"].tolist())
    LHS = spmatrix(V, I, J, size=shape)

    # monta LPP como objeto simples
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

    # monta x_data
    x_data = SimpleNamespace()
    x_data.df = df
    x_data.pf_df = pf_df
    x_data.sats = sats
    x_data.LPP = LPP_obj

    return x_data

base_path = "eoss_instance_20251121"
x_data = load_instance(base_path)

t_total_start = time.time()
res = EOS.solve(
    x_data,
    solution_method="DAG",
    use_existing_score=True,
    score_column="score_scenario",
)
t_total_no_io = time.time() - t_total_start

print("x (qtde selecionados):", int(np.sum(res.x)))
print("obj:", float(-res.score @ res.x))
print("T_solver:", res.time)
print("T_total_no_io:", t_total_no_io)

eval_res = EOS.evaluate(x_data, res)
print("obj avaliado (scenario):", eval_res.scenario)
print("obj avaliado (solution):", eval_res.solution)


t_total_start2 = time.time()
res.calc = EOS.solve(
    x_data,
    solution_method="DAG",
    use_existing_score=False,
)
t_total_no_io2 = time.time() - t_total_start2

print("x (qtde selecionados):", int(np.sum(res.calc.x)))
print("obj:", float(-res.calc.score @ res.calc.x))
print("T_solver:", res.calc.time)
print("T_total_no_io:", t_total_no_io2)

eval_res = EOS.evaluate(x_data, res)
print("obj avaliado (scenario):", eval_res.scenario)
print("obj avaliado (solution):", eval_res.solution)
