import numpy as np
import cvxpy as cpy


def stat_check(stat, N):
    if stat not in [cpy.OPTIMAL, cpy.OPTIMAL_INACCURATE]:
        print(f"*******WARNING********\nPROBLEM WAS NOT OPTIMAL")
        print(f"Problem status is: {stat} and # users is: {N}")


def make_gamma_inv_matrix(N, S, R, c_tilde, d_tilde):
    gamma_inv = np.zeros((N, N))
    for user in range(0, N):
        gamma_sum = np.sum(np.min(c_tilde / d_tilde[user:user+1, 0:R], axis=1))
        gamma_inv[user, user] = 1.0 / min(gamma_sum, 1.0 / d_tilde[user, R])
    return gamma_inv


def eq(c, c_tilde, d_tilde, N, S, R):
    x = np.matrix(np.zeros((N, S)))
    for u in range(0,N):
        x[u,:] = np.min(c_tilde/N/d_tilde[u, 0:R], axis=1).T
        x[u,:] = x[u,:] * np.min(1./N/d_tilde[u, R], np.sum(x[u,:]))/np.sum(x[u,:])
    return x


def cru(c, c_tilde, d_tilde, N, S, R):
    gamma_inv = make_gamma_inv_matrix(N, S, R, c_tilde, d_tilde)
    ones_S_by_1 = np.ones(shape=(S, 1))
    ones_1_by_N = np.ones(shape=(1, N))
    id_func = lambda i: np.array([[1 if k == i else 0 for k in range(N)]])
    x = cpy.Variable(shape=(N, S), nonneg=True)
    constr = []
    constr += [x.T @ d_tilde[:,0:R] <= c_tilde]
    constr += [ones_S_by_1.T @ x.T @ d_tilde[:,R:(R+1)] <= 1]
    constr += [gamma_inv @ x @ ones_S_by_1 >= 1.0 / N * ones_1_by_N.T]
    for i, j in zip(range(N), range(N)):
        if i == j:
            continue
        constr += [id_func(i) @ x @ ones_S_by_1 >= np.min(d_tilde[j, :]/d_tilde[i, :]) * id_func(j) @ x @ ones_S_by_1]
    obj = cpy.Maximize(np.ones(shape=(1, N)) @ gamma_inv @ x @ ones_S_by_1)
    prob = cpy.Problem(obj, constr)
    try:
        prob.solve(solver=cpy.MOSEK, verbose=False)
    except cpy.error.SolverError:
        prob.solve(solver=cpy.SCS, verbose=True)
    stat_check(prob.status, N)
    return x.value


def cu(c, c_tilde, d_tilde, N, S, R):
    gamma_inv = make_gamma_inv_matrix(N, S, R, c_tilde, d_tilde)
    ones_S_by_1 = np.ones(shape=(S, 1))
    ones_1_by_N = np.ones(shape=(1, N))
    id_func = lambda i: np.array([[1 if k == i else 0 for k in range(N)]])
    x = cpy.Variable(shape=(N, S), nonneg=True)
    constr = []
    constr += [x.T @ d_tilde[:,0:R] <= c_tilde]
    constr += [ones_S_by_1.T @ x.T @ d_tilde[:,R:(R+1)] <= 1]
    constr += [gamma_inv @ x @ ones_S_by_1 >= 1.0 / N * ones_1_by_N.T]
    for i, j in zip(range(N), range(N)):
        if i == j:
            continue
        constr += [id_func(i) @ x @ ones_S_by_1 >= np.min(d_tilde[j, :]/d_tilde[i, :]) * id_func(j) @ x @ ones_S_by_1]
    obj = cpy.Maximize(ones_1_by_N @ x @ ones_S_by_1)
    prob = cpy.Problem(obj, constr)
    try:
        prob.solve(solver=cpy.MOSEK, verbose=False)
    except cpy.error.SolverError:
        prob.solve(solver=cpy.SCS, verbose=True)
    stat_check(prob.status, N)
    return x.value


def ru(c, c_tilde, d_tilde, N, S, R):
    gamma_inv = make_gamma_inv_matrix(N, S, R, c_tilde, d_tilde)
    ones_S_by_1 = np.ones(shape=(S,1))
    x = cpy.Variable(shape=(N, S), nonneg=True)
    constr = []
    constr += [x.T @ d_tilde[:,0:R] <= c_tilde]
    constr += [ones_S_by_1.T @ x.T @ d_tilde[:,R:(R+1)] <= 1]
    obj = cpy.Maximize(np.ones(shape=(1, N)) @ gamma_inv @ x @ ones_S_by_1)
    prob = cpy.Problem(obj, constr)
    try:
        prob.solve(solver=cpy.MOSEK, verbose=False)
    except cpy.error.SolverError:
        prob.solve(solver=cpy.SCS, verbose=True)
    stat_check(prob.status, N)
    return x.value


def u(c, c_tilde, d_tilde, N, S, R):
    ones_S_by_1 = np.ones(shape=(S, 1))
    x = cpy.Variable(shape=(N, S), nonneg=True)
    constr = []
    constr += [x.T @ d_tilde[:, 0:R] <= c_tilde]
    constr += [ones_S_by_1.T @ x.T @ d_tilde[:,R:(R+1)] <= 1]
    obj = cpy.Maximize(np.ones(shape=(1, N)) @ x @ ones_S_by_1)
    prob = cpy.Problem(obj, constr)
    try:
        prob.solve(solver=cpy.MOSEK, verbose=False)
    except cpy.error.SolverError:
        prob.solve(solver=cpy.SCS, verbose=True)
    stat_check(prob.status, N)
    return x.value


def mnw(c, c_tilde, d_tilde, N, S, R):
    ones_S_by_1 = np.ones((S,1))
    x = cpy.Variable(shape=(N, S), nonneg=True)
    constr = []
    constr += [x.T @ d_tilde[:,0:R] <= c_tilde]
    constr += [ones_S_by_1.T @ x.T @ d_tilde[:,R:(R+1)] <= 1]
    obj = cpy.Maximize(cpy.sum(cpy.log(x @ ones_S_by_1)))
    prob = cpy.Problem(obj, constr)
    try:
        prob.solve(solver=cpy.MOSEK, verbose=False)
    except cpy.error.SolverError:
        prob.solve(solver=cpy.SCS, verbose=True)
    stat_check(prob.status, N)
    return x.value


def drfer(c, c_tilde, d_tilde, N, S, R):
    ones_N_by_1 = np.ones((N,1))
    ones_S_by_1 = np.ones((S,1))
    dom_mat = np.diagflat(np.squeeze(np.asarray(d_tilde.max(1))))
    g = cpy.Variable(nonneg=True)
    x = cpy.Variable(shape=(N, S), nonneg=True)
    constr = []
    constr += [x.T @ d_tilde[:, 0:R] <= c_tilde]
    constr += [ones_S_by_1.T @ x.T @ d_tilde[:,R:(R+1)] <= 1]
    constr += [dom_mat @ x @ ones_S_by_1 == g * ones_N_by_1]
    obj = cpy.Maximize(g)
    prob = cpy.Problem(obj, constr)
    try:
        prob.solve(solver=cpy.MOSEK, verbose=False)
    except cpy.error.SolverError:
        prob.solve(solver=cpy.SCS, verbose=True)
    stat_check(prob.status, N)
    return x.value


def tsfer(c, c_tilde, d_tilde, N, S, R):
    gamma_inv = make_gamma_inv_matrix(N, S, R, c_tilde, d_tilde)
    ones_N_by_1 = np.ones((N,1))
    ones_S_by_1 = np.ones((S,1))
    g = cpy.Variable(nonneg=True)
    x = cpy.Variable(shape=(N, S), nonneg=True)
    constr = []
    constr += [x.T @ d_tilde[:, 0:R] <= c_tilde]
    constr += [ones_S_by_1.T @ x.T @ d_tilde[:,R:(R+1)] <= 1]
    constr += [gamma_inv @ x @ ones_S_by_1 == g * ones_N_by_1]
    obj = cpy.Maximize(g)
    prob = cpy.Problem(obj, constr)
    try:
        prob.solve(solver=cpy.MOSEK, verbose=False)
    except cpy.error.SolverError:
        prob.solve(solver=cpy.SCS, verbose=True)
        
    stat_check(prob.status, N)
    return x.value
