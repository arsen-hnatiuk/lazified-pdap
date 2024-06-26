import numpy as np
import scipy as sp
import pandas as pd
import logging
from sissopp import FeatureSpace
from sissopp.py_interface import import_dataframe
from lib.default_values import *
from lib.ssn import SSN

logging.basicConfig(
    level=logging.DEBUG,
)


class LGCG_stochastic:
    # An implementation of the stochastic sampling approach to finite LGCG

    def __init__(
        self,
        alpha: float,
        data_path: str,
        test_power: float = 0.05,
        parameter_increase: float = 0.1,
    ) -> None:
        target = pd.read_csv(data_path)["log kappa_L"].to_numpy()
        self.target_norm = np.linalg.norm(target)
        self.target = target / self.target_norm
        self.alpha = alpha
        self.test_power = test_power
        self.parameter_increase = parameter_increase
        self.feature_inputs = import_dataframe.create_inputs(
            df=data_path,
            max_rung=3,
            max_param_depth=2,
            prop_key="log kappa_L",
            calc_type="regression",
            n_sis_select=10,
            allowed_ops=[
                "add",
                "sub",
                "abs_diff",
                "mult",
                "div",
                "inv",
                "abs",
                "exp",
                "log",
                "sin",
                "cos",
                "sq",
                "cb",
                "six_pow",
                "sqrt",
                "cbrt",
                "neg_exp",
            ],
            n_rung_generate=0,
            n_rung_store=-1,
        )
        self.g = get_default_g(self.alpha)
        self.L = 1
        self.j = lambda K, u: 0.5 * np.linalg.norm(
            np.matmul(K, u) - self.target
        ) ** 2 + self.g(u)
        self.rho = lambda K, u: np.matmul(K, u) - self.target
        self.M = (
            self.j(np.zeros((len(self.target), 1)), np.zeros(1)) / self.alpha
        )  # Bound on the norm of iterates
        self.C = 4 * self.L * self.M**2
        self.machine_precision = 1e-11

    def update_epsilon(self, eta: float, epsilon: float) -> float:
        return (self.M * epsilon + 0.5 * self.C * eta**2) / (self.M + self.M * eta)

    def explicit_Phi(
        self,
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        observation_u: np.ndarray,
        observation_v: np.ndarray,
    ) -> float:
        # <rho,K(v-u)>+g(u)-g(v)
        return np.dot(rho, observation_v - observation_u) + self.g(u) - self.g(v)

    def Phi(
        self,
        rho: np.ndarray,
        u: np.ndarray,
        observation_u: np.ndarray,
        observation_v: np.ndarray,
    ) -> float:
        # M*max{0,||p_u(x)||-alpha}+g(u)-<rho,Ku>
        return (
            self.M * (max(0, np.abs(np.dot(rho, observation_v)) - self.alpha))
            + self.g(u)
            - np.dot(rho, observation_u)
        )

    def rejection_probability(self, beta: float, beta_plus: float) -> float:
        sin_square = np.sin(np.arccos(beta)) ** 2
        sin_square_plus = np.sin(np.arccos(beta_plus)) ** 2
        a = 0.5 * (len(self.target) - 1)
        return (
            sp.special.betainc(a, 0.5, sin_square)
            - sp.special.betainc(a, 0.5, sin_square_plus)
        ) / (1 - sp.special.betainc(a, 0.5, sin_square_plus))

    def get_sample_size(self, rejection_probability: float) -> int:
        size = 1
        iterate = 1 - rejection_probability
        while iterate > self.test_power:
            size += 1
            iterate *= 1 - rejection_probability
        return size

    def choose_x(
        self, rho: np.ndarray, observation: np.ndarray, u: np.ndarray, epsilon: float
    ) -> tuple:
        feature_space = FeatureSpace(self.feature_inputs)
        sample_features_raw = np.array([feature.value for feature in feature_space.phi])
        sample_norms = np.linalg.norm(sample_features_raw, axis=1)
        sample_features = np.array(
            [feature / norm for feature, norm in zip(sample_features_raw, sample_norms)]
        )
        total_sample = len(sample_features)
        sample_values = np.abs(np.matmul(sample_features, rho))
        incumbent = np.argmax(sample_values)
        incumbent_dict = {
            "feature": sample_features[incumbent],
            "feature_norm": sample_norms[incumbent],
            "expression": feature_space.phi[incumbent].expr,
        }
        Phi_value = self.explicit_Phi(
            rho=rho,
            u=u,
            v=np.array([self.M]),
            observation_u=observation,
            observation_v=self.M
            * np.sign(np.dot(rho, incumbent_dict["feature"]))
            * incumbent_dict["feature"],
        )
        incumbent_dict["Phi_value"] = Phi_value
        if Phi_value >= self.M * epsilon:
            return incumbent_dict, True
        beta = sample_values[incumbent] / np.linalg.norm(rho)
        beta_plus = min(beta / (1 - self.parameter_increase), 1)
        accept = False
        while beta_plus < 1 and accept == False:
            probability = self.rejection_probability(beta, beta_plus)
            sample_size = self.get_sample_size(rejection_probability=probability)
            while total_sample < sample_size:
                feature_space = FeatureSpace(self.feature_inputs)
                sample_features_raw = np.array(
                    [feature.eval for feature in feature_space.phi]
                )
                sample_norms = np.linalg.norm(sample_features_raw, axis=1)
                sample_features = np.array(
                    [
                        feature / norm
                        for feature, norm in zip(sample_features_raw, sample_norms)
                    ]
                )
                total_sample += len(sample_features)
                sample_values = np.abs(np.matmul(sample_features, rho))
                possible_incumbent = np.argmax(sample_values)
                if sample_values[possible_incumbent] / np.linalg.norm(rho) > beta:
                    incumbent = possible_incumbent
                    incumbent_dict = {
                        "feature": sample_features[incumbent],
                        "feature_norm": sample_norms[incumbent],
                        "expression": feature_space.phi[incumbent].expr,
                    }
                    Phi_value = self.explicit_Phi(
                        rho=rho,
                        u=u,
                        v=np.array([self.M]),
                        observation_u=observation,
                        observation_v=self.M
                        * np.sign(np.dot(rho, incumbent_dict["feature"]))
                        * incumbent_dict["feature"],
                    )
                    incumbent_dict["Phi_value"] = Phi_value
                    if Phi_value >= self.M * epsilon:
                        return incumbent_dict, True
                    beta = sample_values[incumbent] / np.linalg.norm(rho)
                    beta_plus = min(beta / (1 - self.parameter_increase), 1)
                    continue
            accept = True
        return incumbent_dict, False

    def solve(self, tol: float) -> dict:
        u = np.array([])
        active_K_T = np.array([])
        active_norms = np.array([])
        active_expressions = np.array([])
        observation = np.zeros(len(self.target))
        rho = self.rho(np.zeros((len(self.target), 1)), np.zeros(1))
        epsilon = self.j(np.zeros((len(self.target), 1)), np.zeros(1)) / self.M
        k = 1
        eta = 4 / (k + 3)
        epsilon = self.update_epsilon(eta, epsilon)
        x_dict, x_success = self.choose_x(rho, observation, u, epsilon)
        Psi = epsilon
        Phi_value = self.Phi(
            rho=rho,
            u=u,
            observation_u=observation,
            observation_v=x_dict["feature"],
        )
        while Phi_value > tol:
            u_old = u.copy()
            Psi_old = Psi
            if x_dict["expression"] in active_expressions:
                Psi = Psi / 2
            Psi = max(min(Psi, self.M * epsilon), self.machine_precision)
            if x_success:
                if x_dict["expression"] in active_expressions:
                    index = np.where(active_expressions == x_dict["expressions"])[0][0]
                    v = (
                        self.M
                        * np.sign(np.dot(rho, x_dict["feature"]))
                        * np.eye(1, len(u), index)[0]
                    )
                else:
                    if len(active_K_T):
                        active_K_T = np.vstack((active_K_T, x_dict["feature"]))
                    else:
                        active_K_T = np.array([x_dict["feature"]])
                    active_norms = np.append(active_norms, x_dict["feature_norm"])
                    active_expressions = np.append(
                        active_expressions, x_dict["expression"]
                    )
                    u = np.append(u, np.array([0]))
                    v = (
                        self.M
                        * np.sign(np.dot(rho, x_dict["feature"]))
                        * np.eye(1, len(u), len(u) - 1)[0]
                    )
                u = (1 - eta) * u + eta * v
            elif (
                self.explicit_Phi(
                    rho=rho,
                    u=u,
                    v=np.array([0]),
                    observation_u=observation,
                    observation_v=np.zeros(len(self.target)),
                )
                >= self.M * epsilon
            ):
                u = (1 - eta) * u
            elif x_dict["Phi_value"] > 0:
                if x_dict["expression"] in active_expressions:
                    index = np.where(active_expressions == x_dict["expressions"])[0][0]
                    v = (
                        self.M
                        * np.sign(np.dot(rho, x_dict["feature"]))
                        * np.eye(1, len(u), index)[0]
                    )
                else:
                    if len(active_K_T):
                        active_K_T = np.vstack((active_K_T, x_dict["feature"]))
                    else:
                        active_K_T = np.array([x_dict["feature"]])
                    active_norms = np.append(active_norms, x_dict["feature_norm"])
                    active_expressions = np.append(
                        active_expressions, x_dict["expression"]
                    )
                    u = np.append(u, np.array([0]))
                    v = (
                        self.M
                        * np.sign(np.dot(rho, x_dict["feature"]))
                        * np.eye(1, len(u), len(u) - 1)[0]
                    )
                eta_local = x_dict["Phi_value"] / self.C
                u = (1 - eta_local) * u + eta_local * v

            k += 1
            eta = 4 / (k + 3)
            epsilon = self.update_epsilon(eta, epsilon)

            if len(u) != len(u_old) or not np.array_equal(u, u_old) or Psi_old != Psi:
                # Low-dimensional optimization
                ssn = SSN(
                    K=active_K_T.T, alpha=self.alpha, target=self.target, M=self.M
                )
                u_raw = ssn.solve(tol=Psi, u_0=u)

                if len(u_raw) != len(u_old) or not np.array_equal(u_raw, u_old):
                    # SSN found a different solution
                    u_to_keep = np.where(np.abs(u_raw) >= self.machine_precision)[0]
                    u = u_raw[u_to_keep]
                    active_K_T = active_K_T[u_to_keep]
                    active_norms = active_norms[u_to_keep]
                    active_expressions = active_expressions[u_to_keep]
                    observation = np.matmul(active_K_T.T, u)
                    rho = self.rho(active_K_T.T, u)
                    x_dict, x_success = self.choose_x(rho, observation, u, epsilon)
                    Phi_value = self.Phi(
                        rho=rho,
                        u=u,
                        observation_u=observation,
                        observation_v=x_dict["feature"],
                    )

            logging.info(
                f"{k}: Phi {Phi_value:.3E}, epsilon {epsilon:.3E}, support {len(u)}, Psi {Psi:.3E}"
            )
        logging.info(
            f"LGCG converged in {k} iterations to tolerance {tol:.3E} with final sparsity of {len(u)}"
        )
        # Rescale the solution
        for ind, pos in enumerate(active_norms):
            u[ind] /= pos
        u = u * self.target_norm

        return {"u": u, "support": len(u), "expressions": active_expressions}
