import numpy as np
import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.connection import LeviCivitaConnection
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from typing import Callable, Optional, Tuple

class SocietalManifold(RiemannianMetric):
    def __init__(self, dim: int, metric_params: dict):
        super(SocietalManifold, self).__init__(dim=dim)
        self.connection = LeviCivitaConnection(self)
        self.metric_params = metric_params

    def metric_matrix(self, base_point: Optional[gs.array] = None) -> gs.array:
        if base_point is None:
            base_point = gs.zeros(self.dim)
        
        distance = gs.linalg.norm(base_point)
        decay = gs.exp(-distance / self.metric_params['scale'])
        identity = gs.eye(self.dim)
        outer_product = gs.outer(base_point, base_point)
        
        metric = (decay * identity + 
                  (1 - decay) * outer_product / (distance**2 + self.metric_params['epsilon']))
        
        return self.metric_params['amplitude'] * metric + self.metric_params['base'] * identity

    def geodesic(self, initial_point: gs.array, end_point: Optional[gs.array] = None, 
                 initial_tangent_vec: Optional[gs.array] = None) -> gs.array:
        def geodesic_equation(t: float, state: gs.array) -> gs.array:
            position, velocity = gs.split(state, 2)
            christoffel = self.connection.christoffels(position)
            acceleration = -gs.einsum('ijk,j,k->i', christoffel, velocity, velocity)
            return gs.concatenate([velocity, acceleration])

        if end_point is not None:
            initial_tangent_vec = self.log(end_point, initial_point)

        initial_state = gs.concatenate([initial_point, initial_tangent_vec])
        solution = solve_ivp(geodesic_equation, (0., 1.), initial_state, t_eval=gs.linspace(0., 1., 100))
        
        return solution.y[:self.dim].T

    def exp(self, tangent_vec: gs.array, base_point: gs.array) -> gs.array:
        def exp_equation(t: float, state: gs.array) -> gs.array:
            position = state[:self.dim]
            velocity = tangent_vec
            christoffel = self.connection.christoffels(position)
            acceleration = -gs.einsum('ijk,j,k->i', christoffel, velocity, velocity)
            return gs.concatenate([velocity, acceleration])

        initial_state = gs.concatenate([base_point, tangent_vec])
        solution = solve_ivp(exp_equation, (0., 1.), initial_state)
        
        return solution.y[:self.dim, -1]

    def log(self, point: gs.array, base_point: gs.array) -> gs.array:
        def objective(tangent_vec: gs.array) -> float:
            exp_point = self.exp(tangent_vec, base_point)
            return gs.linalg.norm(exp_point - point)

        initial_guess = point - base_point
        result = minimize(objective, initial_guess, method='BFGS')
        
        return result.x

    def curvature(self, point: gs.array) -> gs.array:
        christoffel = self.connection.christoffels(point)
        dim = self.dim
        R = gs.zeros((dim, dim, dim, dim))

        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        term1 = gs.sum(christoffel[i, k, m] * christoffel[m, j, l] for m in range(dim))
                        term2 = gs.sum(christoffel[i, l, m] * christoffel[m, j, k] for m in range(dim))
                        R[i, j, k, l] = term1 - term2

        return R

    def ricci_curvature(self, point: gs.array) -> gs.array:
        full_curvature = self.curvature(point)
        return gs.einsum('ijki', full_curvature)

    def scalar_curvature(self, point: gs.array) -> float:
        ricci = self.ricci_curvature(point)
        metric_inverse = gs.linalg.inv(self.metric_matrix(point))
        return gs.einsum('ij,ij', metric_inverse, ricci)

class ParallelTransporter:
    def __init__(self, manifold: SocietalManifold):
        self.manifold = manifold

    def parallel_transport(self, vector: gs.array, curve: gs.array) -> gs.array:
        def transport_equation(t: float, state: gs.array) -> gs.array:
            position, vector = gs.split(state, 2)
            velocity = gs.gradient(curve, axis=0)[int(t * (len(curve) - 1))]
            christoffel = self.manifold.connection.christoffels(position)
            dvdt = -gs.einsum('ijk,j,k->i', christoffel, velocity, vector)
            return gs.concatenate([velocity, dvdt])

        initial_state = gs.concatenate([curve[0], vector])
        solution = solve_ivp(transport_equation, (0., 1.), initial_state, t_eval=gs.linspace(0., 1., len(curve)))
        
        return solution.y[self.manifold.dim:, -1]

class GeodesicFlow:
    def __init__(self, manifold: SocietalManifold):
        self.manifold = manifold

    def flow(self, initial_point: gs.array, vector_field: Callable[[gs.array], gs.array], 
             time_span: float, num_points: int = 100) -> gs.array:
        def flow_equation(t: float, state: gs.array) -> gs.array:
            return vector_field(state)

        t_eval = gs.linspace(0, time_span, num_points)
        solution = solve_ivp(flow_equation, (0., time_span), initial_point, t_eval=t_eval)
        
        return solution.y.T

class SocietalTensorField:
    def __init__(self, manifold: SocietalManifold, field_function: Callable[[gs.array], gs.array]):
        self.manifold = manifold
        self.field_function = field_function

    def __call__(self, point: gs.array) -> gs.array:
        return self.field_function(point)

    def covariant_derivative(self, point: gs.array, direction: gs.array) -> gs.array:
        jacobian = gs.jacobian(self.field_function, point)
        christoffel = self.manifold.connection.christoffels(point)
        return (jacobian @ direction + 
                gs.einsum('ijk,j,k->i', christoffel, self.field_function(point), direction))

def compute_lyapunov_spectrum(manifold: SocietalManifold, trajectory: gs.array, 
                              num_exponents: int) -> gs.array:
    def variational_equation(t: float, state: gs.array) -> gs.array:
        position, perturbations = gs.split(state, [manifold.dim])
        perturbations = perturbations.reshape(num_exponents, manifold.dim)
        
        velocity = gs.gradient(trajectory, axis=0)[int(t * (len(trajectory) - 1))]
        christoffel = manifold.connection.christoffels(position)
        
        d_perturbations = gs.zeros_like(perturbations)
        for i in range(num_exponents):
            d_perturbations[i] = (-gs.einsum('ijk,j,k->i', christoffel, velocity, perturbations[i]) + 
                                  gs.einsum('ijk,j,k->i', christoffel, perturbations[i], velocity))
        
        return gs.concatenate([velocity, d_perturbations.flatten()])

    initial_perturbations = gs.eye(manifold.dim)[:num_exponents]
    initial_state = gs.concatenate([trajectory[0], initial_perturbations.flatten()])
    
    solution = solve_ivp(variational_equation, (0., 1.), initial_state, t_eval=gs.linspace(0., 1., len(trajectory)))
    
    final_perturbations = solution.y[manifold.dim:, -1].reshape(num_exponents, manifold.dim)
    
    # Perform QR decomposition to extract Lyapunov exponents
    Q, R = gs.linalg.qr(final_perturbations.T)
    
    return gs.log(gs.abs(gs.diag(R))) / (trajectory[-1, 0] - trajectory[0, 0])
