import numpy as np
from scipy import linalg
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

class GPWarper:
    def __init__(self, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', 
                 n_restarts_optimizer=5, normalize_y=True):
        """
        Gaussian Process Wrapper for Policy Transportation.
        """
        # Default Kernel: Constant * RBF + WhiteKernel (Noise)
        if kernel is None:
            # Length scale 0.1 is a good starting point for manipulation
            kernel = C(1.0) * RBF(length_scale=0.1) + WhiteKernel(noise_level=1e-5)
            
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y
        )
        
        # Internal storage for analytical derivative calculation
        self.X_train = None
        self.weights = None
        self.length_scale_val = None

    def fit(self, source_points, target_points):
        """
        Learns the residual field: f(source) = target - source
        """
        residuals = target_points - source_points
        self.gp.fit(source_points, residuals)
        
        # --- PRE-CALCULATE DATA FOR ANALYTICAL JACOBIAN ---
        self.X_train = source_points
        opt_kernel = self.gp.kernel_
        
        # 1. robustly extract length_scale from the optimized kernel
        # Sklearn nests kernels. We look for the RBF component.
        params = opt_kernel.get_params()
        
        # Try finding length_scale in common nesting patterns
        keys_to_check = ['k1__k2__length_scale', 'k1__length_scale', 'length_scale']
        self.length_scale_val = 0.2 # Fallback
        for k in keys_to_check:
            if k in params:
                self.length_scale_val = params[k]
                break
        print(f"[GPWarper] Optimized length_scale: {self.length_scale_val}")
        # 2. Compute weights: K_inv * Y
        # We re-compute K to ensure we have the exact matrix used for prediction
        K = opt_kernel(self.X_train)
        # Add alpha to diagonal (same as sklearn does internally)
        K[np.diag_indices_from(K)] += self.gp.alpha
        
        self.weights = linalg.solve(K, residuals, assume_a='pos')

    def predict(self, points):
        """
        Returns warped point: x' = x + GP(x)
        """
        points = np.atleast_2d(points)
        delta = self.gp.predict(points)
        return (points + delta).squeeze()

    def get_jacobian(self, x):
        """
        Computes J = I + d(Residual)/dx analytically.
        Assumes RBF kernel structure.
        """
        x = np.array(x).reshape(1, -1)
        
        # 1. Difference vectors (N_train, 1, 3)
        diff = self.X_train[:, np.newaxis, :] - x[np.newaxis, :, :]
        
        # 2. Kernel values (RBF part only)
        # We compute exp(-||diff||^2 / 2l^2) * Constant
        # We can use the optimized kernel function, but must be careful to exclude WhiteKernel effects
        # For simplicity/speed in this specific RBF case:
        dist_sq = np.sum(diff**2, axis=2).flatten()
        l2 = self.length_scale_val**2
        
        # Get constant value (C) from kernel if present, else 1.0
        c_val = self.gp.kernel_.get_params().get('k1__k1__constant_value', 1.0)
        
        k_vals = c_val * np.exp(-0.5 * dist_sq / l2) # (N_train,)
        
        # 3. Gradients of the kernel k(x_train, x) w.r.t x
        # d/dx(k) = k * (x_train - x) / l^2
        # Note: diff defined above is (X_train - x)
        coeff = k_vals[:, np.newaxis] / l2 # (N_train, 1)
        dk_dx = coeff[:, :, np.newaxis] * diff # (N_train, 1, 3)
        
        # 4. Jacobian of residuals: weights.T * dk_dx
        # weights: (N, 3), dk_dx: (N, 1, 3) -> J_res: (3, 3)
        J_res = np.einsum("no,nmd->mod", self.weights, dk_dx)[0]
        
        # 5. Total Jacobian = I + J_residual
        return np.eye(3) + J_res