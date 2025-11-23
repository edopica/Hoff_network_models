import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class SocialRelationsModel:
    """
    Social Relations Model for dyadic network data.

    The model decomposes a sociomatrix Y as:
        y_ij = μ + a_i + b_j + ε_ij  (Basic SRM)
    or with covariates:
        y_ij = β^T x_ij + a_i + b_j + ε_ij  (SRRM)

    where:
        - μ: grand mean (basic SRM)
        - β: regression coefficients (SRRM)
        - x_ij: covariate vector for dyad (i,j)
        - a_i: row effect (sender effect)
        - b_j: column effect (receiver effect)
        - ε_ij: residual

    The model estimates variance components:
        - σ²_a: variance of row effects
        - σ²_b: variance of column effects
        - σ_ab: covariance between row and column effects
        - σ²: residual variance
        - ρ: dyadic correlation (correlation between ε_ij and ε_ji)
    """

    def __init__(self, Y, X=None, node_names=None):
        """
        Initialize the SRM with a sociomatrix and optional covariates.

        Parameters
        ----------
        Y : array-like, shape (n_nodes, n_nodes)
            n×n sociomatrix where Y[i,j] is the relationship from i to j
        X : array-like or dict, optional
            n×n×p array of covariates where X[i,j,:] are the p covariates for dyad (i,j)
            Can also be a dictionary with keys:
                - 'row': n×p_r array of row (sender) covariates
                - 'col': n×p_c array of column (receiver) covariates
                - 'dyad': n×n×p_d array of dyadic covariates
        node_names : list of str, optional
            Names of the nodes (default: 0, 1, 2, ...)
        """
        if Y.shape[0] != Y.shape[1]:
            raise ValueError("Y must be a square matrix")

        self.Y = Y
        self.n = Y.shape[0]
        self.node_names = node_names if node_names else [str(i) for i in range(self.n)]

        # Process covariates
        self.X = np.array([])
        self.covariate_names = []
        self.has_covariates = X is not None

        if X is not None:
            if isinstance(X, dict):
                self.X, self.covariate_names = self._construct_covariate_matrix(X)
            else:
                self.X = X
                if X.ndim == 3:
                    self.covariate_names = [f'X{i+1}' for i in range(X.shape[2])]
                else:
                    raise ValueError("X must be n×n×p array or dictionary")

        # Results (to be computed)
        self.mu = np.nan
        self.beta = np.array([])  # Regression coefficients
        self.a = np.array([])
        self.b = np.array([])
        self.E = np.array([])
        self.sigma2_a = np.nan
        self.sigma2_b = np.nan
        self.sigma_ab = np.nan
        self.sigma2 = np.nan
        self.rho = np.nan

    def _construct_covariate_matrix(self, X_dict):
        """
        Construct a full covariate matrix from dictionary of row, column, and dyadic covariates.

        Parameters
        ----------
        X_dict : dict
            Dictionary with keys 'row', 'col', and/or 'dyad'

        Returns
        -------
        X : array-like, shape (n_nodes, n_nodes, n_covariates)
            n×n×p covariate matrix
        names : list of str
            Names of covariates
        """
        covariate_arrays = []
        names = []

        # Row (sender) covariates: x_r,i for each i
        if 'row' in X_dict:
            X_row = X_dict['row']  # n×p_r
            if X_row.ndim == 1:
                X_row = X_row.reshape(-1, 1)
            # Expand to n×n×p_r: X[i,j,k] = X_row[i,k]
            X_row_expanded = np.repeat(X_row[:, np.newaxis, :], self.n, axis=1)
            covariate_arrays.append(X_row_expanded)
            n_row_covariates = X_row.shape[1]
            names.extend([f'row_{i+1}' for i in range(n_row_covariates)])

        # Column (receiver) covariates: x_c,j for each j
        if 'col' in X_dict:
            X_col = X_dict['col']  # n×p_c
            if X_col.ndim == 1:
                X_col = X_col.reshape(-1, 1)
            # Expand to n×n×p_c: X[i,j,k] = X_col[j,k]
            X_col_expanded = np.repeat(X_col[np.newaxis, :, :], self.n, axis=0)
            covariate_arrays.append(X_col_expanded)
            n_col_covariates = X_col.shape[1]
            names.extend([f'col_{i+1}' for i in range(n_col_covariates)])

        # Dyadic covariates: x_d,ij for each (i,j)
        if 'dyad' in X_dict:
            X_dyad = X_dict['dyad']  # n×n×p_d or n×n
            if X_dyad.ndim == 2:
                X_dyad = X_dyad[:, :, np.newaxis]
            covariate_arrays.append(X_dyad)
            n_dyad_covariates = X_dyad.shape[2]
            names.extend([f'dyad_{i+1}' for i in range(n_dyad_covariates)])

        if not covariate_arrays:
            raise ValueError("X_dict must contain at least one of 'row', 'col', or 'dyad'")

        # Concatenate along last dimension
        X = np.concatenate(covariate_arrays, axis=2)
        return X, names

    def fit(self):
        """
        Fit the Social Relations Model to the data.
        If covariates are provided, fits SRRM using alternating estimation.

        Returns
        -------
        self : SocialRelationsModel
            Fitted model
        """
        # Create mask for off-diagonal elements
        mask = ~np.eye(self.n, dtype=bool)

        if not self.has_covariates:
            # Basic SRM without covariates
            # Grand mean (excluding diagonal)
            self.mu = np.mean(self.Y[mask])

            # Row effects (sender effects)
            self.a = np.array([np.mean(self.Y[i, mask[i]]) for i in range(self.n)]) - self.mu

            # Column effects (receiver effects)
            self.b = np.array([np.mean(self.Y[mask[:, j], j]) for j in range(self.n)]) - self.mu

            # Residuals
            self.E = self.Y - (self.mu + self.a[:, np.newaxis] + self.b[np.newaxis, :])
        else:
            # SRRM with covariates - use alternating estimation
            self._fit_srrm(mask)

        # Variance components
        self.sigma2_a = np.var(self.a, ddof=1)
        self.sigma2_b = np.var(self.b, ddof=1)
        self.sigma_ab = np.cov(self.a, self.b)[0, 1]
        self.sigma2 = np.var(self.E[mask], ddof=1)

        # Dyadic correlation
        epsilon_ij = []
        epsilon_ji = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                epsilon_ij.append(self.E[i, j])
                epsilon_ji.append(self.E[j, i])
        self.rho = np.corrcoef(epsilon_ij, epsilon_ji)[0, 1]

        return self

    def _fit_srrm(self, mask, max_iter=10, tol=1e-6):
        """
        Fit SRRM using alternating least squares.

        Parameters
        ----------
        mask : array-like, bool
            Boolean mask for off-diagonal elements
        max_iter : int, default=10
            Maximum number of iterations
        tol : float, default=1e-6
            Convergence tolerance
        """
        # Initialize row and column effects
        self.a = np.zeros(self.n)
        self.b = np.zeros(self.n)

        # Reshape X for regression: (n*n, p)
        p = self.X.shape[2]
        X_flat = self.X.reshape(-1, p)
        Y_flat = self.Y.flatten()
        mask_flat = mask.flatten()

        for iteration in range(max_iter):
            a_old, b_old = self.a.copy(), self.b.copy()

            # Step 1: Estimate β given a, b
            # Y - a1^T - 1b^T = X β + ε
            Y_adjusted = self.Y - self.a[:, np.newaxis] - self.b[np.newaxis, :]
            Y_adjusted_flat = Y_adjusted.flatten()

            # OLS regression on off-diagonal elements
            X_masked = X_flat[mask_flat]
            Y_masked = Y_adjusted_flat[mask_flat]
            self.beta = np.linalg.lstsq(X_masked, Y_masked, rcond=None)[0]

            # Step 2: Estimate a, b given β
            # Y - Xβ = a1^T + 1b^T + ε
            M = (self.X @ self.beta).reshape(self.n, self.n)
            Y_residual = self.Y - M

            # Row effects (excluding diagonal)
            self.a = np.array([np.mean(Y_residual[i, mask[i]]) for i in range(self.n)])

            # Column effects (excluding diagonal)
            self.b = np.array([np.mean(Y_residual[mask[:, j], j]) for j in range(self.n)])

            # Center row and column effects to ensure identifiability
            self.a = self.a - np.mean(self.a)
            self.b = self.b - np.mean(self.b)

            # Check convergence
            if (np.max(np.abs(self.a - a_old)) < tol and
                np.max(np.abs(self.b - b_old)) < tol):
                break

        # Compute final residuals
        M = (self.X @ self.beta).reshape(self.n, self.n)
        self.E = self.Y - M - self.a[:, np.newaxis] - self.b[np.newaxis, :]
        self.mu = np.nan  # No grand mean in SRRM (absorbed into intercept if included)

    def summary(self):
        """
        Get a summary of the fitted model.

        Returns
        -------
        summary_df : pandas.DataFrame
            Summary statistics of the model
        """
        if self.a.size == 0:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        total_var = self.sigma2_a + 2*self.sigma_ab + self.sigma2_b + self.sigma2

        if not self.has_covariates:
            summary = {
                'Component': ['Grand Mean (μ)', 'Row Variance (σ²_a)',
                             'Column Variance (σ²_b)', 'Row-Col Covariance (σ_ab)',
                             'Residual Variance (σ²)', 'Dyadic Correlation (ρ)'],
                'Value': [self.mu, self.sigma2_a, self.sigma2_b,
                         self.sigma_ab, self.sigma2, self.rho],
                '% of Total Var': [np.nan,
                                  100*self.sigma2_a/total_var,
                                  100*self.sigma2_b/total_var,
                                  100*2*self.sigma_ab/total_var,
                                  100*self.sigma2/total_var,
                                  np.nan]
            }
        else:
            # SRRM summary includes regression coefficients
            components = []
            values = []
            pct_var = []

            # Add regression coefficients
            for i, name in enumerate(self.covariate_names):
                components.append(f'β_{name}')
                values.append(self.beta[i])
                pct_var.append(np.nan)

            # Add variance components
            components.extend(['Row Variance (σ²_a)', 'Column Variance (σ²_b)',
                              'Row-Col Covariance (σ_ab)', 'Residual Variance (σ²)',
                              'Dyadic Correlation (ρ)'])
            values.extend([self.sigma2_a, self.sigma2_b, self.sigma_ab, self.sigma2, self.rho])
            pct_var.extend([100*self.sigma2_a/total_var,
                           100*self.sigma2_b/total_var,
                           100*2*self.sigma_ab/total_var,
                           100*self.sigma2/total_var,
                           np.nan])

            summary = {
                'Component': components,
                'Value': values,
                '% of Total Var': pct_var
            }

        return pd.DataFrame(summary)

    def get_effects(self):
        """
        Get the estimated node effects.

        Returns
        -------
        effects_df : pandas.DataFrame
            Node effects (row and column)
        """
        if self.a.size == 0:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        return pd.DataFrame({
            'Node': self.node_names,
            'Row Effect (a_i)': self.a,
            'Column Effect (b_i)': self.b
        })

    def plot_effects(self, figsize=(14, 6), save_path=None):
        """
        Create visualization of model results (Figure 1 from Hoff 2018).

        Parameters
        ----------
        figsize : tuple, default=(14, 6)
            Figure size
        save_path : str, optional
            Path to save the figure (default: None, don't save)

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if self.a.size == 0:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Left panel: Row effects vs Column effects
        ax1 = axes[0]
        ax1.scatter(self.a, self.b, s=100, alpha=0.7)
        for i, name in enumerate(self.node_names):
            ax1.annotate(name, (self.a[i], self.b[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.set_xlabel('Row Effect (âᵢ) - Sender', fontsize=12)
        ax1.set_ylabel('Column Effect (b̂ᵢ) - Receiver', fontsize=12)
        ax1.set_title('Sender vs Receiver Effects', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        corr_ab = np.corrcoef(self.a, self.b)[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {corr_ab:.3f}',
                transform=ax1.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Right panel: Dyadic residuals
        epsilon_ij = []
        epsilon_ji = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                epsilon_ij.append(self.E[i, j])
                epsilon_ji.append(self.E[j, i])

        ax2 = axes[1]
        ax2.scatter(epsilon_ij, epsilon_ji, s=50, alpha=0.6)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        lims = [min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
                max(ax2.get_xlim()[1], ax2.get_ylim()[1])]
        ax2.plot(lims, lims, 'r--', alpha=0.3, linewidth=1)

        ax2.set_xlabel('ε̂ᵢ,ⱼ', fontsize=12)
        ax2.set_ylabel('ε̂ⱼ,ᵢ', fontsize=12)
        ax2.set_title('Dyadic Residual Correlation', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        ax2.text(0.05, 0.95, f'Correlation (ρ): {self.rho:.3f}',
                transform=ax2.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return fig


def load_ir90s_data(data_folder, countries=None, load_covariates=False):
    """
    Load and prepare the IR90s export data with optional covariates.

    Parameters
    ----------
    data_folder : str
        Path to data folder
    countries : list of str, optional
        List of country codes to include (default: all countries)
    load_covariates : bool, default=False
        Whether to load node and dyadic covariates

    Returns
    -------
    Y : array-like
        Sociomatrix in log billions of dollars
    country_names : list
        List of country codes
    covariates : dict or None
        Dictionary with keys 'row', 'col', 'dyad' containing covariate arrays
        Only returned if load_covariates=True
    """
    # Load dyadic data
    dyad_df = pd.read_csv(f'{data_folder}/IR90s_dyadvars.csv')
    country_codes = dyad_df.iloc[:, 0].values

    if countries is None:
        countries = list(country_codes)

    # Find indices of countries
    country_indices = {}
    for i, code in enumerate(country_codes):
        if code in countries:
            country_indices[code] = i

    # Build sociomatrix
    n = len(countries)
    Y = np.zeros((n, n))

    for i, sender in enumerate(countries):
        if sender in country_indices:
            row_idx = country_indices[sender]
            for j, receiver in enumerate(countries):
                col_name = f'{receiver}.exports'
                Y[i, j] = dyad_df.iloc[row_idx][col_name]

    # Convert to log scale
    Y_log = np.log(Y + 0.01)

    covariates = None
    if load_covariates:
        # Load node variables
        node_df = pd.read_csv(f'{data_folder}/IR90s_nodevars.csv')

        # Build covariate arrays
        X_row = np.zeros((n, 2))  # GDP and polity for senders
        X_col = np.zeros((n, 2))  # GDP and polity for receivers
        X_dyad = np.zeros((n, n))  # Distance between countries

        for i, sender in enumerate(countries):
            if sender in country_indices:
                row_idx = country_indices[sender]
                # Get sender's GDP and polity
                X_row[i, 0] = node_df.iloc[row_idx]['gdp']  # log GDP
                X_row[i, 1] = node_df.iloc[row_idx]['polity']

                # Get receiver's GDP and polity (same source)
                X_col[i, 0] = node_df.iloc[row_idx]['gdp']  # log GDP
                X_col[i, 1] = node_df.iloc[row_idx]['polity']

                # Get distances
                for j, receiver in enumerate(countries):
                    dist_col = f'{receiver}.distance'
                    if dist_col in dyad_df.columns:
                        X_dyad[i, j] = dyad_df.iloc[row_idx][dist_col]

        # Convert distance to log scale
        X_dyad = np.log(X_dyad + 1)  # Add 1 to avoid log(0)

        covariates = {
            'row': X_row,
            'col': X_col,
            'dyad': X_dyad
        }

        # Store names for later use
        covariates['names'] = {
            'row': ['GDP', 'Polity'],
            'col': ['GDP', 'Polity'],
            'dyad': ['Distance']
        }

    return Y_log, countries, covariates


if __name__ == "__main__":
    # Example usage - Basic SRM
    print("="*70)
    print("EXAMPLE 1: BASIC SOCIAL RELATIONS MODEL (Section 2.1)")
    print("="*70)

    print("\nLoading IR90s export data (subset of 7 countries)...")
    countries_subset = ['USA', 'JPN', 'CHN', 'ITA', 'NTH', 'DEN', 'FRN']
    Y, countries, _ = load_ir90s_data('./data', countries=countries_subset)

    print(f"\nFitting Social Relations Model for {len(countries)} countries...")
    srm = SocialRelationsModel(Y, node_names=countries)
    srm.fit()

    print("\n" + "="*70)
    print("SOCIAL RELATIONS MODEL - RESULTS")
    print("="*70)

    print("\nModel Summary:")
    print(srm.summary().to_string(index=False))

    print("\n\nNode Effects:")
    print(srm.get_effects().to_string(index=False))

    print("\n\nCreating visualization...")
    srm.plot_effects(save_path='srm_results.png')
    plt.show()

    # Example usage - SRRM with covariates
    print("\n\n" + "="*70)
    print("EXAMPLE 2: SOCIAL RELATIONS REGRESSION MODEL (Section 2.2)")
    print("="*70)

    print("\nLoading IR90s export data with covariates (30 countries)...")
    countries_30 = ['USA', 'JPN', 'CHN', 'ITA', 'NTH', 'DEN', 'FRN',
                    'ARG', 'AUL', 'BEL', 'BNG', 'BRA', 'CAN', 'COL', 'EGY',
                    'IND', 'INS', 'IRN', 'MEX', 'PAK', 'PHI', 'POL', 'ROK',
                    'SAF', 'SAU', 'SPN', 'SWD', 'TAW', 'THI', 'TUR']
    Y_30, countries_30, covariates = load_ir90s_data('./data', countries=countries_30,
                                                      load_covariates=True)

    print(f"\nFitting Social Relations Regression Model for {len(countries_30)} countries...")
    print("Covariates: Exporter GDP, Exporter Polity, Importer GDP, Importer Polity, Distance")

    srrm = SocialRelationsModel(Y_30, X=covariates, node_names=countries_30)
    srrm.fit()

    print("\n" + "="*70)
    print("SOCIAL RELATIONS REGRESSION MODEL - RESULTS")
    print("="*70)

    print("\nModel Summary (with regression coefficients):")
    print(srrm.summary().to_string(index=False))

    print("\n\nNode Effects (after controlling for covariates):")
    print(srrm.get_effects().to_string(index=False))

    print("\n" + "="*70)