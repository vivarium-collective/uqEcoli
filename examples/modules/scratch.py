import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    from scipy.integrate import solve_ivp
    import polars as pl
    import dataclasses as dc
    from uuid import uuid4

    def f(
        params: np.ndarray,
        param_names: list[str],
        baseline_value: float,
        n_timesteps: int,
        random_seed: int = 42,
        timestep: float = 1.0,
    ) -> np.ndarray:
        """
        Stochastic Whole Cell Model (WCM) timeseries generator for E. coli.

        Models 8 coupled biological variables using SDEs (Langevin formulation):
          0: glucose        - extracellular carbon source (mM)
          1: atp            - energy currency (mM)
          2: precursors     - biosynthetic building blocks (mM, proxy for aa/ntp pools)
          3: ribosomes      - translational machinery (molecules/cell)
          4: mrna           - bulk mRNA pool (molecules/cell)
          5: proteins       - bulk protein pool (molecules/cell)
          6: cell_volume    - cell size proxy (fL)
          7: growth_rate    - instantaneous growth rate (1/hr)

        Parameter mapping (by param_names, with fallback defaults):
          - 'mu_max'        : max growth rate (default 0.8 /hr)
          - 'k_glucose'     : glucose Michaelis constant (default 0.015 mM)
          - 'atp_yield'     : ATP yield per glucose (default 28.0)
          - 'k_atp'         : ATP saturation for ribosomes (default 1.0 mM)
          - 'rib_efficiency': ribosome catalytic rate (default 20.0 aa/s/ribosome)
          - 'mrna_deg'      : mRNA degradation rate (default 0.1 /min)
          - 'prot_deg'      : protein degradation rate (default 0.001 /min)
          - 'noise_scale'   : global intrinsic noise scaling (default 1.0)

        Returns
        -------
        np.ndarray of shape (n_timesteps, 8)
            Each column is one biological variable trajectory.
        """
        rng = np.random.default_rng(random_seed)

        # ------------------------------------------------------------------ #
        # 1. Map params → named biological constants
        # ------------------------------------------------------------------ #
        param_dict = dict(zip(param_names, params))

        def get(name, default):
            return float(param_dict[name]) if name in param_dict else default

        mu_max = get("mu_max", 0.8)  # /hr
        k_glucose = get("k_glucose", 0.015)  # mM
        atp_yield = get("atp_yield", 28.0)  # ATP/glucose
        k_atp = get("k_atp", 1.0)  # mM
        rib_efficiency = get("rib_efficiency", 20.0)  # aa/s/ribosome → scales protein synthesis
        mrna_deg = get("mrna_deg", 0.1)  # /min
        prot_deg = get("prot_deg", 0.001)  # /min
        noise_scale = get("noise_scale", 1.0)

        # ------------------------------------------------------------------ #
        # 2. Baseline-scaled initial conditions
        # ------------------------------------------------------------------ #
        # baseline_value is treated as a scaling factor on the reference cell state
        b = baseline_value / 1.0  # normalised (reference = 1.0)

        glucose_0 = b * 10.0  # mM  – batch culture starting [glucose]
        atp_0 = b * 3.0  # mM  – typical cytoplasmic ATP
        precursor_0 = b * 2.0  # mM  – amino acid / NTP pool proxy
        ribosome_0 = b * 7_000  # molecules/cell
        mrna_0 = b * 1_500  # molecules/cell
        protein_0 = b * 2_000_000  # molecules/cell
        volume_0 = b * 1.0  # fL
        mu_0 = 0.0  # 1/hr

        state = np.array([glucose_0, atp_0, precursor_0, ribosome_0, mrna_0, protein_0, volume_0, mu_0])

        # ------------------------------------------------------------------ #
        # 3. Time grid  (assume 1 timestep = 1 minute)
        # ------------------------------------------------------------------ #
        dt = 1.0 / 60.0  # hours per step (1 min)
        sqrt_dt = np.sqrt(dt)

        # ------------------------------------------------------------------ #
        # 4. Intrinsic noise amplitudes (biochemical / Langevin)
        #    σ scaled as ~sqrt(mean) to mimic Poisson shot noise
        # ------------------------------------------------------------------ #
        def noise_amplitudes(s):
            glucose, atp, prec, rib, mrna, prot, vol, mu = s
            eps = 1e-9
            sigma = noise_scale * np.array([
                0.005 * np.sqrt(abs(glucose) + eps),  # glucose fluctuations
                0.02 * np.sqrt(abs(atp) + eps),  # ATP noise
                0.015 * np.sqrt(abs(prec) + eps),  # precursor noise
                0.01 * np.sqrt(abs(rib) + eps),  # ribosome birth/death
                0.03 * np.sqrt(abs(mrna) + eps),  # mRNA burst noise
                0.005 * np.sqrt(abs(prot) + eps),  # protein noise
                0.002 * vol,  # volume measurement noise
                0.005 * (abs(mu) + 0.01),  # growth rate fluctuation
            ])
            return sigma

        # ------------------------------------------------------------------ #
        # 5. Deterministic drift  f(s, t)  — coupled ODEs (per hour)
        # ------------------------------------------------------------------ #
        def drift(s):
            glucose, atp, prec, rib, mrna, prot, vol, mu = s

            # Clamp to physiological bounds
            glucose = max(glucose, 0.0)
            atp = max(atp, 0.0)
            prec = max(prec, 0.0)
            rib = max(rib, 0.0)
            mrna = max(mrna, 0.0)
            prot = max(prot, 0.0)
            vol = max(vol, 0.1)

            # --- Monod growth kinetics ---
            mu_inst = mu_max * glucose / (k_glucose + glucose)  # /hr

            # --- ATP metabolism ---
            glucose_uptake = mu_inst * glucose / (k_glucose + glucose + 1e-9)  # relative flux
            atp_prod = atp_yield * glucose_uptake * 0.5  # production (scaled)
            atp_cons = mu_inst * 2.0 + 0.5  # growth + maintenance ATP draw

            # --- Precursor (amino acid / NTP) pool ---
            prec_synth = atp * prec_prod_rate(atp, k_atp) * 0.3
            prec_cons = rib * rib_efficiency / 3600 * 0.001 * (prec / (prec + 0.5))

            # --- mRNA dynamics (transcription – degradation – dilution) ---
            transcription = 0.5 * mu_inst * mrna  # growth-coupled transcription
            mrna_loss = (mrna_deg + mu_inst) * mrna  # degradation + dilution

            # --- Ribosome dynamics ---
            rib_synth = 0.0008 * mu_inst * prot  # fraction of protein synthesis = ribosomes
            rib_loss = (0.005 + mu_inst) * rib

            # --- Protein dynamics ---
            prot_synth = rib_efficiency / 3600 * rib * (atp / (k_atp + atp)) * (prec / (prec + 0.5))
            prot_loss = (prot_deg + mu_inst) * prot

            # --- Volume (cell growth) ---
            dvol = mu_inst * vol

            # --- Glucose consumption ---
            dglucose = -mu_inst * glucose * 0.15

            # --- Pack derivatives ---
            ds = np.array([
                dglucose,  # glucose
                atp_prod - atp_cons,  # atp
                prec_synth - prec_cons,  # precursors
                rib_synth - rib_loss,  # ribosomes
                transcription - mrna_loss,  # mrna
                prot_synth - prot_loss,  # proteins
                dvol,  # volume
                (mu_inst - mu) * 10.0,  # growth_rate (relaxation to mu_inst)
            ])
            return ds

        def prec_prod_rate(atp, k):
            return atp / (k + atp)

        # ------------------------------------------------------------------ #
        # 6. Euler–Maruyama integration
        # ------------------------------------------------------------------ #
        trajectory = np.empty((n_timesteps, 8))
        trajectory[0] = state.copy()

        for i in range(1, n_timesteps):
            f = drift(state)
            sigma = noise_amplitudes(state)
            dW = rng.standard_normal(8)

            state = state + f * dt + sigma * dW * sqrt_dt

            # Hard physiological lower bounds
            state[0] = max(state[0], 0.0)  # glucose ≥ 0
            state[1] = np.clip(state[1], 0.0, 10.0)  # ATP 0–10 mM
            state[2] = max(state[2], 0.0)  # precursors ≥ 0
            state[3] = max(state[3], 0.0)  # ribosomes ≥ 0
            state[4] = max(state[4], 0.0)  # mRNA ≥ 0
            state[5] = max(state[5], 0.0)  # proteins ≥ 0
            state[6] = max(state[6], 0.1)  # volume ≥ 0.1 fL
            state[7] = np.clip(state[7], 0.0, 2.0)  # growth rate 0–2 /hr

            trajectory[i] = state.copy()

        # return trajectory
        t = np.arange(0, trajectory.shape[0], timestep)
        print(t.shape)
        d = dict(zip(param_names, trajectory))
        # d.update({"time": t})
        return trajectory

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
