# PHASE 5 REFACTORED DETECTION METHODS
# These will replace the corresponding methods in quantum_protocol.py

def _apply_losses_phase5(self, alice_results, bob_results, alice_bases, bob_bases,
                        time_tags_A=None, time_tags_B=None, has_signal=None):
    """
    Apply all loss mechanisms using per-side click modeling (Phase 5 refactoring).

    KEY CHANGE: Track signal and noise clicks PER DETECTOR, not as coincidence.
    This allows signal+noise coincidences (dominant at high loss).

    Click model:
        click_A = signal_A OR noise_A
        click_B = signal_B OR noise_B
        coincidence = click_A AND click_B

    Args:
        alice_results: Alice's measurement results (quantum correlated)
        bob_results: Bob's measurement results (quantum correlated)
        alice_bases: Alice's basis choices
        bob_bases: Bob's basis choices
        time_tags_A: Alice's time tags (optional)
        time_tags_B: Bob's time tags (optional)
        has_signal: Multi-pair signal mask (k>0 events), optional

    Returns:
        Tuple of filtered results, bases, and time tags
    """
    n = len(alice_results)

    # ========================================================================
    # STEP 1: Initialize per-side signal masks
    # ========================================================================

    # Start with photon generation (has_signal from multi-pair model)
    # has_signal = True if k>0 pairs generated
    sig_A = has_signal if has_signal is not None else np.ones(n, dtype=bool)
    sig_B = has_signal if has_signal is not None else np.ones(n, dtype=bool)

    # ========================================================================
    # STEP 2: Apply channel losses (per side)
    # ========================================================================

    # Fiber loss (applied per arm)
    if self.config.enable_fiber_loss:
        loss_dB_A = self.config.distance_km_A * self.config.fiber_loss_dB_per_km + self.config.loss_dB_A
        loss_dB_B = self.config.distance_km_B * self.config.fiber_loss_dB_per_km + self.config.loss_dB_B

        # Repeater model (optional)
        if getattr(self.config, "enable_repeaters", False) and getattr(self.config, "num_repeaters", 0) > 0:
            total_gain = self.config.num_repeaters * self.config.repeater_gain_dB
            loss_dB_A = max(0.0, loss_dB_A - total_gain)
            loss_dB_B = max(0.0, loss_dB_B - total_gain)

        eta_A = 10 ** (-loss_dB_A / 10)
        eta_B = 10 ** (-loss_dB_B / 10)

        # Apply per side
        sig_A &= self.rng.random(n) < eta_A
        sig_B &= self.rng.random(n) < eta_B

    # Satellite loss (global - both photons share same path)
    if self.config.enable_satellite:
        sat_loss_dB = compute_satellite_loss(self.config)
        eta_sat = 10 ** (-sat_loss_dB / 10)

        # Apply globally (both photons affected equally)
        survival = self.rng.random(n) < eta_sat
        sig_A &= survival
        sig_B &= survival

    # Insertion loss (applied per side)
    if self.config.enable_insertion_loss:
        eta_ins = 10 ** (-self.config.insertion_loss_dB / 10)
        sig_A &= self.rng.random(n) < eta_ins
        sig_B &= self.rng.random(n) < eta_ins

    # ========================================================================
    # STEP 3: Apply detector efficiency (per side)
    # ========================================================================

    if self.config.enable_detector_loss:
        # Heralding efficiency (global filter)
        heralding = self.rng.random(n) < self.config.heralding_efficiency
        sig_A &= heralding
        sig_B &= heralding

        # End detector efficiency (per detector)
        eta_det = self.config.end_detector_efficiency
        sig_A &= self.rng.random(n) < eta_det
        sig_B &= self.rng.random(n) < eta_det

    # ========================================================================
    # STEP 4: Compute noise clicks (per side)
    # ========================================================================

    # Get coincidence window for noise rate calculation
    coinc_window = getattr(self.config, 'coincidence_window_ns', 1.0)

    # Calculate per-side noise click probabilities
    p_noise_A, p_noise_B = self._compute_noise_probabilities(n, coinc_window)

    # Generate noise clicks (independent per side)
    noise_A = self.rng.random(n) < p_noise_A
    noise_B = self.rng.random(n) < p_noise_B

    # ========================================================================
    # STEP 5: Combine signal and noise into total clicks
    # ========================================================================

    click_A = sig_A | noise_A
    click_B = sig_B | noise_B

    # ========================================================================
    # STEP 6: Apply deadtime filtering (per detector)
    # ========================================================================

    if self.config.enable_deadtime and time_tags_A is not None:
        click_A = self._apply_deadtime_vectorized(click_A, time_tags_A, self.config.deadtime_ns)
        click_B = self._apply_deadtime_vectorized(click_B, time_tags_B, self.config.deadtime_ns)

    # ========================================================================
    # STEP 7: Apply saturation (per detector)
    # ========================================================================

    if self.config.enable_saturation:
        if self.config.repetition_rate_Hz > self.config.saturation_rate:
            saturation_prob = self.config.saturation_rate / self.config.repetition_rate_Hz
            click_A &= self.rng.random(n) < saturation_prob
            click_B &= self.rng.random(n) < saturation_prob

    # ========================================================================
    # STEP 8: Determine measurement results
    # ========================================================================

    # If noise-only click (no signal), result is random
    # If signal click, use quantum result
    alice_results = np.where(sig_A, alice_results, self.rng.integers(0, 2, n))
    bob_results = np.where(sig_B, bob_results, self.rng.integers(0, 2, n))

    # ========================================================================
    # STEP 9: Coincidence = both detectors click
    # ========================================================================

    coincidence = click_A & click_B

    # ========================================================================
    # STEP 10: Filter to coincidence events
    # ========================================================================

    filtered = (
        alice_results[coincidence],
        bob_results[coincidence],
        alice_bases[coincidence],
        bob_bases[coincidence]
    )

    if time_tags_A is not None:
        return (*filtered, time_tags_A[coincidence], time_tags_B[coincidence])
    return (*filtered, None, None)


def _compute_noise_probabilities(self, n, coincidence_window_ns=1.0):
    """
    Compute per-side noise click probabilities using independent union.

    Sources: dark counts, background light, afterpulsing

    Formula: P(noise) = 1 - (1-P_dark)*(1-P_bg)*(1-P_afterpulse)

    Returns:
        Tuple of (p_noise_A, p_noise_B)
    """
    # Dark counts
    p_dark = 0.0
    if self.config.enable_dark_counts:
        if self.config.use_dark_cps:
            p_dark = self.config.dark_cps * coincidence_window_ns * 1e-9
        else:
            p_dark = self.config.dark_prob
        p_dark = float(np.clip(p_dark, 0.0, 1.0))

    # Background light
    p_bg = 0.0
    if self.config.enable_background:
        if hasattr(self.config, 'background_cps'):
            p_bg = self.config.background_cps * coincidence_window_ns * 1e-9
        else:
            p_bg = self.config.Y0
        p_bg = float(np.clip(p_bg, 0.0, 1.0))

    # Satellite background (daytime)
    p_sat_bg = 0.0
    if self.config.enable_satellite and self.config.is_daytime:
        p_sat_bg = self.config.satellite_background_cps / self.config.repetition_rate_Hz
        p_sat_bg = float(np.clip(p_sat_bg, 0.0, 1.0))

    # Afterpulsing (simplified: treat as increased dark rate)
    # Full model would generate delayed clicks
    p_afterpulse = 0.0
    if self.config.enable_afterpulsing:
        # Simplified: add to noise probability
        # Proper model: trigger delayed clicks after real clicks
        p_afterpulse = self.config.afterpulsing_prob * 0.1  # Scaled down
        p_afterpulse = float(np.clip(p_afterpulse, 0.0, 1.0))

    # INDEPENDENT UNION (not simple sum!)
    # P(A or B or C) = 1 - P(not A) * P(not B) * P(not C)
    p_no_dark = 1.0 - p_dark
    p_no_bg = 1.0 - p_bg
    p_no_sat = 1.0 - p_sat_bg
    p_no_ap = 1.0 - p_afterpulse

    p_noise = 1.0 - (p_no_dark * p_no_bg * p_no_sat * p_no_ap)
    p_noise = float(np.clip(p_noise, 0.0, 1.0))

    # For now, use same noise for both sides
    # Could make them different if detectors have different characteristics
    return p_noise, p_noise


def _apply_deadtime_vectorized(self, clicks, time_tags, deadtime_ns):
    """
    Apply detector deadtime filtering (vectorized, per detector).

    Suppresses clicks that occur within deadtime_ns of a previous click.

    Args:
        clicks: Boolean array of click events
        time_tags: Time stamps of events
        deadtime_ns: Deadtime in nanoseconds

    Returns:
        Filtered click array
    """
    if not np.any(clicks):
        return clicks

    # Get times of clicks
    click_indices = np.where(clicks)[0]
    click_times = time_tags[click_indices]

    # Compute intervals between successive clicks
    intervals = np.diff(click_times)

    # Mark clicks too close to previous
    # First click always survives
    too_close = np.concatenate([[False], intervals < deadtime_ns])

    # Create filtered clicks array
    filtered_clicks = clicks.copy()

    # Suppress clicks that are too close
    suppressed_indices = click_indices[too_close]
    filtered_clicks[suppressed_indices] = False

    return filtered_clicks
