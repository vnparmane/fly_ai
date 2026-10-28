"""Brian2-based backend for Drosophila brain simulations."""

import logging
import numpy as np
import scipy.sparse as sp
from textwrap import dedent

import brian2 as b2
from brian2 import mV, ms, Hz, prefs

prefs.codegen.target = "numpy"

from flybrain.brain.lif import LIFParams, SpikeBatch, SpikingBrain
from flybrain.data.connectome import Connectome

logger = logging.getLogger(__name__)


class Brian2Brain:
    """SpikingBrain implementation using the Brian2 simulation framework.
    Faithfully reproduces the Shiu et al. (2024) LIF architecture.
    """

    def __init__(self, conn: Connectome, params: LIFParams | None = None) -> None:
        if params is None:
            params = LIFParams()
        self.params = params
        self.conn = conn
        self.n_neurons = len(conn.neuron_ids)

        logger.info(f"Building Brian2Brain network (N={self.n_neurons})...")

        # Map params to Brian2 units namespace
        self.ns = {
            "v_0": params.v_rest_mv * mV,
            "v_rst": params.v_reset_mv * mV,
            "v_th": params.v_thresh_mv * mV,
            "t_mbr": params.tau_mem_ms * ms,
            "tau": params.tau_syn_ms * ms,
            "t_rfc": params.refractory_ms * ms,
            "t_dly": params.delay_ms * ms,
            "w_syn": params.w_syn_mv * mV,
            "f_poi": params.poisson_weight_scale,
        }

        # Equations matching Shiu et al. (2024)
        eqs = dedent("""
            dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
            dg/dt = -g / tau               : volt (unless refractory)
            rfc                            : second
        """)

        # Neuron group
        self.neu = b2.NeuronGroup(
            N=self.n_neurons,
            model=eqs,
            method="linear",
            threshold="v > v_th",
            reset="v = v_rst; g = 0 * mV",
            refractory="rfc",
            namespace=self.ns,
            name="fly_neurons",
        )
        self.neu.v = self.ns["v_0"]
        self.neu.g = 0 * mV
        self.neu.rfc = self.ns["t_rfc"]

        # Synapses
        logger.info("Initializing Brian2 Synapses...")
        self.syn = b2.Synapses(
            self.neu,
            self.neu,
            model="w : volt",
            on_pre="g += w",
            delay=self.ns["t_dly"],
            namespace=self.ns,
            name="fly_synapses",
        )

        # Connect synapses from CSR matrix
        # conn.W is CSR where W[post, pre] = sign * syn_count
        # sp.find returns (row, col, data) = (post, pre, weight)
        post_idx, pre_idx, weights = sp.find(conn.W)
        if len(pre_idx) > 0:
            self.syn.connect(i=pre_idx, j=post_idx)
            self.syn.w = weights * self.ns["w_syn"]

        # Poisson input layer: dynamic PoissonGroup connected to neu
        self._poisson_rates = np.zeros(self.n_neurons, dtype=np.float32) * Hz
        self.poi_group = b2.PoissonGroup(self.n_neurons, rates=self._poisson_rates, name="fly_poisson_inputs")
        
        self.poi_syn = b2.Synapses(
            self.poi_group,
            self.neu,
            model="w_poi : volt",
            on_pre="v += w_poi",
            namespace=self.ns,
            name="fly_poisson_synapses",
        )
        self.poi_syn.connect(j="i")  # 1-to-1 connection
        self.poi_syn.w_poi = self.ns["w_syn"] * self.ns["f_poi"]

        # Spike monitor to record emitted spikes
        self.spk_mon = b2.SpikeMonitor(self.neu)

        # Encapsulate inside Brian2 Network
        self.net = b2.Network(self.neu, self.syn, self.poi_group, self.poi_syn, self.spk_mon)
        self.current_time_ms = 0.0

        # Store initial pristine state snapshot
        self.net.store("initial")
        logger.info("Brian2Brain network initialized successfully.")

    def reset(self, seed: int) -> None:
        """Reset network state variables back to resting potential."""
        b2.seed(seed)
        np.random.seed(seed)
        self.net.restore("initial")
        self._poisson_rates[:] = 0.0 * Hz
        self.poi_group.rates = self._poisson_rates
        self.neu.v = self.ns["v_0"]
        self.neu.g = 0.0 * mV
        self.neu.rfc = self.ns["t_rfc"]
        self.current_time_ms = 0.0

    def set_poisson_input(self, neuron_idx: np.ndarray, rate_hz: np.ndarray) -> None:
        """Set continuous Poisson firing rates on designated neurons."""
        if len(neuron_idx) == 0:
            return
        neuron_idx = np.asarray(neuron_idx, dtype=np.int32)
        rate_hz = np.asarray(rate_hz, dtype=np.float32)

        # Update rates in PoissonGroup
        rates_copy = np.array(self.poi_group.rates / Hz)
        rates_copy[neuron_idx] = rate_hz
        self.poi_group.rates = rates_copy * Hz

        # For neurons receiving Poisson drive > 0, set refractory to 0 as in Shiu model
        active_mask = rate_hz > 0
        if np.any(active_mask):
            self.neu.rfc[neuron_idx[active_mask]] = 0 * ms
        inactive_mask = rate_hz == 0
        if np.any(inactive_mask):
            self.neu.rfc[neuron_idx[inactive_mask]] = self.ns["t_rfc"]

    def step(self, duration_ms: float) -> SpikeBatch:
        """Advance the simulation by duration_ms and return the emitted spikes."""
        t_start = self.net.t / ms
        self.net.run(duration_ms * ms, report=None)
        t_end = self.net.t / ms

        # Extract spikes that occurred in this interval
        spike_times = np.array(self.spk_mon.t / ms, dtype=np.float32)
        spike_indices = np.array(self.spk_mon.i, dtype=np.int32)

        batch_mask = (spike_times >= t_start) & (spike_times < t_end)
        batch_times = spike_times[batch_mask] - t_start
        batch_indices = spike_indices[batch_mask]

        self.current_time_ms = float(t_end)

        return SpikeBatch(
            t_ms=batch_times,
            neuron_idx=batch_indices,
            duration_ms=float(duration_ms),
        )
