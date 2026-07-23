# ecm_pack: 自包含的「可定制 ECM + 电路耦合 + 热网络」电池包仿真库
from .data import lookup_1d, lookup_nd, as_callable, build_3d_table
from .ecm import ECMCellSpec, ECMCell
from .circuit import Netlist, solve_circuit, setup_circuit, setup_two_group, setup_series_bypass
from .thermal import ThermalNetwork
from .thermal3d import Cell3DThermal
from .defaults import cell_314ah_spec, make_ocv_314ah, R0_314ah, R1_314ah, C1_314ah, dUdT_314ah
from .pack import Pack

__all__ = [
    "lookup_1d", "lookup_nd", "as_callable", "build_3d_table",
    "ECMCellSpec", "ECMCell",
    "Netlist", "solve_circuit", "setup_circuit", "setup_two_group", "setup_series_bypass",
    "ThermalNetwork", "Cell3DThermal",
    "cell_314ah_spec", "make_ocv_314ah", "R0_314ah", "R1_314ah", "C1_314ah", "dUdT_314ah",
    "Pack",
]
