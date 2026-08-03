# ecm_pack: 自包含的「可定制 ECM + 电路耦合 + 热网络」电池包仿真库
from .aging import AgingParams, AgingState
from .circuit import Netlist, setup_circuit, setup_series_bypass, setup_two_group, solve_circuit
from .data import as_callable, build_3d_table, lookup_1d, lookup_nd, temp_aware
from .defaults import (
    C1_314ah, R0_314ah, R1_314ah, cell_314ah_spec, dUdT_314ah, make_ocv_314ah,
    build_r0_soc_t_table, build_r1_soc_t_table,
    capacity_empirical, r0_empirical, r1_empirical, c1_empirical,
)
from .ecm import ECMCell, ECMCellSpec
from .pack import Pack
from .thermal import ThermalNetwork
from .thermal3d import Cell3DThermal, CellThermalModel
from .thermal3d_stack import StackThermal3D
__all__ = [
    "lookup_1d", "lookup_nd", "as_callable", "build_3d_table", "temp_aware",
    "ECMCellSpec", "ECMCell",
    "Netlist", "solve_circuit", "setup_circuit", "setup_two_group", "setup_series_bypass",
    "ThermalNetwork", "CellThermalModel", "Cell3DThermal", "StackThermal3D",
    "cell_314ah_spec", "make_ocv_314ah", "R0_314ah", "R1_314ah", "C1_314ah", "dUdT_314ah",
    "build_r0_soc_t_table", "build_r1_soc_t_table",
    "r0_empirical", "r1_empirical", "c1_empirical", "capacity_empirical",
    "AgingParams", "AgingState",
    "Pack",
]
