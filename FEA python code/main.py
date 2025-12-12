# -*- coding: utf-8 -*-
# made by changsub
# Global analysis configuration for linear and nonlinear simulations.
# Defines the AnalysisConfig dataclass, common I/O paths, material
# parameters, and load/boundary-condition specifications that are
# shared by the linear postprocessing and nonlinear driver scripts.

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---- Type Hints ----
NodeTuple = Tuple[int, Optional[float], Optional[float], Optional[float]]

# =============================================================================
# Text Loader Utilities (Same as before - No changes needed here)
# =============================================================================
_WS_MAP = {'\u00a0': ' ', '\u2007': ' ', '\u202f': ' '}

def _clean_text(s: str) -> str:
    for bad, rep in _WS_MAP.items(): s = s.replace(bad, rep)
    return s.replace('\t', ' ').replace(',', ' ').replace(';', ' ')

def _expand_range_token(tok: str) -> List[int]:
    t = tok.strip()
    if re.fullmatch(r'\d+:\d+(?::\-?\d+)?', t):
        parts = [int(x) for x in t.split(':')]
        if len(parts) == 2: a, b = parts; step = 1 if a <= b else -1
        else: a, b, step = parts
        return list(range(a, b + (1 if step > 0 else -1), step))
    if re.fullmatch(r'\d+\-\d+', t):
        a, b = [int(x) for x in t.split('-', 1)]
        return list(range(a, b + (1 if a <= b else -1), 1 if a <= b else -1))
    if re.fullmatch(r'\d+', t): return [int(t)]
    raise ValueError(f"bad token: {tok}")

def load_nodes_txt(path: str) -> List[int]:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"node txt not found: {path}")
    raw = p.read_text(encoding='utf-8-sig', errors='ignore')
    raw = "\n".join(line.split('#', 1)[0] for line in raw.splitlines())
    tokens = [t for t in re.split(r'\s+', _clean_text(raw)) if t]
    out = []
    for t in tokens: out.extend(_expand_range_token(t))
    return sorted(set(out))

def load_int_pairs_txt(path: str) -> List[Tuple[int, int]]:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"pair txt not found: {path}")
    nums = [int(x) for x in re.findall(r'\d+', _clean_text(p.read_text(encoding='utf-8-sig', errors='ignore')))]
    if len(nums) % 2 != 0: raise ValueError(f"Odd count in pairs file: {path}")
    return list(zip(nums[::2], nums[1::2]))

def load_faces_nodes_txt(path: str, arity: Optional[int] = None) -> List[Tuple[int, ...]]:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"faces_node txt not found: {path}")
    nums = [int(x) for x in re.findall(r'\d+', _clean_text(p.read_text(encoding='utf-8-sig', errors='ignore')))]
    if not nums: return []
    if arity is None: arity = 3 if len(nums) % 3 == 0 and len(nums) % 4 != 0 else 4
    if len(nums) % arity != 0: raise ValueError(f"Count {len(nums)} not div by {arity} in {path}")
    return [tuple(nums[i:i+arity]) for i in range(0, len(nums), arity)]

def _expand_sets_from_txt(sets, *, face_arity=None):
    if not sets: return {}
    out = {}
    for name, cfg0 in sets.items():
        cfg = dict(cfg0)
        if "nodes_txt" in cfg: cfg["nodes"] = load_nodes_txt(cfg.pop("nodes_txt"))
        if "seed_nodes_txt" in cfg: cfg["seed_nodes"] = load_nodes_txt(cfg.pop("seed_nodes_txt"))
        if "elements_txt" in cfg: cfg["elements"] = load_nodes_txt(cfg.pop("elements_txt"))
        if "faces_txt" in cfg: cfg["faces"] = load_int_pairs_txt(cfg.pop("faces_txt"))
        if "faces_node_txt" in cfg: cfg["faces_node"] = load_faces_nodes_txt(cfg.pop("faces_node_txt"), arity=face_arity)
        out[name] = cfg
    return out

def build_dirichlet_list(nodes, ux=None, uy=None, uz=None): return [(int(n), ux, uy, uz) for n in nodes]
def build_scalar_dirichlet_list(nodes, val): return [(int(n), val) for n in nodes]
def merge_dirichlet_by_node(existing, new_items):
    merged = {t[0]: t for t in existing + new_items}
    return [merged[k] for k in sorted(merged.keys())]

# =============================================================================
# Analysis Configuration Class
# =============================================================================
@dataclass
class AnalysisConfig:
    # Core Settings
    Dimension: str = "2D"
    system: str    = "linear elasticity"
    analytical_conditions: str = "plane strain"
    element_order: str = "Quadratuc"
    element_shape: str = "Quad"
    integration: str = "full"

    # I/O Paths
    inp_path: str = ""
    custom_vtk_prefix: Optional[str] = None
    vtk_filename: Optional[str] = None

    # Material Properties
    E: Optional[float] = 2.4
    nu: Optional[float] = 0.4999
    alpha: Optional[float] = 0.0
    T0: Optional[float] = 0.0
    T1: Optional[float] = 0.0
    k_scalar: Optional[float] = 1.0
    D_scalar: Optional[float] = 1.0
    thickness: float = 1.0

    # Load Settings
    load: str = "distributed"
    distributed_type: str = "uniform"
    boundary_force: str = "O"
    body_force: str = "X"
    body_force_vec_2d: Optional[Tuple[float, float]] = None
    body_force_vec_3d: Optional[Tuple[float, float, float]] = None

    # Solver Settings (Nonlinear)
    nonlinear_mode: str = "native"
    nsteps: int = 1
    atol: float = 1e-8
    rtol: float = 1e-8
    stol: float = 1e-8
    max_iter: int = 60
    adaptive: bool = False

    # BC Containers (Elasticity)
    dirichlet_list_3d: List[NodeTuple] = field(default_factory=list)
    displacement_list_3d: List[NodeTuple] = field(default_factory=list)
    dirichlet_sets_3d: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dirichlet_txt_groups: Optional[List[Tuple]] = None

    traction_list_3d: List[Dict[str, Any]] = field(default_factory=list)
    traction_sets_3d: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    traction_txt_groups: Optional[List[Tuple]] = None

    concentrated_force_list_3d: List[Dict[str, Any]] = field(default_factory=list)
    concentrated_force_txt_groups: Optional[List[Tuple]] = None

    # BC Containers (Heat/Diffusion - kept for compatibility)
    heat_dirichlet_list_3d: List[Tuple[int, float]] = field(default_factory=list)
    heat_flux_list_3d: List[Dict[str, Any]] = field(default_factory=list)
    heat_source_list_3d: List[Dict[str, Any]] = field(default_factory=list)
    heat_dirichlet_txt_groups: Optional[List[Tuple]] = None
    heat_flux_txt_groups: Optional[List[Tuple]] = None
    heat_source_elements_txt_groups: Optional[List[Tuple]] = None
    
    diff_dirichlet_list_3d: List[Tuple[int, float]] = field(default_factory=list)
    diff_flux_list_3d: List[Dict[str, Any]] = field(default_factory=list)
    diff_source_list_3d: List[Dict[str, Any]] = field(default_factory=list)
    diff_dirichlet_txt_groups: Optional[List[Tuple]] = None
    diff_flux_txt_groups: Optional[List[Tuple]] = None
    diff_source_elements_txt_groups: Optional[List[Tuple]] = None
    
    # Placeholder Sets (Heat/Diffusion)
    heat_dirichlet_sets_3d: Dict = field(default_factory=dict)
    heat_flux_sets_3d: Dict = field(default_factory=dict)
    heat_source_sets_3d: Dict = field(default_factory=dict)
    diff_dirichlet_sets_3d: Dict = field(default_factory=dict)
    diff_flux_sets_3d: Dict = field(default_factory=dict)
    diff_source_sets_3d: Dict = field(default_factory=dict)

    # Post-Init Automation

    # Post-Init Automation
    def __post_init__(self):
        """Expand TXT-based groups into explicit lists for all physics."""

        # --------------------
        # 1) Elasticity: Dirichlet (Fixed vs Prescribed 자동 분류)
        # --------------------
        if self.dirichlet_txt_groups:
            gen_fixed = []
            gen_disp = []
            
            for p, ux, uy, uz in self.dirichlet_txt_groups:
                # 파일에서 노드 로드
                nodes = load_nodes_txt(p)
                items = build_dirichlet_list(nodes, ux, uy, uz)
                
                # 입력값 중 None이 아닌 값들을 추출
                vals = [v for v in (ux, uy, uz) if v is not None]
                
                # [핵심] 값이 하나라도 있고, 그 절대값이 0(허용오차)보다 크면 강제 변위로 분류
                is_prescribed = any(abs(v) > 1e-14 for v in vals)
                
                if is_prescribed:
                    gen_disp.extend(items)
                else:
                    gen_fixed.extend(items)
            
            # 각각의 리스트에 병합
            self.dirichlet_list_3d = merge_dirichlet_by_node(self.dirichlet_list_3d, gen_fixed)
            self.displacement_list_3d = merge_dirichlet_by_node(self.displacement_list_3d, gen_disp)

        # --------------------
        # 2) Elasticity: Traction (force / pressure)
        # --------------------
        if self.traction_txt_groups:
            for t in self.traction_txt_groups:
                nodes = load_nodes_txt(t[0])
                # (path, tx, ty, tz) or (path, pressure)
                if len(t) >= 4:
                    self.traction_list_3d.append({
                        "nodes": nodes,
                        "traction": [float(t[1]), float(t[2]), float(t[3])],
                    })
                else:
                    self.traction_list_3d.append({
                        "nodes": nodes,
                        "pressure": float(t[1]),
                    })

        # --------------------
        # 3) Elasticity: Concentrated nodal forces
        # --------------------
        if self.concentrated_force_txt_groups:
            for p, fx, fy, fz in self.concentrated_force_txt_groups:
                self.concentrated_force_list_3d.append({
                    "nodes": load_nodes_txt(p),
                    "F": [float(fx), float(fy), float(fz)],
                })

        # 4) Heat transfer: Dirichlet(T)
        if self.heat_dirichlet_txt_groups:
            genT = []
            for item in self.heat_dirichlet_txt_groups:
                if len(item) >= 2:
                    p = item[0]
                    Tval = float(item[1])
                    nodes = load_nodes_txt(p)
                    genT.extend(build_scalar_dirichlet_list(nodes, Tval))
            self.heat_dirichlet_list_3d = (self.heat_dirichlet_list_3d or []) + genT

        # 5) Heat transfer: Neumann (heat flux qn)
        if self.heat_flux_txt_groups:
            for item in self.heat_flux_txt_groups:
                # 포맷: (path, qn)
                if len(item) >= 2:
                    p = item[0]
                    qn = float(item[1])
                    nodes = load_nodes_txt(p)
                    # strict-list 포맷: {"nodes":[...], "qn": value}
                    self.heat_flux_list_3d.append({
                        "nodes": nodes,
                        "qn": qn,
                    })

        # --------------------
        # 6) Heat transfer: volumetric source from TXT (optional)
        # --------------------
        if self.heat_source_elements_txt_groups:
            for item in self.heat_source_elements_txt_groups:
                # 포맷 예시: (path, qv)
                if len(item) >= 2:
                    p = item[0]
                    qv = float(item[1])
                    elems = load_nodes_txt(p)  # element id 리스트를 재사용
                    self.heat_source_list_3d.append({
                        "elems": elems,
                        "qv": qv,
                    })

        # --------------------
        # 7) Diffusion: Dirichlet (concentration)
        # --------------------
        if self.diff_dirichlet_txt_groups:
            genC = []
            for item in self.diff_dirichlet_txt_groups:
                # 포맷: (path, C)
                if len(item) >= 2:
                    p = item[0]
                    Cval = float(item[1])
                    nodes = load_nodes_txt(p)
                    # (node_label, C)
                    genC.extend(build_scalar_dirichlet_list(nodes, Cval))
            self.diff_dirichlet_list_3d = (self.diff_dirichlet_list_3d or []) + genC

        # --------------------
        # 8) Diffusion: Neumann (mass flux)
        # --------------------
        if self.diff_flux_txt_groups:
            for item in self.diff_flux_txt_groups:
                # 포맷: (path, jn)
                if len(item) >= 2:
                    p = item[0]
                    jn = float(item[1])
                    nodes = load_nodes_txt(p)
                    self.diff_flux_list_3d.append({
                        "nodes": nodes,
                        "jn": jn,
                    })

        # --------------------
        # 9) Diffusion: volumetric source from TXT (optional)
        # --------------------
        if self.diff_source_elements_txt_groups:
            for item in self.diff_source_elements_txt_groups:
                # 포맷: (path, sv)
                if len(item) >= 2:
                    p = item[0]
                    sv = float(item[1])
                    elems = load_nodes_txt(p)
                    self.diff_source_list_3d.append({
                        "elems": elems,
                        "sv": sv,
                    })

        # --------------------
        # 10) Expand *set*-based definitions from TXT (faces / elements)
        # --------------------
        f_arity = 3 if str(self.element_shape).lower().startswith("tet") or str(self.element_shape).lower().startswith("tri") else 4
        self.dirichlet_sets_3d = _expand_sets_from_txt(self.dirichlet_sets_3d, face_arity=f_arity)
        self.traction_sets_3d  = _expand_sets_from_txt(self.traction_sets_3d,  face_arity=f_arity)
        self.heat_dirichlet_sets_3d = _expand_sets_from_txt(self.heat_dirichlet_sets_3d, face_arity=f_arity)
        self.heat_flux_sets_3d      = _expand_sets_from_txt(self.heat_flux_sets_3d,      face_arity=f_arity)
        self.heat_source_sets_3d    = _expand_sets_from_txt(self.heat_source_sets_3d,    face_arity=f_arity)
        self.diff_dirichlet_sets_3d = _expand_sets_from_txt(self.diff_dirichlet_sets_3d, face_arity=f_arity)
        self.diff_flux_sets_3d      = _expand_sets_from_txt(self.diff_flux_sets_3d,      face_arity=f_arity)
        self.diff_source_sets_3d    = _expand_sets_from_txt(self.diff_source_sets_3d,    face_arity=f_arity)

# =============================================================================
# USER SETTING BLOCK
# =============================================================================
setting = AnalysisConfig(
    # ---- core ----
    Dimension="3D",
    system="hyperelastic",
    analytical_conditions="3D",
    element_order="Linear",
    element_shape="Tet",  
    integration="full",

    # ---- input/output ----
    inp_path=r"D:\Changsub\연구실\GPU 연구 과제\열차폐코팅\FE inhouse code\nonlineaer fe in-house code\input file/C3D4_compressible_single_test.inp",
    custom_vtk_prefix=r"C3D4_compressible_single_test",
    vtk_filename="C3D4_compressible_single_testl.vtk",

    # ---- material (Abaqus Benchmark Exact Values) ----
    
    #C3D4 simple test: C10=0.4615, D1=1
    E= 2.4,
    nu = 0.3,
    
    #ASTM uniaxial test: C10=0.1352737791, D1= 7.39266206e-4
    #E  = 0.81161562,
    #nu = 0.49995,
    
    #xu paper 3d cooks membrane: labda=100, mu=40
    #E  = 108.5714286,
    #nu = 0.357142857,
    
    #2D incomressible cooks membrane: C10=0.4, D1=2.5e-4 => K=8000, mu=0.8
    #E=2.39994666, 
    #nu=0.4999666,
    
    #C3D8 simple test: C10=4, D1=0.15
    #E= 2.4,
    #nu = 0.49995,
    # ---- BC/loads ----
    dirichlet_txt_groups=[(r"D:\Changsub\연구실\GPU 연구 과제\열차폐코팅\FE inhouse code\nonlineaer fe in-house code\BC_txt/C3D4_compressible_single_test_dirichlet.txt", 0.0, 0.0, 0),
                          ],
    traction_txt_groups=[(r"D:\Changsub\연구실\GPU 연구 과제\열차폐코팅\FE inhouse code\nonlineaer fe in-house code\BC_txt/C3D4_compressible_single_test_neumann.txt", 0, 0.1, 0)],
    #body_force_vec_2d=(0.0, 0),
    body_force_vec_3d=(0.0, 0, 0),
    thickness=1.0,
)

#setting.bc_mode = "list"       
#setting.traction_mode = "list" 

# ---- Nonlinear Solver Options ----
setting.nonlinear_mode = "native"
setting.nsteps = 10    # Adaptive will cut this back if needed
setting.adaptive = True   # [CRITICAL] Enable adaptive time stepping
setting.atol = 1e-5
setting.rtol = 1e-5
setting.stol = 1e-8
setting.max_iter = 30