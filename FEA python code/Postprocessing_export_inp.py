# -*- coding: utf-8 -*-
# postprocessing.py — 입력/BC 라우팅 자동화 + diffusion 문제 해결판 (+BC 요약 로깅/CSV)
# made by changsub (refactor + logging by gpt)

from pathlib import Path
import numpy as np
import math
import inspect
from typing import Dict, List, Tuple, Any, Iterable, Optional, Set

from Preprocessing_export_inp import Material_Property, Mesh
from Element_formulation_export_inp import Shape_Function, Stiffness_Matrix, coerce_element_settings
from Assembly_and_solve_eq_export_inp import (
    Partition_Method, Linear_elasticity_BC, Heat_transfer_BC, Diffusion_BC
)
from make_vtk import (
    write_vtk_and_txt,
    write_vtk_and_txt_heat,
    write_vtk_and_txt_diffusion,
    write_vtk_and_txt_elasticity_3d,
    write_vtk_and_txt_heat_3d,
    write_vtk_and_txt_diffusion_3d
)

# === 모든 입력/BC는 main.py의 setting 사용 ===
from main import setting

# ==== 로깅 토글 ====
VERBOSE = True  # 콘솔 요약 출력/CSV 요약 내보내기 켜기

# === (중요) 경계면 필터링: 끔 ===
# 요청대로 내부면 여부와 무관하게 TXT로 지정된 노드가 포함된 페이스면 전부 적분합니다.
BOUNDARY_ONLY = False

# --- 백워드 호환 (메서드 명칭/시그니처 패치) ---
# Heat: 메서드 이름 보정
if not hasattr(Heat_transfer_BC, "boundary_heat_flux") and hasattr(Heat_transfer_BC, "assemble_boundary_flux"):
    Heat_transfer_BC.boundary_heat_flux = Heat_transfer_BC.assemble_boundary_flux
if not hasattr(Heat_transfer_BC, "Boundary_heat_flux") and hasattr(Heat_transfer_BC, "assemble_boundary_flux"):
    Heat_transfer_BC.Boundary_heat_flux = Heat_transfer_BC.assemble_boundary_flux
if not hasattr(Heat_transfer_BC, "heat_source") and hasattr(Heat_transfer_BC, "Heat_source"):
    Heat_transfer_BC.heat_source = Heat_transfer_BC.Heat_source

# Diffusion: 메서드 이름 보정
if not hasattr(Diffusion_BC, "Boundary_concentration_flux") and hasattr(Diffusion_BC, "assemble_boundary_flux"):
    Diffusion_BC.Boundary_concentration_flux = Diffusion_BC.assemble_boundary_flux
if not hasattr(Diffusion_BC, "Body_concentration_flux") and hasattr(Diffusion_BC, "Body_source"):
    Diffusion_BC.Body_concentration_flux = Diffusion_BC.Body_source

# ---- (핵심) Dirichlet_BC(load) 시그니처 호환 어댑터 ----
def _patch_dirichlet_bc_load_shim():
    try:
        sig = inspect.signature(Heat_transfer_BC.Dirichlet_BC)
        need_load = ("load" in sig.parameters) and (sig.parameters["load"].default is inspect._empty)
    except Exception:
        need_load = False

    if need_load:
        _orig = Heat_transfer_BC.Dirichlet_BC

        def _shim(self, *args, **kwargs):
            if not args and "load" not in kwargs:
                kwargs["load"] = "heat flux" if getattr(setting, "boundary_heat_flux", "X") == "O" else getattr(setting, "load", "none")
            return _orig(self, *args, **kwargs)

        Heat_transfer_BC.Dirichlet_BC = _shim

    try:
        sigd = inspect.signature(Diffusion_BC.Dirichlet_BC)
        need_load_d = ("load" in sigd.parameters) and (sigd.parameters["load"].default is inspect._empty)
    except Exception:
        need_load_d = False

    if need_load_d:
        _orig_d = Diffusion_BC.Dirichlet_BC

        def _shim_d(self, *args, **kwargs):
            if not args and "load" not in kwargs:
                kwargs["load"] = "mass flux" if getattr(setting, "boundary_mass_concentration", "X") == "O" else getattr(setting, "load", "none")
            return _orig_d(self, *args, **kwargs)

        Diffusion_BC.Dirichlet_BC = _shim_d

_patch_dirichlet_bc_load_shim()


# ======================
# 유틸: range/리스트 확장기
# ======================
def _expand_range_like(obj):
    """ "1:10", "1-10", {"from":1,"to":10,"step":2}, [혼합] → 정수 리스트 """
    out = []
    if obj is None:
        return out
    if isinstance(obj, (list, tuple, set)):
        for it in obj:
            out.extend(_expand_range_like(it))
        return out
    if isinstance(obj, str):
        s = obj.strip()
        if ":" in s:
            a, b = s.split(":", 1); out.extend(range(int(a), int(b)+1))
        elif "-" in s:
            a, b = s.split("-", 1); out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(s))
        return out
    if isinstance(obj, dict):
        a = int(obj.get("from", 0)); b = int(obj.get("to", -1)); st = int(obj.get("step", 1))
        out.extend(range(a, b+1, st)); return out
    if isinstance(obj, int):
        return [obj]
    raise ValueError(f"지원하지 않는 range 형식: {obj!r}")


# ======================
# (신규) 요소 페이스 생성 유틸 (Tet/Hex)
# ======================
def _element_faces(conn: Iterable[int], shape: str) -> List[Tuple[int, ...]]:
    """
    요소의 로컬 노드 인덱스(conn: 0-based row 인덱스들)로부터
    '실제 페이스'의 로컬 인덱스 조합을 반환.
    - Tet(4): 4 faces (3 nodes)
    - Hex(8): 6 faces (4 nodes) — Abaqus C3D8 표준 순서 기준
    """
    nodes = list(conn)
    shp = (shape or "").lower()
    if shp.startswith("tet"):
        if len(nodes) < 4: return []
        faces = [(nodes[i] , nodes[j], nodes[k]) for (i,j,k) in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))]
        return [tuple(f) for f in faces]
    if shp.startswith("hex"):
        if len(nodes) < 8: return []
        n = nodes
        faces = [(n[0],n[1],n[2],n[3]),
                 (n[4],n[5],n[6],n[7]),
                 (n[0],n[1],n[5],n[4]),
                 (n[2],n[3],n[7],n[6]),
                 (n[0],n[3],n[7],n[4]),
                 (n[1],n[2],n[6],n[5])]
        return [tuple(f) for f in faces]
    return []


def _rowidx_to_label(mesh: Mesh, ridx: int) -> int:
    return int(getattr(mesh, "row2id", {}).get(ridx, ridx + 1))


def _face_row_to_labels(mesh: Mesh, face_rows: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(_rowidx_to_label(mesh, r) for r in face_rows)


def _expand_nodes_to_seedfaces(
    mesh: Mesh,
    element_shape: str,
    nodes_label_list: Iterable[int],
    value_key: str,
    value: Any
) -> List[Dict[str, Any]]:
    """
    nodes(노드 라벨 모음) → 요소의 '모든 페이스' 중
    그 페이스의 모든 라벨이 nodes에 '포함'되면 seed_nodes로 채택.

    - 경계/내부 구분 없음(BOUNDARY_ONLY=False)
    - 중복 제거: 같은 요소에서 동일 면이 중복될 일은 없으므로 '완전 동일 튜플'만 막음
      (이웃요소 반대방향은 다른 순서가 되어 중복으로 간주하지 않음 → 그대로 두 번 적분 가능)
    - value_key == "pressure": value는 스칼라
      value_key == "traction": value는 [tx,ty,tz] (리스트/튜플/ndarray)
    """
    nodes_set: Set[int] = set(int(x) for x in nodes_label_list)
    faces_out: List[Dict[str, Any]] = []
    seen_exact: Set[Tuple[int, ...]] = set()

    for e_conn in mesh.conn:
        faces = _element_faces(e_conn, element_shape)
        for fr in faces:
            flabels = _face_row_to_labels(mesh, fr)   # 라벨 튜플(면 순서 유지)
            if all((lab in nodes_set) for lab in flabels):
                if flabels in seen_exact:
                    continue
                seen_exact.add(flabels)

                # 값 포맷 보정(스칼라 or 벡터 유지)
                if value_key == "traction":
                    if isinstance(value, (list, tuple, np.ndarray)):
                        v_out = [float(v) for v in value]
                    else:
                        raise TypeError("traction 값은 [tx,ty,tz] 형식이어야 합니다.")
                else:
                    v_out = float(value)

                faces_out.append({"seed_nodes": list(flabels), value_key: v_out})

    return faces_out


def _expand_all_flux_like_specs_to_faces(
    mesh: Mesh,
    element_shape: str,
    lst: Optional[List[Dict[str, Any]]],
    *,
    value_key: str
) -> List[Dict[str, Any]]:
    """
    list 안의 각 사양에 대해:
      - 이미 seed_nodes가 있으면 그대로 보존
      - nodes 가 있으면 nodes → seed_faces 로 확장하여 추가
    (주의) 이 유틸은 heat/diff의 스칼라 값(qn/jn) 전용으로 사용
    """
    if not isinstance(lst, list):
        return []

    out: List[Dict[str, Any]] = []
    for spec in lst:
        if not isinstance(spec, dict):
            continue
        spec = dict(spec)  # copy
        # seed_nodes 기반: 그대로 둠
        if "seed_nodes" in spec and isinstance(spec["seed_nodes"], (list, tuple)):
            out.append(spec)
        # nodes 기반: 확장
        if "nodes" in spec and isinstance(spec["nodes"], (list, tuple)):
            if value_key not in spec:
                continue
            nodes_labels = [int(x) for x in spec["nodes"]]
            value = float(spec[value_key])  # heat/diff는 스칼라
            faces = _expand_nodes_to_seedfaces(mesh, element_shape, nodes_labels, value_key, value)
            out.extend(faces)
    return out


# ======================
# heat/diffusion 설정 확장
# ======================
def _expand_heat_cfgs():
    cfg = {}

    cfg["dirichlet"] = {
        "mode": getattr(setting, "heat_dirichlet_mode", "off"),
        "list": getattr(setting, "heat_dirichlet_list_3d", None),
        "sets": getattr(setting, "heat_dirichlet_sets_3d", None),
    }

    # Neumann (qn)
    sets_qn = getattr(setting, "heat_flux_sets_3d", None)
    if isinstance(sets_qn, dict):
        sets_qn = {k: dict(v) for k, v in sets_qn.items()}
        for _, v in sets_qn.items():
            if "nodes" in v: v["nodes"] = _expand_range_like(v["nodes"])
            if "faces" in v: v["faces"] = _expand_range_like(v["faces"])
            if "seed_nodes" in v: v["seed_nodes"] = _expand_range_like(v["seed_nodes"])
    cfg["flux"] = {
        "mode": getattr(setting, "heat_flux_mode", "off"),
        "list": getattr(setting, "heat_flux_list_3d", None),
        "sets": sets_qn,
    }

    # Source (qv)
    sets_qv = getattr(setting, "heat_source_sets_3d", None)
    if isinstance(sets_qv, dict):
        sets_qv = {k: dict(v) for k, v in sets_qv.items()}
        for _, v in sets_qv.items():
            if "elements" in v: v["elements"] = _expand_range_like(v["elements"])
            if "seed_nodes" in v: v["seed_nodes"] = _expand_range_like(v["seed_nodes"])
    cfg["source"] = {
        "mode": getattr(setting, "heat_source_mode", "off"),
        "list": getattr(setting, "heat_source_list_3d", None),
        "sets": sets_qv,
    }
    return cfg


def _expand_diff_cfgs():
    cfg = {}

    cfg["dirichlet"] = {
        "mode": getattr(setting, "diff_dirichlet_mode", "off"),
        "list": getattr(setting, "diff_dirichlet_list_3d", None),
        "sets": getattr(setting, "diff_dirichlet_sets_3d", None),
    }

    sets_flux = getattr(setting, "diff_flux_sets_3d", None)
    if isinstance(sets_flux, dict):
        sets_flux = {k: dict(v) for k, v in sets_flux.items()}
        for _, v in sets_flux.items():
            if "nodes" in v: v["nodes"] = _expand_range_like(v["nodes"])
            if "faces" in v: v["faces"] = _expand_range_like(v["faces"])
            if "seed_nodes" in v: v["seed_nodes"] = _expand_range_like(v["seed_nodes"])
    cfg["flux"] = {
        "mode": getattr(setting, "diff_flux_mode", "off"),
        "list": getattr(setting, "diff_flux_list_3d", None),
        "sets": sets_flux,
    }

    sets_src = getattr(setting, "diff_source_sets_3d", None)
    if isinstance(sets_src, dict):
        sets_src = {k: dict(v) for k, v in sets_src.items()}
        for _, v in sets_src.items():
            if "elements" in v: v["elements"] = _expand_range_like(v["elements"])
            if "seed_nodes" in v: v["seed_nodes"] = _expand_range_like(v["seed_nodes"])
            if "sy" in v and "sv" not in v:
                v["sv"] = v.pop("sy")
    cfg["source"] = {
        "mode": getattr(setting, "diff_source_mode", "off"),
        "list": getattr(setting, "diff_source_list_3d", None),
        "sets": sets_src,
    }
    return cfg


# ======================
# 라우터: system에 맞는 인자만 전달
# ======================
def _route_bc_kwargs(system, setting, heat_cfg, diff_cfg):
    """선택 system에 필요한 슬롯만 채워서 Partition_Method로 전달."""
    if system == "linear elasticity":
        return dict(
            load = setting.load,  # "concentrated" | "distributed" | "displacement"
            boundary_force = setting.boundary_force,
            body_force     = setting.body_force,
            T0 = setting.T0, T1 = setting.T1, alpha = setting.alpha, E = setting.E, nu = setting.nu,
            k_scalar = None, D_scalar = None,
            boundary_heat_flux = None, source_heat_flux = None,
            boundary_mass_concentration = None, body_mass_concentration = None,
            # elasticity 슬롯
            dirichlet_list_3d = setting.dirichlet_list_3d if setting.bc_mode in ("list","both") else None,
            dirichlet_sets_3d = setting.dirichlet_sets_3d if setting.bc_mode in ("set","both") else None,
            traction_mode     = setting.traction_mode,
            traction_list_3d  = setting.traction_list_3d if setting.traction_mode in ("list","both") else None,
            traction_sets_3d  = setting.traction_sets_3d  if setting.traction_mode in ("set","both")  else None,
            # 집중하중 전달
            concentrated_force_list_3d = getattr(setting, "concentrated_force_list_3d", None),
        )

    if system == "heat transfer":
        heat_load = "heat flux" if getattr(setting, "boundary_heat_flux", "X") == "O" else "none"
        return dict(
            load = heat_load,
            boundary_force = None, body_force = None,
            T0 = setting.T0, T1 = setting.T1, alpha = setting.alpha, E = None, nu = None,
            k_scalar = setting.k_scalar, D_scalar = None,
            boundary_heat_flux = setting.boundary_heat_flux,   # "O"/"X"
            source_heat_flux   = setting.source_heat_flux,     # "O"/"X"
            boundary_mass_concentration = None, body_mass_concentration = None,
            # heat 슬롯
            heat_dirichlet_mode    = heat_cfg["dirichlet"]["mode"],
            heat_dirichlet_list_3d = heat_cfg["dirichlet"]["list"],
            heat_dirichlet_sets_3d = heat_cfg["dirichlet"]["sets"],
            heat_flux_mode         = heat_cfg["flux"]["mode"],
            heat_flux_list_3d      = heat_cfg["flux"]["list"],
            heat_flux_sets_3d      = heat_cfg["flux"]["sets"],
            heat_source_mode       = heat_cfg["source"]["mode"],
            heat_source_list_3d    = heat_cfg["source"]["list"],
            heat_source_sets_3d    = heat_cfg["source"]["sets"],
            # elasticity 슬롯 비움
            dirichlet_list_3d=None, dirichlet_sets_3d=None,
            traction_mode="off", traction_list_3d=None, traction_sets_3d=None,
        )

    if system == "diffusion":
        diff_load = "mass flux" if getattr(setting, "boundary_mass_concentration", "X") == "O" else "none"
        return dict(
            load = diff_load,
            boundary_force = None, body_force = None,
            T0 = None, T1 = None, alpha = None, E = None, nu = None,
            k_scalar = None, D_scalar = setting.D_scalar,
            boundary_heat_flux = None, source_heat_flux = None,
            boundary_mass_concentration = setting.boundary_mass_concentration,  # "O"/"X"
            body_mass_concentration     = setting.body_mass_concentration,      # "O"/"X"
            # diffusion 슬롯
            diff_dirichlet_mode    = diff_cfg["dirichlet"]["mode"],
            diff_dirichlet_list_3d = diff_cfg["dirichlet"]["list"],
            diff_dirichlet_sets_3d = diff_cfg["dirichlet"]["sets"],
            diff_flux_mode         = diff_cfg["flux"]["mode"],
            diff_flux_list_3d      = diff_cfg["flux"]["list"],
            diff_flux_sets_3d      = diff_cfg["flux"]["sets"],
            diff_source_mode       = diff_cfg["source"]["mode"],
            diff_source_list_3d    = diff_cfg["source"]["list"],
            diff_source_sets_3d    = diff_cfg["source"]["sets"],
            # elasticity/heat 슬롯 비움
            dirichlet_list_3d=None, dirichlet_sets_3d=None,
            traction_mode="off", traction_list_3d=None, traction_sets_3d=None,
        )

    raise ValueError(f"Unknown system: {system}")


# ======================
# 0) 입력/조건
# ======================
Dimension = setting.Dimension
system = setting.system
element_order = setting.element_order
element_shape = setting.element_shape
integration = setting.integration
analytical_conditions = setting.analytical_conditions
inp_path = setting.inp_path
custom_vtk_prefix = setting.custom_vtk_prefix

# 물성
E, nu, alpha = setting.E, setting.nu, setting.alpha
T0, T1 = setting.T0, setting.T1
k_scalar = setting.k_scalar
D_scalar = setting.D_scalar

# 하중/분포
load = setting.load
distributed_type = setting.distributed_type

# ======================
# 1) 메쉬 로드
# ======================
def map_user_to_etype_prefix(Dimension, element_shape, element_order, analytical_conditions):
    Dimension = Dimension.upper()
    shp = element_shape.capitalize()
    ord_ = element_order.capitalize()
    cond = (analytical_conditions or "").lower()

    if Dimension == "2D":
        if shp == "Quad":
            if ord_ == "Linear":
                return "CPS4" if cond == "plane stress" else "CPE4"
            elif ord_ == "Quadratic":
                return "CPS8" if cond == "plane stress" else "CPE8"
        elif shp == "Tri":
            if ord_ == "Linear":
                return "CPS3" if cond == "plane stress" else "CPE3"
            elif ord_ == "Quadratic":
                return "CPS6" if cond == "plane stress" else "CPE6"
    elif Dimension == "3D":
        if shp == "Hex":
            return "C3D8" if ord_ == "Linear" else "C3D20"
        elif shp == "Tet":
            return "C3D4" if ord_ == "Linear" else "C3D10"
    return None

prefer_type = map_user_to_etype_prefix(Dimension, element_shape, element_order, analytical_conditions)
mesh = Mesh.from_inp(inp_path, prefer_type=prefer_type, debug=True)

# ---- 1-based 라벨 헬퍼 (노드/요소/face번호) ----
def _node_label(row_idx: int) -> int:
    return int(getattr(mesh, "row2id", {}).get(row_idx, row_idx + 1))

def _elem_label(eidx: int) -> int:
    elem_ids = getattr(mesh, "elem_ids", None)
    if isinstance(elem_ids, (list, tuple, np.ndarray)) and 0 <= eidx < len(elem_ids):
        try:
            return int(elem_ids[eidx])
        except Exception:
            pass
    return int(eidx + 1)

def _face_id_1based(fidx_zero_based: int) -> int:
    return int(fidx_zero_based + 1)

# ---- Material matrix 준비 (탄성만) ----
# ---- Material matrix 준비 (탄성만) ----
# 여기서는 "해석 설정"에 따른 차원/조건을 기준으로 재료 행렬을 만든다.
# INP에서 읽은 Dimension/요소 정보는 메쉬 일관성 체크용으로만 사용.
problem = Material_Property(system, Dimension if Dimension is not None else mesh.Dimension)
if system == "linear elasticity":
    dim_for_D = Dimension if Dimension is not None else mesh.Dimension
    if dim_for_D == "2D":
        D_matrix = problem.Hookean_matrix_2D(E, nu, analytical_conditions)
    else:
        D_matrix = problem.Hookean_matrix_3D(E, nu, "3D")
    print("D matrix: ", D_matrix)

# INP 요소설정과 입력이 다르면 경고만 출력하고, 사용자 설정은 유지
if (mesh.Dimension != Dimension) or (mesh.element_shape != element_shape) or (mesh.element_order != element_order):
    print("[WARN] 입력 요소설정과 INP가 다릅니다. INP 정보는 참고만 하고, 사용자 설정을 유지합니다.")
    print(f"  입력: Dim={Dimension}, Shape={element_shape}, Order={element_order}")
    print(f"  INP : Dim={mesh.Dimension}, Shape={mesh.element_shape}, Order={mesh.element_order}, etype={getattr(mesh,'inp_element_type',None)}")


# ======================
# 2) 형상함수/적분점
# ======================
Dimension, element_shape, element_order = coerce_element_settings(mesh, Dimension, element_shape, element_order)
shape_function = Shape_Function(analytical_conditions, element_order, element_shape, integration)
shape_function.calculate_N_dN(Dimension)

def _get_gp_weights(sf):
    try:
        W = np.asarray(sf.W, dtype=float)
        if W.shape[0] == len(sf.gp):
            return W
    except Exception:
        pass
    return np.ones(len(sf.gp), dtype=float)

W_gp = _get_gp_weights(shape_function)

# ========= Abaqus식 centroid 외삽 =========
def _extrap_centroid_from_gp(vals_gp, gp_list):
    ngp, m = vals_gp.shape
    dim = len(gp_list[0])
    if ngp == 1:
        return vals_gp[0, :]
    idx_sorted = sorted(range(ngp), key=lambda i: tuple(gp_list[i]))
    G = vals_gp[idx_sorted, :]
    s = abs(gp_list[idx_sorted[0]][0])
    if not (np.isclose(s, 1.0/np.sqrt(3.0), rtol=1e-6, atol=1e-8)):
        return G.mean(axis=0)
    r = 1.0 / s
    E1 = 0.5 * np.array([[1.0 + r, 1.0 - r],
                         [1.0 - r, 1.0 + r]], dtype=float)
    if dim == 2 and ngp == 4:
        E = np.kron(E1, E1)
        nodal = E @ G
        return nodal.mean(axis=0)
    elif dim == 3 and ngp == 8:
        E = np.kron(E1, np.kron(E1, E1))
        nodal = E @ G
        return nodal.mean(axis=0)
    else:
        return G.mean(axis=0)

# ======================
# 3) Global K 조립
# ======================
K_matrix = Stiffness_Matrix(
    E, nu,
    k_scalar, D_scalar,
    analytical_conditions, system, Dimension,
    element_order, element_shape, integration,
    mesh
)
K_global = K_matrix.global_K()

# ======================
# 4) 분포 적분점 수 힌트(기존 로직 유지)
# ======================
if element_order == "Linear" and (distributed_type in ("uniform", None)):
    n_gp = 1
elif element_order == "Linear" and (distributed_type in ("user_defined", None)):
    n_gp = 2
elif element_order == "Quadratic" and (distributed_type in ("uniform", None)) and element_shape == "Tri":
    n_gp = 2
elif element_order == "Quadratic" and (distributed_type in ("uniform", None)) and element_shape == "Quad":
    n_gp = 2
elif element_order == "Quadratic" and (distributed_type in ("uniform", None)) and element_shape == "Hex":
    n_gp = 3
elif element_order == "Quadratic" and (distributed_type in ("uniform", None)) and element_shape == "Tet":
    n_gp = None
elif element_order == "Quadratic" and (distributed_type in ("user_defined", None)):
    n_gp = 2
else:
    n_gp = 2

# ======================
# 5) 설정 확장 + (신규) nodes→faces 확장
# ======================
_heat_cfg = _expand_heat_cfgs()
_diff_cfg = _expand_diff_cfgs()

def _len_safe(x):
    if isinstance(x, list): return len(x)
    if isinstance(x, dict): return len(x)
    return 0

# --- (신규) 모든 물리현상에 대해 nodes→seed_faces 확장 (경계 필터링 없음)
# Heat (qn)
if system == "heat transfer" and Dimension == "3D" and _heat_cfg["flux"]["list"]:
    expanded = _expand_all_flux_like_specs_to_faces(
        mesh, element_shape, _heat_cfg["flux"]["list"], value_key="qn"
    )
    _heat_cfg["flux"]["list"] = expanded

# Diffusion (jn) — 3D Hex/Tet에서만 사용
if system == "diffusion" and Dimension == "3D" and _diff_cfg["flux"]["list"]:
    expanded = _expand_all_flux_like_specs_to_faces(
        mesh, element_shape, _diff_cfg["flux"]["list"], value_key="jn"
    )
    _diff_cfg["flux"]["list"] = expanded

# Elastic Traction (pressure/traction) — 벡터도 지원
# ---- Elasticity: Neumann (traction/pressure) ----
# 3D 문제에서만 nodes -> seed_faces 변환
if system == "linear elasticity" and Dimension == "3D":
    tr_list = getattr(setting, "traction_list_3d", None)
    if tr_list:
        out_tr = []
        for spec in tr_list:
            # 사용자가 이미 seed_nodes로 준 경우는 그대로 사용
            if "seed_nodes" in spec:
                out_tr.append(spec)
                continue

            # nodes + traction / pressure 를 seed_faces로 변환
            if "nodes" in spec:
                nodes = spec["nodes"]
                if "traction" in spec:
                    faces = _expand_nodes_to_seedfaces(
                        mesh, element_shape, nodes,
                        "traction", spec["traction"]
                    )
                    out_tr.extend(faces)
                elif "pressure" in spec:
                    faces = _expand_nodes_to_seedfaces(
                        mesh, element_shape, nodes,
                        "pressure", spec["pressure"]
                    )
                    out_tr.extend(faces)

        setting.traction_list_3d = out_tr


# --- (A) 설정 요약 콘솔 출력 ---
def _print_bc_config_summary(system, mesh, setting, heat_cfg, diff_cfg):
    if not VERBOSE:
        return

    def _len(x):
        if isinstance(x, (list, tuple, set, dict, np.ndarray)): return len(x)
        return 0

    def _summ_set_dict(title, d):
        if not isinstance(d, dict) or not d:
            print(f"[BC] {title}: (no sets)")
            return
        print(f"[BC] {title}: {len(d)} set(s)")
        for name, spec in d.items():
            faces = _len(spec.get("faces"))
            nodes = _len(spec.get("nodes"))
            elems = _len(spec.get("elements"))
            seeds = _len(spec.get("seed_nodes"))
            val_keys = [k for k in ("pressure", "traction", "qn", "qv", "sv", "jn", "value", "T", "C", "node_force") if k in spec]
            vals = ", ".join([f"{k}={spec[k]}" for k in val_keys])
            extra = []
            if "ang_tol_deg" in spec: extra.append(f"ang_tol={spec['ang_tol_deg']}°")
            if "dilate_deg" in spec:  extra.append(f"dilate={spec['dilate_deg']}°")
            if "relax_first_ring" in spec: extra.append(f"relax1st={spec['relax_first_ring']}")
            extra_s = (", " + ", ".join(extra)) if extra else ""
            print(f"  - set='{name}': faces={faces}, nodes={nodes}, elements={elems}, seed_nodes={seeds}"
                  f"{extra_s}{(', ' + vals) if vals else ''}")

    print("\n========== [BC CONFIG SUMMARY] ==========")
    print(f"Dim={mesh.Dimension}, System={system}, Shape={mesh.element_shape}, Order={mesh.element_order}")

    if system == "linear elasticity":
        mode = getattr(setting, "bc_mode", "off")
        print(f"[Dirichlet] mode={mode}")
        if mode in ("list", "both"):
            dl = getattr(setting, "dirichlet_list_3d", None)
            print(f"  list count={_len(dl)}")
        if mode in ("set", "both"):
            ds = getattr(setting, "dirichlet_sets_3d", None)
            _summ_set_dict("Dirichlet sets", ds)

        tmode = getattr(setting, "traction_mode", "off")
        print(f"[Neumann/Traction] mode={tmode}")
        if tmode in ("list", "both"):
            tl = getattr(setting, "traction_list_3d", None)
            print(f"  list count={_len(tl)}")
        if tmode in ("set", "both"):
            ts = getattr(setting, "traction_sets_3d", None)
            _summ_set_dict("Traction sets", ts)

        # ▼ 추가: Concentrated nodal forces 요약
        conc = getattr(setting, "concentrated_force_list_3d", None)
        if conc:
            tot = np.zeros(3)
            cnt = 0
            for spec in conc:
                if not isinstance(spec, dict): continue
                import numpy as _np
                F = _np.asarray(spec.get("F", [0,0,0]), float).reshape(-1,)
                if F.size == 2: F = _np.array([F[0], F[1], 0.0])
                if "nodes" in spec and spec["nodes"]:
                    n = len(spec["nodes"]); tot[:3] += n * F[:3]; cnt += n
                elif "node" in spec:
                    tot[:3] += F[:3]; cnt += 1
            print(f"[Concentrated] items={len(conc)}, applied_nodes={cnt}, total≈{tot.tolist()}")
        else:
            print("[Concentrated] (none)")

    if system == "heat transfer":
        print(f"[Dirichlet(T)] mode={heat_cfg['dirichlet']['mode']}, list={_len(heat_cfg['dirichlet']['list'])}")
        _summ_set_dict("Dirichlet(T) sets", heat_cfg['dirichlet']['sets'])
        print(f"[Neumann(qn)] mode={heat_cfg['flux']['mode']}, list={_len(heat_cfg['flux']['list'])}")
        _summ_set_dict("Heat flux sets", heat_cfg['flux']['sets'])
        print(f"[Source(qv)] mode={heat_cfg['source']['mode']}, list={_len(heat_cfg['source']['list'])}")
        _summ_set_dict("Heat source sets", heat_cfg['source']['sets'])

    if system == "diffusion":
        print(f"[Dirichlet(C)] mode={diff_cfg['dirichlet']['mode']}, list={_len(diff_cfg['dirichlet']['list'])}")
        _summ_set_dict("Dirichlet(C) sets", diff_cfg['dirichlet']['sets'])
        print(f"[Neumann(J·n)] mode={diff_cfg['flux']['mode']}, list={_len(diff_cfg['flux']['list'])}")
        _summ_set_dict("Diffusion flux sets", diff_cfg['flux']['sets'])
        print(f"[Source(sv)] mode={diff_cfg['source']['mode']}, list={_len(diff_cfg['source']['list'])}")
        _summ_set_dict("Diffusion source sets", diff_cfg['source']['sets'])
    print("=========================================\n")


_print_bc_config_summary(system, mesh, setting, _heat_cfg, _diff_cfg)

kwargs = _route_bc_kwargs(system, setting, _heat_cfg, _diff_cfg)

# ======================
# 6) Partition & Solve
# ======================
divide_matrix = Partition_Method(
    K_global, system, Dimension, element_shape, element_order,
    analytical_conditions, integration,
    kwargs.pop("load"), n_gp, distributed_type,
    mesh=mesh,
    **kwargs
)
BC = divide_matrix.bc

# --- (NEW) 경계 face CSV 디버그 덤프 ---
try:
    Path("bc_reports").mkdir(parents=True, exist_ok=True)
    # 통합: 현재 list에 실려있는 face 형태를 그대로 덤프
    def _dump_face_list_csv(list_items: Optional[List[Dict[str, Any]]], path: str, value_key: str):
        if not list_items: 
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            hdr = ["idx", "seed_nodes", value_key]
            w.writerow(hdr)
            for i, spec in enumerate(list_items, 1):
                if "seed_nodes" in spec and value_key in spec:
                    w.writerow([i, " ".join(str(int(x)) for x in spec["seed_nodes"]), spec[value_key]])

    # ▼ 탄성: pressure/traction 모두 저장
    def _dump_elastic_faces(list_items: Optional[List[Dict[str, Any]]], path: str):
        if not list_items:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "seed_nodes", "pressure", "tx", "ty", "tz"])
            for i, spec in enumerate(list_items, 1):
                if "seed_nodes" not in spec:
                    continue
                sn = " ".join(str(int(x)) for x in spec.get("seed_nodes", []))
                p  = spec.get("pressure")
                tv = spec.get("traction")
                if isinstance(tv, (list, tuple, np.ndarray)) and len(tv) >= 3:
                    tx, ty, tz = [float(tv[0]), float(tv[1]), float(tv[2])]
                else:
                    tx = ty = tz = None
                w.writerow([i, sn, p, tx, ty, tz])

    if system == "heat transfer":
        _dump_face_list_csv(_heat_cfg["flux"]["list"], "bc_reports/heat_flux_list_faces.csv", "qn")
    elif system == "diffusion":
        _dump_face_list_csv(_diff_cfg["flux"]["list"], "bc_reports/diff_flux_list_faces.csv", "jn")
    elif system == "linear elasticity":
        _dump_elastic_faces(getattr(setting, "traction_list_3d", None), "bc_reports/traction_list_faces.csv")
except Exception as e:
    print(f"[BC CSV] 내보내기 중 예외: {e}")

# --- (B) 실제 적용 결과 요약/CSV 내보내기 ---
def _print_effective_bc_counts(system, mesh, divide_matrix, BC, out_dir: Path, prefix: str):
    if not VERBOSE:
        return
    # Dirichlet DoF 수
    n_dofs_total = (mesh.NoN * (3 if mesh.Dimension == "3D" and system == "linear elasticity" else
                                2 if mesh.Dimension == "2D" and system == "linear elasticity" else
                                1))
    n_DoEs = len(getattr(BC, "DoEs", []))
    n_DoFs = len(getattr(BC, "DoFs", []))
    F_E = np.asarray(getattr(divide_matrix, "F_E", np.array([]))).reshape(-1)
    F_F = np.asarray(getattr(divide_matrix, "F_F", np.array([]))).reshape(-1)
    nz_FE = int(np.sum(np.abs(F_E) > 0))
    nz_FF = int(np.sum(np.abs(F_F) > 0))
    print("---------- [BC EFFECTIVE COUNTS] ----------")
    print(f"Total DOFs = {n_dofs_total}, Dirichlet DoEs = {n_DoEs}, Free DoFs = {n_DoFs}")
    print(f"RHS: |F_E|>0 count = {nz_FE}, |F_F|>0 count = {nz_FF}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{prefix}_BC_EffectiveCounts.csv"
    with open(fp, "w", encoding="utf-8") as f:
        f.write("TotalDOFs,Dirichlet_DoEs,Free_DoFs,RHS_FE_nz,RHS_FF_nz\n")
        f.write(f"{n_dofs_total},{n_DoEs},{n_DoFs},{nz_FE},{nz_FF}\n")
    print(f"[WRITE] {fp}")
    print("-------------------------------------------\n")


def _export_bc_csv_summaries(system, mesh, setting, heat_cfg, diff_cfg, out_dir: Path, prefix: str):
    if not VERBOSE:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_rows(path, header, rows):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(r)
        print(f"[WRITE] {path}")

    if system == "linear elasticity":
        if setting.bc_mode in ("list", "both") and getattr(setting, "dirichlet_list_3d", None):
            rows = []
            for tpl in setting.dirichlet_list_3d:
                node = tpl[0]
                ux = tpl[1] if len(tpl) > 1 else None
                uy = tpl[2] if len(tpl) > 2 else None
                uz = tpl[3] if len(tpl) > 3 else None
                rows.append([node, ux, uy, uz])
            _write_rows(out_dir / f"{prefix}_Dirichlet_List.csv",
                        ["Node(AsGiven)","ux","uy","uz"], rows)
        if setting.bc_mode in ("set", "both") and getattr(setting, "dirichlet_sets_3d", None):
            rows = []
            for name, spec in setting.dirichlet_sets_3d.items():
                rows.append([
                    name,
                    len(spec.get("faces",[]) if isinstance(spec.get("faces"), (list,tuple,set)) else []),
                    len(spec.get("nodes",[]) if isinstance(spec.get("nodes"), (list,tuple,set)) else []),
                    len(spec.get("elements",[]) if isinstance(spec.get("elements"), (list,tuple,set)) else []),
                    len(spec.get("seed_nodes",[]) if isinstance(spec.get("seed_nodes"), (list,tuple,set)) else []),
                    spec.get("ux"), spec.get("uy"), spec.get("uz"),
                    spec.get("ang_tol_deg"), spec.get("dilate_deg"), spec.get("relax_first_ring")
                ])
            _write_rows(out_dir / f"{prefix}_Dirichlet_Sets_Summary.csv",
                        ["SetName","faces","nodes","elements","seed_nodes","ux","uy","uz","ang_tol_deg","dilate_deg","relax_first_ring"],
                        rows)

        # Traction list after expansion (pressure/traction 모두 출력)
        if getattr(setting, "traction_mode", "off") in ("list","both") and getattr(setting, "traction_list_3d", None):
            rows = []
            for i, spec in enumerate(setting.traction_list_3d, 1):
                tx = ty = tz = None
                if isinstance(spec.get("traction"), (list, tuple, np.ndarray)) and len(spec["traction"]) >= 3:
                    tx, ty, tz = [float(spec["traction"][0]), float(spec["traction"][1]), float(spec["traction"][2])]
                rows.append([i, " ".join(str(int(x)) for x in spec.get("seed_nodes", [])),
                             spec.get("pressure"), tx, ty, tz])
            _write_rows(out_dir / f"{prefix}_Traction_List_Faces.csv",
                        ["Idx","seed_nodes","pressure","tx","ty","tz"], rows)

        # ▼ 추가: Concentrated 리스트 CSV
        conc = getattr(setting, "concentrated_force_list_3d", None)
        if conc:
            rows = []
            for i, spec in enumerate(conc, 1):
                F = np.asarray(spec.get("F", [0,0,0]), float).reshape(-1,)
                if F.size == 2: F = np.array([F[0], F[1], 0.0])
                if "nodes" in spec and spec["nodes"]:
                    ns = " ".join(str(int(x)) for x in spec["nodes"])
                elif "node" in spec:
                    ns = str(int(spec["node"]))
                else:
                    ns = ""
                rows.append([i, ns, float(F[0]), float(F[1]), float(F[2])])
            _write_rows(out_dir / f"{prefix}_Concentrated_List.csv",
                        ["Idx","nodes","Fx","Fy","Fz"], rows)

    if system == "heat transfer":
        if heat_cfg["dirichlet"]["list"]:
            rows = [[n, val] for (n,val) in heat_cfg["dirichlet"]["list"]]
            _write_rows(out_dir / f"{prefix}_Heat_Dirichlet_List.csv", ["Node(AsGiven)","T"], rows)
        if heat_cfg["dirichlet"]["sets"]:
            rows = []
            for name, spec in heat_cfg["dirichlet"]["sets"].items():
                rows.append([name, len(spec.get("seed_nodes",[])), spec.get("T"),
                             spec.get("ang_tol_deg"), spec.get("dilate_deg"), spec.get("relax_first_ring")])
            _write_rows(out_dir / f"{prefix}_Heat_Dirichlet_Sets_Summary.csv",
                        ["SetName","seed_nodes","T","ang_tol_deg","dilate_deg","relax_first_ring"], rows)
        if heat_cfg["flux"]["list"]:
            rows = []
            for i, spec in enumerate(heat_cfg["flux"]["list"], 1):
                rows.append([i, " ".join(str(int(x)) for x in spec.get("seed_nodes", [])), spec.get("qn")])
            _write_rows(out_dir / f"{prefix}_Heat_Flux_List_Faces.csv",
                        ["Idx","seed_nodes","qn"], rows)
        if heat_cfg["source"]["list"]:
            rows = []
            for i, spec in enumerate(heat_cfg["source"]["list"], 1):
                rows.append([i, len(spec.get("elements",[])), len(spec.get("seed_nodes",[])), spec.get("qv")])
            _write_rows(out_dir / f"{prefix}_Heat_Source_List_Summary.csv",
                        ["Idx","elements","seed_nodes","qv"], rows)

    if system == "diffusion":
        if diff_cfg["dirichlet"]["list"]:
            rows = [[n, val] for (n,val) in diff_cfg["dirichlet"]["list"]]
            _write_rows(out_dir / f"{prefix}_Diff_Dirichlet_List.csv", ["Node(AsGiven)","C"], rows)
        if diff_cfg["dirichlet"]["sets"]:
            rows = []
            for name, spec in diff_cfg["dirichlet"]["sets"].items():
                rows.append([name, len(spec.get("seed_nodes",[])), spec.get("C"),
                             spec.get("ang_tol_deg"), spec.get("dilate_deg"), spec.get("relax_first_ring")])
            _write_rows(out_dir / f"{prefix}_Diff_Dirichlet_Sets_Summary.csv",
                        ["SetName","seed_nodes","C","ang_tol_deg","dilate_deg","relax_first_ring"], rows)
        if diff_cfg["flux"]["list"]:
            rows = []
            for i, spec in enumerate(diff_cfg["flux"]["list"], 1):
                rows.append([i, " ".join(str(int(x)) for x in spec.get("seed_nodes", [])), spec.get("jn")])
            _write_rows(out_dir / f"{prefix}_Diff_Flux_List_Faces.csv",
                        ["Idx","seed_nodes","jn"], rows)
        if diff_cfg["source"]["list"]:
            rows = []
            for i, spec in enumerate(diff_cfg["source"]["list"], 1):
                rows.append([i, len(spec.get("elements",[])), len(spec.get("seed_nodes",[])), spec.get("sv")])
            _write_rows(out_dir / f"{prefix}_Diff_Source_List_Summary.csv",
                        ["Idx","elements","seed_nodes","sv"], rows)

# --- (B-2) 콘솔요약을 CSV 한 장으로도 저장 (Top summary) ---
def _export_top_summary_csv(system, mesh, setting, heat_cfg, diff_cfg, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_BC_Config_TopSummary.csv"
    def _len(x):
        if isinstance(x, (list, tuple, set, dict, np.ndarray)): return len(x)
        return 0
    # elasticity counts
    el_bc_mode = getattr(setting, "bc_mode", "off")
    el_dir_list = getattr(setting, "dirichlet_list_3d", None) if el_bc_mode in ("list","both") else None
    el_dir_sets = getattr(setting, "dirichlet_sets_3d", None) if el_bc_mode in ("set","both") else None
    tr_mode    = getattr(setting, "traction_mode", "off")
    tr_list    = getattr(setting, "traction_list_3d", None) if tr_mode in ("list","both") else None
    tr_sets    = getattr(setting, "traction_sets_3d", None) if tr_mode in ("set","both") else None
    conc       = getattr(setting, "concentrated_force_list_3d", None)

    conc_nodes = 0
    conc_tot = np.zeros(3)
    if isinstance(conc, list):
        for spec in conc:
            if not isinstance(spec, dict): continue
            F = np.asarray(spec.get("F", [0,0,0]), float).reshape(-1,)
            if F.size == 2: F = np.array([F[0], F[1], 0.0])
            if "nodes" in spec and spec["nodes"]:
                n = len(spec["nodes"]); conc_nodes += n; conc_tot += n * F[:3]
            elif "node" in spec:
                conc_nodes += 1; conc_tot += F[:3]

    # heat/diff counts
    ht = heat_cfg
    df = diff_cfg

    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Key","Value"])
        w.writerow(["Dim", mesh.Dimension])
        w.writerow(["System", system])
        w.writerow(["Shape", mesh.element_shape])
        w.writerow(["Order", mesh.element_order])
        # elasticity
        w.writerow(["El.Dirichlet.mode", el_bc_mode])
        w.writerow(["El.Dirichlet.list_count", _len(el_dir_list)])
        w.writerow(["El.Dirichlet.set_count", _len(el_dir_sets)])
        w.writerow(["El.Traction.mode", tr_mode])
        w.writerow(["El.Traction.list_count", _len(tr_list)])
        w.writerow(["El.Traction.set_count", _len(tr_sets)])
        w.writerow(["El.Concentrated.items", _len(conc)])
        w.writerow(["El.Concentrated.applied_nodes", conc_nodes])
        w.writerow(["El.Concentrated.total_Fx", float(conc_tot[0])])
        w.writerow(["El.Concentrated.total_Fy", float(conc_tot[1])])
        w.writerow(["El.Concentrated.total_Fz", float(conc_tot[2])])
        # heat
        w.writerow(["Heat.Dirichlet.mode", ht["dirichlet"]["mode"]])
        w.writerow(["Heat.Dirichlet.list_count", _len(ht["dirichlet"]["list"])])
        w.writerow(["Heat.Dirichlet.set_count", _len(ht["dirichlet"]["sets"])])
        w.writerow(["Heat.Flux.mode", ht["flux"]["mode"]])
        w.writerow(["Heat.Flux.list_count", _len(ht["flux"]["list"])])
        w.writerow(["Heat.Flux.set_count", _len(ht["flux"]["sets"])])
        w.writerow(["Heat.Source.mode", ht["source"]["mode"]])
        w.writerow(["Heat.Source.list_count", _len(ht["source"]["list"])])
        w.writerow(["Heat.Source.set_count", _len(ht["source"]["sets"])])
        # diffusion
        w.writerow(["Diff.Dirichlet.mode", df["dirichlet"]["mode"]])
        w.writerow(["Diff.Dirichlet.list_count", _len(df["dirichlet"]["list"])])
        w.writerow(["Diff.Dirichlet.set_count", _len(df["dirichlet"]["sets"])])
        w.writerow(["Diff.Flux.mode", df["flux"]["mode"]])
        w.writerow(["Diff.Flux.list_count", _len(df["flux"]["list"])])
        w.writerow(["Diff.Flux.set_count", _len(df["flux"]["sets"])])
        w.writerow(["Diff.Source.mode", df["source"]["mode"]])
        w.writerow(["Diff.Source.list_count", _len(df["source"]["list"])])
        w.writerow(["Diff.Source.set_count", _len(df["source"]["sets"])])
    print(f"[WRITE] {path}")

# ======================
# 7) Postprocessing (원본 로직 유지 + BC요약)
# ======================
def fmt3(ncols, int_cols=None):
    fmt = ["%.3f"] * ncols
    if int_cols:
        for i in int_cols:
            fmt[i] = "%d"
    return fmt

# 출력 폴더/접두어 준비
if Dimension == "2D":
    prefix_default = f"vtk_out/{system}_2D_{element_shape}_{element_order}"
else:
    prefix_default = f"vtk_out/{system}_3D_{element_shape}_{element_order}"
prefix = custom_vtk_prefix or prefix_default
out_dir = Path(prefix).parent; out_dir.mkdir(parents=True, exist_ok=True)
base = Path(prefix).name

_print_effective_bc_counts(system, mesh, divide_matrix, BC, out_dir, base)
_export_bc_csv_summaries(system, mesh, setting, _heat_cfg, _diff_cfg, out_dir, base)
_export_top_summary_csv(system, mesh, setting, _heat_cfg, _diff_cfg, out_dir, base)

# 이하 계산/VTK 출력부는 기존과 동일 (생략 없이 유지)
# --------------------------------------------------------------------------------
# 2D/3D 각 시스템별 결과 내보내기 (원본 코드 그대로)
# --------------------------------------------------------------------------------

if Dimension == "2D":
    # ---------- Linear Elasticity ----------
    if system == "linear elasticity":
        d = np.zeros(2 * mesh.NoN)
        d[BC.DoEs] = BC.d_E
        d[BC.DoFs] = divide_matrix.d_F

        U1 = d[0::2]; U2 = d[1::2]
        U = np.column_stack([U1, U2])

        r_E = divide_matrix.K_E @ divide_matrix.d_E + divide_matrix.K_EF @ divide_matrix.d_F - divide_matrix.F_E

        n_elem = len(mesh.conn)
        n_gp   = len(shape_function.gp)
        S_gp   = np.zeros((n_elem, n_gp, 3))
        for e_idx, c in enumerate(mesh.conn):
            xIe = mesh.NL[c, :]
            element_dof = []
            B = np.zeros((3, 2 * shape_function.NPE))
            for node in c:
                element_dof.extend([2 * node, 2 * node + 1])
            d_local = d[element_dof]

            for g_idx, gp in enumerate(shape_function.gp):
                local_dN = shape_function.gradshape(gp, shape_function.NPE, Dimension)
                J = np.dot(local_dN, xIe)
                invJ = np.linalg.inv(J)
                global_dN = invJ @ local_dN
                B[0, 0::2] = global_dN[0, :]
                B[1, 1::2] = global_dN[1, :]
                B[2, 0::2] = global_dN[1, :]
                B[2, 1::2] = global_dN[0, :]

                eps = (B @ d_local).reshape(-1)

                delta_T = T1 - T0
                e0_ps   = np.array([1.0, 1.0, 0.0])
                e0_pe   = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

                if analytical_conditions == "plane strain":
                    Efac = E/((1+nu)*(1-2*nu))
                    D_3D = Efac * np.array([
                        [1-nu, nu,   nu,   0, 0, 0],
                        [nu,   1-nu, nu,   0, 0, 0],
                        [nu,   nu,   1-nu, 0, 0, 0],
                        [0,    0,    0,    (1-2*nu)/2.0, 0, 0],
                        [0,    0,    0,    0, (1-2*nu)/2.0, 0],
                        [0,    0,    0,    0, 0, (1-2*nu)/2.0]
                    ])
                    eps_th_3d = alpha*delta_T*e0_pe
                    sigma_3D  = -(D_3D @ eps_th_3d)
                    sigma_th  = np.array([sigma_3D[0], sigma_3D[1], sigma_3D[3]])
                    sigma     = (D_matrix @ eps) + sigma_th
                else:
                    D2 = E/(1-nu**2) * np.array([[1, nu, 0],[nu, 1, 0],[0, 0, (1-nu)/2.0]])
                    eps_th = alpha*delta_T*e0_ps
                    sigma_th = -(D2 @ eps_th)
                    sigma    = (D_matrix @ eps) + sigma_th

                S_gp[e_idx, g_idx, :] = [float(sigma[0]), float(sigma[1]), float(sigma[2])]

        nodes  = mesh.NL
        elems  = np.array(mesh.conn, dtype=int)
        write_vtk_and_txt(nodes, elems, U, S_gp, prefix, open_paraview_flag=True)

        node_ids = np.array([mesh.row2id.get(i, i+1) for i in range(mesh.NoN)], dtype=int)
        elem_ids = np.arange(1, n_elem+1, dtype=int)

        Uz = np.zeros(mesh.NoN, dtype=float)
        disp2d = np.column_stack([node_ids, mesh.NL[:,0], mesh.NL[:,1], Uz])
        np.savetxt(out_dir / f"{base}_Displacement.csv", disp2d, delimiter=",",
                   header="NodeID,X,Y,U", comments="", fmt=fmt3(4, [0]))

        gp_table = np.zeros((n_elem*n_gp, 5), dtype=float); k = 0
        for e in range(n_elem):
            for g in range(n_gp):
                gp_table[k,0] = elem_ids[e]
                gp_table[k,1] = g+1
                gp_table[k,2:5] = S_gp[e,g,:]
                k += 1
        np.savetxt(out_dir / f"{base}_Stress_Gauss.csv", gp_table, delimiter=",",
                   header="ElemID,GaussID,S11,S22,S12", comments="", fmt=fmt3(5, [0,1]))

        S_centroid = np.zeros((n_elem, 3), dtype=float)
        tiny = 1e-30
        is_quad_22 = (element_shape == "Quad" and len(shape_function.gp) == 4)
        for e_idx, c in enumerate(mesh.conn):
            if is_quad_22:
                S_centroid[e_idx, :] = _extrap_centroid_from_gp(S_gp[e_idx, :, :], shape_function.gp)
            else:
                xIe = mesh.NL[c, :]; wsum = 0.0; acc  = np.zeros(3, dtype=float)
                for g_idx, gp in enumerate(shape_function.gp):
                    local_dN = shape_function.gradshape(gp, shape_function.NPE, Dimension)
                    J = np.dot(local_dN, xIe)
                    detJ = abs(np.linalg.det(J)); wg = float(W_gp[g_idx])
                    acc  += S_gp[e_idx, g_idx, :] * detJ * wg; wsum += detJ * wg
                S_centroid[e_idx, :] = acc / max(wsum, tiny)

        C = np.zeros((n_elem,2), dtype=float)
        for e_idx, c in enumerate(mesh.conn): C[e_idx,:] = mesh.NL[c,:].mean(axis=0)
        cent = np.column_stack([elem_ids, C[:,0], C[:,1], S_centroid])
        np.savetxt(out_dir / f"{base}_Stress_Centroid.csv", cent, delimiter=",",
                   header="ElemID,Cx,Cy,S11,S22,S12", comments="", fmt=fmt3(6, [0]))

    # ---------- Heat Transfer ----------
    elif system == "heat transfer":
        d = np.zeros(mesh.NoN); d[BC.DoEs] = BC.d_E; d[BC.DoFs] = divide_matrix.d_F
        T = d.copy()

        n_elem = len(mesh.conn); n_gp = len(shape_function.gp)
        Q_gp = np.zeros((n_elem, n_gp, 2), dtype=float)

        for e_idx, c in enumerate(mesh.conn):
            xIe = mesh.NL[c, :]; element_dof = list(c); d_local = d[element_dof]
            for g_idx, gp in enumerate(shape_function.gp):
                local_dN = shape_function.gradshape(gp, shape_function.NPE, Dimension)
                J = np.dot(local_dN, xIe); invJ = np.linalg.inv(J)
                global_dN = invJ @ local_dN
                B = np.zeros((2, shape_function.NPE)); B[0, :] = global_dN[0, :]; B[1, :] = global_dN[1, :]
                q = -k_scalar * (B @ d_local)
                Q_gp[e_idx, g_idx, :] = [float(q[0]), float(q[1])]

        nodes  = mesh.NL; elems  = np.array(mesh.conn, dtype=int)
        write_vtk_and_txt_heat(nodes, elems, T, Q_gp, prefix, open_paraview_flag=True)

        node_ids = np.array([mesh.row2id.get(i, i+1) for i in range(mesh.NoN)], dtype=int)
        elem_ids = np.arange(1, n_elem+1, dtype=int)

        tnode = np.column_stack([node_ids, mesh.NL[:,0], mesh.NL[:,1], T])
        np.savetxt(out_dir / f"{base}_Temperature.csv", tnode, delimiter=",",
                   header="NodeID,X,Y,T", comments="", fmt=fmt3(4, [0]))

        Q_centroid = np.zeros((n_elem,2), dtype=float)
        Q_mag      = np.zeros(n_elem, dtype=float)
        C = np.zeros((n_elem,2), dtype=float)
        T_centroid = np.zeros(n_elem, dtype=float)

        tiny = 1e-30
        for e_idx, c in enumerate(mesh.conn):
            C[e_idx,:] = mesh.NL[c,:].mean(axis=0)
            T_centroid[e_idx] = T[list(c)].mean()
            xIe = mesh.NL[c, :]; wsum = 0.0; acc  = np.zeros(2, dtype=float)
            for g_idx, gp in enumerate(shape_function.gp):
                local_dN = shape_function.gradshape(gp, shape_function.NPE, Dimension)
                J = np.dot(local_dN, xIe); detJ = abs(np.linalg.det(J)); wg = float(W_gp[g_idx])
                acc  += Q_gp[e_idx, g_idx, :] * detJ * wg; wsum += detJ * wg
            Q_centroid[e_idx, :] = acc / max(wsum, tiny)
            Q_mag[e_idx] = np.linalg.norm(Q_centroid[e_idx, :])

        qcent = np.column_stack([elem_ids, C[:,0], C[:,1], T_centroid, Q_centroid, Q_mag])
        np.savetxt(out_dir / f"{base}_HeatFlux_Centroid.csv", qcent, delimiter=",",
                   header="ElemID,Cx,Cy,T,HFL1,HFL2,HFL_mag", comments="", fmt=fmt3(7, [0]))

    # ---------- Diffusion ----------
    elif system == "diffusion":
        d = np.zeros(mesh.NoN); d[BC.DoEs] = BC.d_E; d[BC.DoFs] = divide_matrix.d_F
        c = d.copy()

        n_elem = len(mesh.conn); n_gp = len(shape_function.gp)
        J_gp = np.zeros((n_elem, n_gp, 2), dtype=float)

        for e_idx, cidx in enumerate(mesh.conn):
            xIe = mesh.NL[cidx, :]; element_dof = list(cidx); d_local = d[element_dof]
            for g_idx, gp in enumerate(shape_function.gp):
                local_dN = shape_function.gradshape(gp, shape_function.NPE, Dimension)
                Jmat = np.dot(local_dN, xIe); invJ = np.linalg.inv(Jmat)
                global_dN = invJ @ local_dN
                B = np.zeros((2, shape_function.NPE)); B[0, :] = global_dN[0, :]; B[1, :] = global_dN[1, :]
                Jvec = -D_scalar * (B @ d_local)
                J_gp[e_idx, g_idx, :] = [float(Jvec[0]), float(Jvec[1])]

        nodes  = mesh.NL; elems  = np.array(mesh.conn, dtype=int)
        write_vtk_and_txt_diffusion(nodes, elems, c, J_gp, prefix, open_paraview_flag=True)

        node_ids = np.array([mesh.row2id.get(i, i+1) for i in range(mesh.NoN)], dtype=int)
        elem_ids = np.arange(1, n_elem+1, dtype=int)

        cnode = np.column_stack([node_ids, mesh.NL[:,0], mesh.NL[:,1], c])
        np.savetxt(out_dir / f"{base}_Concentration.csv", cnode, delimiter=",",
                   header="NodeID,X,Y,C", comments="", fmt=fmt3(4, [0]))

        J_centroid = np.zeros((n_elem,2), dtype=float)
        J_mag      = np.zeros(n_elem, dtype=float)
        Cc = np.zeros((n_elem,2), dtype=float)
        C_centroid = np.zeros(n_elem, dtype=float)

        tiny = 1e-30
        for e_idx, cidx in enumerate(mesh.conn):
            Cc[e_idx,:] = mesh.NL[cidx,:].mean(axis=0)
            C_centroid[e_idx] = c[list(cidx)].mean()
            xIe = mesh.NL[cidx, :]; wsum = 0.0; acc  = np.zeros(2, dtype=float)
            for g_idx, gp in enumerate(shape_function.gp):
                local_dN = shape_function.gradshape(gp, shape_function.NPE, Dimension)
                Jmat = local_dN @ xIe; detJ = abs(np.linalg.det(Jmat)); wg = float(W_gp[g_idx])
                acc  += J_gp[e_idx, g_idx, :] * detJ * wg; wsum += detJ * wg
            J_centroid[e_idx, :] = acc / max(wsum, tiny)
            J_mag[e_idx] = np.linalg.norm(J_centroid[e_idx, :])

        jcent = np.column_stack([elem_ids, Cc[:,0], Cc[:,1], C_centroid, J_centroid, J_mag])
        np.savetxt(out_dir / f"{base}_MassFlux_Centroid.csv", jcent, delimiter=",",
                   header="ElemID,Cx,Cy,C,MFL1,MFL2,MFL_mag", comments="", fmt=fmt3(7, [0]))

elif Dimension == "3D":

    if system == "linear elasticity":
        d = np.zeros(3 * mesh.NoN)
        d[BC.DoEs] = BC.d_E
        d[BC.DoFs] = divide_matrix.d_F

        U1 = d[0::3]; U2 = d[1::3]; U3 = d[2::3]
        U = np.column_stack([U1, U2, U3])

        r_E = (divide_matrix.K_E @ divide_matrix.d_E
               + divide_matrix.K_EF @ divide_matrix.d_F
               - divide_matrix.F_E)

        n_elem = len(mesh.conn)
        n_gp   = len(shape_function.gp)
        S_gp   = np.zeros((n_elem, n_gp, 6), dtype=float)
        VM_gp  = np.zeros((n_elem, n_gp), dtype=float)

        delta_T = T1 - T0
        eps_th_3d = alpha * delta_T * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        sigma_th  = -(D_matrix @ eps_th_3d)  # 6x1

        def vmises6(s):
            s11, s22, s33, s12, s13, s23 = s
            j2 = 0.5*((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2) + 3.0*(s12**2 + s13**2 + s23**2)
            return float(np.sqrt(max(j2, 0.0)))

        for e_idx, conn_e in enumerate(mesh.conn):
            xIe = mesh.NL[conn_e, :]
            edofs = []
            for nd in conn_e: edofs.extend([3*nd, 3*nd + 1, 3*nd + 2])
            d_local = d[edofs]

            for g_idx, gp in enumerate(shape_function.gp):
                local_dN = shape_function.gradshape(gp, shape_function.NPE, Dimension)  # (3,NPE)
                J = local_dN @ xIe
                invJ = np.linalg.inv(J)
                global_dN = invJ @ local_dN
                dNdx, dNdy, dNdz = global_dN[0, :], global_dN[1, :], global_dN[2, :]

                B = np.zeros((6, 3 * shape_function.NPE), dtype=float)
                B[0, 0::3] = dNdx
                B[1, 1::3] = dNdy
                B[2, 2::3] = dNdz
                B[3, 0::3] = dNdy; B[3, 1::3] = dNdx
                B[4, 1::3] = dNdz; B[4, 2::3] = dNdy
                B[5, 2::3] = dNdx; B[5, 0::3] = dNdz

                eps = B @ d_local
                sigma = (D_matrix @ eps) + sigma_th
                S_gp[e_idx, g_idx, :] = sigma
                VM_gp[e_idx, g_idx]   = vmises6(sigma)

        nodes  = mesh.NL
        elems  = np.array(mesh.conn, int)
        S_out = S_gp.copy()
        S_out[:, :, [4, 5]] = S_out[:, :, [5, 4]]  # S13↔S23 swap(출력 규약)
        write_vtk_and_txt_elasticity_3d(nodes, elems, U, S_out, prefix, open_paraview_flag=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(prefix).name
        n_elem = len(mesh.conn); n_gp = len(shape_function.gp)

        node_ids = np.array([mesh.row2id.get(i, i+1) for i in range(mesh.NoN)], dtype=int)
        elem_ids = np.arange(1, n_elem+1, dtype=int)

        disp3d = np.column_stack([node_ids, mesh.NL[:,0], mesh.NL[:,1], mesh.NL[:,2], U1, U2, U3])
        np.savetxt(out_dir / f"{base}_Displacement.csv", disp3d, delimiter=",",
                   header="NodeID,X,Y,Z,U1,U2,U3", comments="",
                   fmt=("%.0f","%.3f","%.3f","%.3f","%.6f","%.6f","%.6f"))

        gp_tbl = np.zeros((n_elem*n_gp, 8), dtype=float); k = 0
        for e in range(n_elem):
            for g in range(n_gp):
                gp_tbl[k,0] = elem_ids[e]; gp_tbl[k,1] = g+1; gp_tbl[k,2:8] = S_out[e,g,:]; k += 1
        np.savetxt(out_dir / f"{base}_Stress_Gauss.csv", gp_tbl, delimiter=",",
                   header="ElemID,GaussID,S11,S22,S33,S12,S13,S23", comments="",
                   fmt=("%.0f","%.0f","%.6f","%.6f","%.6f","%.6f","%.6f","%.6f"))

        vm_tbl = np.zeros((n_elem*n_gp, 3), dtype=float); k = 0
        for e in range(n_elem):
            for g in range(n_gp):
                vm_tbl[k,0] = elem_ids[e]; vm_tbl[k,1] = g+1; vm_tbl[k,2] = VM_gp[e,g]; k += 1
        np.savetxt(out_dir / f"{base}_VonMises_Gauss.csv", vm_tbl, delimiter=",",
                   header="ElemID,GaussID,VMises", comments="", fmt=("%.0f","%.0f","%.6f"))

        S_centroid = np.zeros((n_elem,6), dtype=float)
        for e_idx in range(n_elem):
            S_centroid[e_idx, :] = _extrap_centroid_from_gp(S_out[e_idx, :, :], shape_function.gp)

        VM_centroid = np.array([
            math.sqrt(max(
                0.5*((s[0]-s[1])**2 + (s[1]-s[2])**2 + (s[2]-s[0])**2) + 3.0*(s[3]**2 + s[4]**2 + s[5]**2),
                0.0
            )) for s in S_centroid
        ])

        C = np.zeros((n_elem,3), dtype=float)
        for e_idx, c in enumerate(mesh.conn): C[e_idx,:] = mesh.NL[c,:].mean(axis=0)
        cent = np.column_stack([elem_ids, C[:,0], C[:,1], C[:,2], S_centroid, VM_centroid])
        np.savetxt(out_dir / f"{base}_Stress_Centroid.csv", cent, delimiter=",",
                   header="ElemID,Cx,Cy,Cz,S11,S22,S33,S12,S13,S23,VMises", comments="",
                   fmt=("%.0f","%.3f","%.3f","%.3f","%.6f","%.6f","%.6f","%.6f","%.6f","%.6f","%.6f"))

        rxryrz = {}
        for dof_idx, val in zip(BC.DoEs, r_E.flatten()):
            node = dof_idx // 3; comp = dof_idx % 3
            if node not in rxryrz: rxryrz[node] = [0.0, 0.0, 0.0]
            rxryrz[node][comp] += float(val)
        if rxryrz:
            r_rows = []
            for node, (rx,ry,rz) in sorted(rxryrz.items()):
                nid = mesh.row2id.get(node, node+1)
                r_rows.append([nid, rx, ry, rz, math.sqrt(rx*rx + ry*ry + rz*rz)])
            r_arr = np.array(r_rows, dtype=float)
            np.savetxt(out_dir / f"{base}_Reactions.csv", r_arr, delimiter=",",
                       header="NodeID,Rx,Ry,Rz,Rmag", comments="", fmt=("%.0f","%.6f","%.6f","%.6f","%.6f"))

        Fg = np.zeros(3*mesh.NoN, dtype=float)
        Fg[BC.DoEs] = divide_matrix.F_E; Fg[BC.DoFs] = divide_matrix.F_F
        node_ids = np.array([mesh.row2id.get(i, i+1) for i in range(mesh.NoN)], dtype=int)
        Fxyz = np.column_stack([
            node_ids, Fg[0::3], Fg[1::3], Fg[2::3],
            np.linalg.norm(np.column_stack([Fg[0::3], Fg[1::3], Fg[2::3]]), axis=1)
        ])
        np.savetxt(out_dir / f"{base}_RHS_perNode.csv", Fxyz, delimiter=",",
                   header="NodeID,Fx,Fy,Fz,Fmag", comments="", fmt=("%.0f","%.6f","%.6f","%.6f","%.6f"))

    elif system == "heat transfer":
        d = np.zeros(mesh.NoN); d[BC.DoEs] = BC.d_E; d[BC.DoFs] = divide_matrix.d_F
        T = d[:]

        n_elem = len(mesh.conn); n_gp = len(shape_function.gp)
        Q_gp = np.zeros((n_elem, n_gp, 3), dtype=float)

        for e_idx, conn_e in enumerate(mesh.conn):
            xIe = mesh.NL[conn_e, :]; d_local = d[conn_e]
            for g_idx, gp in enumerate(shape_function.gp):
                local_dN = shape_function.gradshape(gp, shape_function.NPE, Dimension)
                J = local_dN @ xIe; invJ = np.linalg.inv(J)
                global_dN = invJ @ local_dN
                B = global_dN
                q = -k_scalar * (B @ d_local)
                Q_gp[e_idx, g_idx, :] = q

        nodes  = mesh.NL; elems  = np.array(mesh.conn, int)
        write_vtk_and_txt_heat_3d(nodes, elems, T, Q_gp, prefix, open_paraview_flag=True)

        node_ids = np.array([mesh.row2id.get(i,i+1) for i in range(mesh.NoN)], int)
        elem_ids = np.arange(1, n_elem+1, 1)

        tnode = np.column_stack([node_ids, mesh.NL, T])
        np.savetxt(out_dir/f"{base}_Temperature.csv", tnode, delimiter=",",
                   header="NodeID,X,Y,Z,T", comments="", fmt=fmt3(5,[0]))

        Q_centroid = np.zeros((n_elem,3), dtype=float)
        Q_mag      = np.zeros(n_elem, dtype=float)
        C = np.array([mesh.NL[c,:].mean(axis=0) for c in mesh.conn])
        T_centroid = np.array([T[list(c)].mean() for c in mesh.conn])

        tiny = 1e-30
        for e_idx, conn_e in enumerate(mesh.conn):
            xIe = mesh.NL[conn_e, :]; wsum = 0.0; acc  = np.zeros(3, dtype=float)
            for g_idx, gp in enumerate(shape_function.gp):
                local_dN = shape_function.gradshape(gp, shape_function.NPE, Dimension)
                J = local_dN @ xIe; detJ = abs(np.linalg.det(J)); wg = float(W_gp[g_idx])
                acc  += Q_gp[e_idx, g_idx, :] * detJ * wg; wsum += detJ * wg
            Q_centroid[e_idx, :] = acc / max(wsum, tiny)
            Q_mag[e_idx] = np.linalg.norm(Q_centroid[e_idx, :])

        qcent = np.column_stack([elem_ids, C, T_centroid, Q_centroid, Q_mag])
        np.savetxt(out_dir/f"{base}_HeatFlux_Centroid.csv", qcent, delimiter=",",
                   header="ElemID,Cx,Cy,Cz,T,HFL1,HFL2,HFL3,HFL_mag", comments="", fmt=fmt3(9, [0]))

    elif system == "diffusion":
        d = np.zeros(mesh.NoN); d[BC.DoEs] = BC.d_E; d[BC.DoFs] = divide_matrix.d_F
        c = d[:]

        n_elem = len(mesh.conn); n_gp = len(shape_function.gp)
        J_gp = np.zeros((n_elem, n_gp, 3), dtype=float)

        for e_idx, conn_e in enumerate(mesh.conn):
            xIe = mesh.NL[conn_e,:]; d_local = d[conn_e]
            for g_idx,gp in enumerate(shape_function.gp):
                local_dN = shape_function.gradshape(gp, shape_function.NPE, Dimension)
                Jmat = local_dN @ xIe; invJ = np.linalg.inv(Jmat)
                global_dN = invJ @ local_dN
                B = global_dN
                Jvec = -D_scalar * (B @ d_local)
                J_gp[e_idx,g_idx,:] = Jvec

        # === 센트로이드(가중평균) 사용 ===
        J_centroid = np.zeros((n_elem,3), dtype=float)
        tiny = 1e-30
        for e_idx, conn_e in enumerate(mesh.conn):
            xIe = mesh.NL[conn_e, :]; wsum = 0.0; acc  = np.zeros(3, dtype=float)
            for g_idx, gp in enumerate(shape_function.gp):
                local_dN = shape_function.gradshape(gp, shape_function.NPE, Dimension)
                Jmat = local_dN @ xIe; detJ = abs(np.linalg.det(Jmat)); wg = float(W_gp[g_idx])
                acc  += J_gp[e_idx, g_idx, :] * detJ * wg; wsum += detJ * wg
            J_centroid[e_idx, :] = acc / max(wsum, tiny)

        nodes = mesh.NL; elems = np.array(mesh.conn,int)
        write_vtk_and_txt_diffusion_3d(nodes, elems, c, J_centroid, prefix, open_paraview_flag=True)

        node_ids = np.array([mesh.row2id.get(i,i+1) for i in range(mesh.NoN)],int)
        elem_ids = np.arange(1, n_elem+1, dtype=int)

        cnode = np.column_stack([node_ids,mesh.NL,c])
        np.savetxt(out_dir/f"{base}_Concentration.csv", cnode, delimiter=",",
                   header="NodeID,X,Y,Z,C", comments="", fmt=fmt3(5,[0]))

        J_mag = np.linalg.norm(J_centroid, axis=1)
        Cc = np.array([mesh.NL[np.asarray(cidx, int), :].mean(axis=0) for cidx in mesh.conn])
        C_centroid = np.array([c[np.asarray(cidx, int)].mean() for cidx in mesh.conn])

        jcent = np.column_stack([elem_ids, Cc, C_centroid, J_centroid, J_mag])
        np.savetxt(out_dir/f"{base}_MassFlux_Centroid.csv", jcent, delimiter=",",
                   header="ElemID,Cx,Cy,Cz,C,MFL1,MFL2,MFL3,MFL_mag", comments="", fmt=fmt3(9, [0]))
