# -*- coding: utf-8 -*-
"""
VTK(legacy .vtk, UNSTRUCTURED_GRID) Exporter + ParaView 자동 실행
- 2D: Tri3, Tri6, Quad4, Quad8
- 3D: Tet4, Tet10, Hex8, Hex20

Abaqus-style variable names:
- Linear Elasticity
  * 2D
      CELL_DATA : S11_C, S22_C, S12_C, MISES_C
      POINT_DATA: U (vector), U1, U2, [옵션] S11_P, S22_P, S12_P, MISES_P
  * 3D
      CELL_DATA : S11_C, S22_C, S33_C, S12_C, S13_C, S23_C, MISES_C
      POINT_DATA: U (vector), U1, U2, U3, [옵션] S11_P ... S23_P, MISES_P
- Heat Transfer
  * 2D
      POINT_DATA: NT, [옵션] VECTORS HFL_P + HFL1_P, HFL2_P, HFLMAG_P
      CELL_DATA : VECTORS HFL_C + HFL1_C, HFL2_C, HFLMAG_C
  * 3D
      POINT_DATA: NT, [옵션] VECTORS HFL_P + HFL1_P, HFL2_P, HFL3_P, HFLMAG_P
      CELL_DATA : VECTORS HFL_C + HFL1_C, HFL2_C, HFL3_C, HFLMAG_C
- Diffusion
  * 2D
      POINT_DATA: Conc, [옵션] VECTORS MFL_P + MFL1_P, MFL2_P, MFLMAG_P
      CELL_DATA : VECTORS MFL_C + MFL1_C, MFL2_C, MFLMAG_C
  * 3D
      POINT_DATA: Conc, [옵션] VECTORS MFL_P + MFL1_P, MFL2_P, MFL3_P, MFLMAG_P
      CELL_DATA : VECTORS MFL_C + MFL1_C, MFL2_C, MFL3_C, MFLMAG_C
"""

import numpy as np
import os, sys, subprocess, shutil, glob
from typing import List, Optional, Sequence, Union, Dict

# ---------------- VTK cell types ----------------
# 2D
VTK_TRIANGLE = 5
VTK_QUAD = 9
VTK_QUADRATIC_TRIANGLE = 22
VTK_QUADRATIC_QUAD = 23
# 3D
VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_QUADRATIC_TETRA = 24
VTK_QUADRATIC_HEXAHEDRON = 25

# ============================================================
# ===============  Common helpers (2D & 3D)  ================
# ============================================================
def _guess_paraview_path() -> Optional[str]:
    env = os.environ.get("PARAVIEW_EXE")
    if env and os.path.isfile(env):
        return env
    exe = shutil.which("paraview") or shutil.which("paraview.exe")
    if exe:
        return exe
    if sys.platform.startswith("win"):
        for base in (r"C:\Program Files", r"C:\Program Files (x86)"):
            for p in glob.glob(os.path.join(base, "ParaView*")):
                cand = os.path.join(p, "bin", "paraview.exe")
                if os.path.isfile(cand):
                    return cand
    elif sys.platform == "darwin":
        for app in sorted(glob.glob("/Applications/ParaView*.app"), reverse=True):
            cand = os.path.join(app, "Contents", "MacOS", "paraview")
            if os.path.isfile(cand):
                return cand
    return None

def open_in_paraview(vtk_path: str, paraview_path: Optional[str] = None) -> None:
    vtk_abs = os.path.abspath(vtk_path)
    if not os.path.isfile(vtk_abs):
        raise FileNotFoundError(f"VTK 파일을 찾을 수 없습니다: {vtk_abs}")
    exe = paraview_path or _guess_paraview_path()
    if not exe or not os.path.isfile(exe):
        raise RuntimeError(
            "ParaView 실행 파일을 찾지 못했습니다.\n"
            "- 환경변수 PARAVIEW_EXE에 경로 지정하거나,\n"
            "- open_in_paraview(vtk, paraview_path='.../paraview(.exe)')로 전달하세요."
        )
    subprocess.Popen([exe, vtk_abs])
    print(f"[ParaView] 실행: {exe} {vtk_abs}")

# ============================================================
# ========================  2D  ==============================
# ============================================================
class VTKExporter2D:
    def __init__(self):
        self.nodes: Optional[np.ndarray] = None      # (n_nodes, 3)
        self.elements: List[np.ndarray] = []         # 각 셀의 노드 인덱스 배열
        self.cell_types: List[int] = []              # 각 셀의 VTK 타입 코드

        # ----- POINT_DATA -----
        self.U: Optional[np.ndarray] = None          # (n_nodes, 3)
        self.point_scalars: Dict[str, np.ndarray] = {}  # name -> (n_nodes,)

        # ----- CELL_DATA: Stress -----
        self.stress_gauss: Optional[np.ndarray] = None
        self.stress_centroid: Optional[np.ndarray] = None
        self.gauss_weights: Optional[np.ndarray] = None

        # ----- CELL_DATA: Flux (Heat/Diffusion) -----
        self.flux_name: Optional[str] = None               # 'HFL' or 'MFL'
        self.flux_gauss: Optional[np.ndarray] = None       # (n_e, n_gp, 2)
        self.flux_centroid: Optional[np.ndarray] = None    # (n_e, 2)
        self.flux_gauss_weights: Optional[np.ndarray] = None

    # ---------- setters ----------
    def set_mesh(self,
                 node_coords: np.ndarray,
                 element_connectivity: Union[List[Sequence[int]], np.ndarray],
                 input_order: str = 'vtk',
                 conn_base: str = 'auto'):
        nodes = np.array(node_coords, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
            raise ValueError("node_coords must be (n,2) or (n,3)")
        if nodes.shape[1] == 2:
            nodes = np.hstack([nodes, np.zeros((nodes.shape[0], 1), dtype=float)])
        self.nodes = nodes

        if isinstance(element_connectivity, np.ndarray) and element_connectivity.dtype != object:
            elems = [np.array(row, dtype=int) for row in element_connectivity]
        else:
            elems = [np.array(e, dtype=int) for e in element_connectivity]

        if conn_base not in ('0', '1', 'auto'):
            raise ValueError("conn_base must be 'auto', '0', or '1'")
        if conn_base == '1':
            elems = [e - 1 for e in elems]
        elif conn_base == 'auto':
            mx = max(int(e.max()) for e in elems)
            if self.nodes is not None and mx >= self.nodes.shape[0]:
                elems = [e - 1 for e in elems]

        if input_order.lower() == 'abaqus':
            elems = [self._abaqus_to_vtk_order_2d(e) for e in elems]

        self.elements = elems
        self.cell_types = [self._resolve_cell_type_2d(len(e)) for e in self.elements]

    def set_displacements(self, U: np.ndarray):
        U = np.asarray(U, dtype=float)
        assert self.nodes is not None, "set_mesh를 먼저 호출하세요."
        if U.ndim != 2 or U.shape[0] != self.nodes.shape[0] or U.shape[1] not in (2, 3):
            raise AssertionError("U는 (n_nodes,2) 또는 (n_nodes,3) 이어야 합니다.")
        if U.shape[1] == 2:
            U = np.hstack([U, np.zeros((U.shape[0], 1), dtype=float)])
        self.U = U

    def add_point_scalar(self, name: str, values: np.ndarray):
        arr = np.asarray(values, dtype=float).reshape(-1)
        assert self.nodes is not None, "set_mesh를 먼저 호출하세요."
        if arr.shape[0] != self.nodes.shape[0]:
            raise AssertionError(f"POINT_SCALAR '{name}' 길이가 노드 수와 다릅니다.")
        self.point_scalars[name] = arr

    def set_stresses(self,
                     stress: Union[np.ndarray, List[np.ndarray]],
                     gauss_weights: Optional[Union[np.ndarray, Sequence[float]]] = None):
        if stress is None:
            self.stress_gauss = None
            self.stress_centroid = None
            self.gauss_weights = None
            return
        if isinstance(stress, list):
            arrs = [np.asarray(s, dtype=float) for s in stress]
            n_elems = len(arrs)
            n_comp = arrs[0].shape[1]
            max_gp = max(a.shape[0] for a in arrs)
            S = np.full((n_elems, max_gp, n_comp), np.nan, dtype=float)
            for e, a in enumerate(arrs):
                S[e, :a.shape[0], :] = a
            self.stress_gauss = S
            self.stress_centroid = None
        else:
            S = np.asarray(stress, dtype=float)
            if S.ndim == 3:
                self.stress_gauss = S
                self.stress_centroid = None
            elif S.ndim == 2:
                self.stress_gauss = None
                self.stress_centroid = S
            else:
                raise ValueError("stress 형식 오류")
        self.gauss_weights = None if gauss_weights is None else np.asarray(gauss_weights, dtype=float)

    def set_fluxes(self,
                   name: str,
                   flux: Union[np.ndarray, List[np.ndarray]],
                   gauss_weights: Optional[Union[np.ndarray, Sequence[float]]] = None):
        self.flux_name = str(name)
        if isinstance(flux, list):
            arrs = [np.asarray(f, dtype=float) for f in flux]
            n_elems = len(arrs)
            max_gp = max(a.shape[0] for a in arrs)
            F = np.full((n_elems, max_gp, 2), np.nan, dtype=float)
            for e, a in enumerate(arrs):
                F[e, :a.shape[0], :] = a
            self.flux_gauss = F
            self.flux_centroid = None
        else:
            F = np.asarray(flux, dtype=float)
            if F.ndim == 3:
                assert F.shape[2] == 2, "flux의 마지막 차원은 2여야 합니다."
                self.flux_gauss = F
                self.flux_centroid = None
            elif F.ndim == 2:
                assert F.shape[1] == 2, "flux의 마지막 차원은 2여야 합니다."
                self.flux_gauss = None
                self.flux_centroid = F
            else:
                raise ValueError("flux 형식 오류")
        self.flux_gauss_weights = None if gauss_weights is None else np.asarray(gauss_weights, dtype=float)

    # ---------- helpers ----------
    def _resolve_cell_type_2d(self, n: int) -> int:
        if n == 3: return VTK_TRIANGLE
        if n == 4: return VTK_QUAD
        if n == 6: return VTK_QUADRATIC_TRIANGLE
        if n == 8: return VTK_QUADRATIC_QUAD
        raise ValueError(f"지원하지 않는 2D 요소 노드수: {n}")

    def _abaqus_to_vtk_order_2d(self, conn: np.ndarray) -> np.ndarray:
        # 대부분 동일. 필요 시 여기에서 매핑.
        return conn.copy()

    def _ensure_comp_names_2d(self, n_comp: int) -> List[str]:
        if n_comp == 3: return ['S11', 'S22', 'S12']
        if n_comp == 4: return ['S11', 'S22', 'S33', 'S12']
        raise ValueError("2D 응력 성분수는 3 또는 4여야 합니다.")

    def _vm_plane(self, s: np.ndarray) -> float:
        if len(s) == 3:
            s11, s22, s12 = s
            return np.sqrt(s11**2 + s22**2 - s11*s22 + 3.0*s12**2)
        elif len(s) == 4:
            s11, s22, s33, s12 = s
            term = 0.5*((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2) + 3.0*(s12**2)
            return np.sqrt(term)
        else:
            raise ValueError("VM 계산 벡터 길이는 3 또는 4여야 합니다.")

    def _aggregate_centroid_stress(self) -> np.ndarray:
        if self.stress_centroid is not None:
            return self.stress_centroid
        assert self.stress_gauss is not None, "집계할 가우스 응력이 없습니다."
        S = self.stress_gauss
        n_e, n_gp, n_comp = S.shape
        out = np.zeros((n_e, n_comp), dtype=float)
        for e in range(n_e):
            Se = S[e]
            mask = ~np.isnan(Se).any(axis=1)
            if not np.any(mask):
                out[e] = np.nan
                continue
            Sev = Se[mask]
            if self.gauss_weights is not None:
                if self.gauss_weights.ndim == 1 and self.gauss_weights.shape[0] == n_gp:
                    w = self.gauss_weights[mask]
                elif self.gauss_weights.ndim == 2 and self.gauss_weights.shape == (n_e, n_gp):
                    w = self.gauss_weights[e, mask]
                else:
                    w = np.ones(Sev.shape[0])
            else:
                w = np.ones(Sev.shape[0])
            wsum = w.sum()
            out[e] = (Sev * (w[:, None]/wsum)).sum(axis=0) if wsum != 0 else np.nanmean(Sev, axis=0)
        return out

    def _aggregate_centroid_flux2(self) -> np.ndarray:
        if self.flux_centroid is not None:
            return self.flux_centroid
        assert self.flux_gauss is not None, "집계할 가우스 플럭스가 없습니다."
        F = self.flux_gauss
        n_e, n_gp, _ = F.shape
        out = np.zeros((n_e, 2), dtype=float)
        for e in range(n_e):
            Fe = F[e]
            mask = ~np.isnan(Fe).any(axis=1)
            if not np.any(mask):
                out[e] = np.nan
                continue
            Fev = Fe[mask]
            if self.flux_gauss_weights is not None:
                if self.flux_gauss_weights.ndim == 1 and self.flux_gauss_weights.shape[0] == n_gp:
                    w = self.flux_gauss_weights[mask]
                elif self.flux_gauss_weights.ndim == 2 and self.flux_gauss_weights.shape == (n_e, n_gp):
                    w = self.flux_gauss_weights[e, mask]
                else:
                    w = np.ones(Fev.shape[0])
            else:
                w = np.ones(Fev.shape[0])
            wsum = w.sum()
            out[e] = (Fev * (w[:, None]/wsum)).sum(axis=0) if wsum != 0 else np.nanmean(Fev, axis=0)
        return out

    # ---- area & cell->point averaging (2D) ----
    def _cell_area_2d(self, e: np.ndarray) -> float:
        xy = self.nodes[e, :2]
        x = xy[:, 0]; y = xy[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def _cell_to_point_average_2d(self, C: np.ndarray) -> np.ndarray:
        """
        C: (n_elem, n_comp) -> (n_nodes, n_comp), area-weighted
        """
        n_nodes = self.nodes.shape[0]
        n_comp  = C.shape[1]
        num   = np.zeros((n_nodes, n_comp), dtype=float)
        denom = np.zeros((n_nodes,), dtype=float)
        for e_idx, e in enumerate(self.elements):
            val = C[e_idx]
            if np.isnan(val).any():  # skip invalid
                continue
            w = self._cell_area_2d(e)
            for nid in e:
                num[nid]   += w * val
                denom[nid] += w
        out = np.full((n_nodes, n_comp), np.nan, dtype=float)
        mask = denom > 0
        out[mask] = num[mask] / denom[mask, None]
        return out

    # ---------- writer (2D) ----------
    def write_legacy_vtk(self,
                         filename: str,
                         stress_mode: str = 'centroid',   # (고정) 내부 집계는 센트로이드 평균
                         flux_mode: str = 'centroid',     # (고정) 내부 집계는 센트로이드 평균
                         write_U_vectors: bool = True,
                         write_U_scalars: bool = True,
                         float_fmt_coords: str = "%.15g",
                         float_fmt_data: str = "%.8e",
                         mirror_txt_path: Optional[str] = None,
                         export_point_from_cell: bool = False) -> bool:  # ★ 옵션 추가
        assert self.nodes is not None and len(self.elements) > 0, "메쉬가 설정되지 않았습니다."
        npts = self.nodes.shape[0]
        nelem = len(self.elements)
        total_ints = sum(1 + len(e) for e in self.elements)

        S_C = None
        if (self.stress_centroid is not None) or (self.stress_gauss is not None):
            S_C = self._aggregate_centroid_stress() if self.stress_centroid is None else self.stress_centroid

        F_C = None
        if (self.flux_centroid is not None) or (self.flux_gauss is not None):
            F_C = self._aggregate_centroid_flux2() if self.flux_centroid is None else self.flux_centroid

        # ★ 노드 평균(옵션)
        S_P = None
        F_P = None
        if export_point_from_cell and (S_C is not None):
            S_P = self._cell_to_point_average_2d(S_C)
        if export_point_from_cell and (F_C is not None):
            F_P = self._cell_to_point_average_2d(F_C)

        def _w(fhs, s: str):
            for fh in fhs:
                fh.write(s)

        try:
            fhs = []
            with open(filename, 'w') as fvtk:
                fhs.append(fvtk)
                if mirror_txt_path:
                    ftxt = open(mirror_txt_path, 'w')
                    fhs.append(ftxt)
                else:
                    ftxt = None

                _w(fhs, "# vtk DataFile Version 3.0\n")
                _w(fhs, "2D FEA export (CENTROID + optional POINT averages)\n")
                _w(fhs, "ASCII\nDATASET UNSTRUCTURED_GRID\n")

                _w(fhs, f"POINTS {npts} float\n")
                for p in self.nodes:
                    _w(fhs, f"{float_fmt_coords % p[0]} {float_fmt_coords % p[1]} {float_fmt_coords % p[2]}\n")

                _w(fhs, f"CELLS {nelem} {total_ints}\n")
                for e in self.elements:
                    _w(fhs, str(len(e)));  _w(fhs, "".join(f" {int(nid)}" for nid in e));  _w(fhs, "\n")

                _w(fhs, f"CELL_TYPES {nelem}\n")
                for ct in self.cell_types: _w(fhs, f"{ct}\n")

                # ---------- POINT_DATA ----------
                if (self.U is not None) or (len(self.point_scalars) > 0) or (S_P is not None) or (F_P is not None):
                    _w(fhs, f"POINT_DATA {npts}\n")

                    # U
                    if self.U is not None and (write_U_vectors or write_U_scalars):
                        if write_U_vectors:
                            _w(fhs, "VECTORS U float\n")
                            for u in self.U:
                                _w(fhs, f"{float_fmt_data % u[0]} {float_fmt_data % u[1]} {float_fmt_data % u[2]}\n")
                        if write_U_scalars:
                            _w(fhs, "SCALARS U1 float 1\nLOOKUP_TABLE default\n")
                            for u in self.U: _w(fhs, f"{float_fmt_data % u[0]}\n")
                            _w(fhs, "SCALARS U2 float 1\nLOOKUP_TABLE default\n")
                            for u in self.U: _w(fhs, f"{float_fmt_data % u[1]}\n")

                    # 사용자 정의 POINT_SCALARS
                    for name, arr in self.point_scalars.items():
                        _w(fhs, f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
                        for v in arr: _w(fhs, f"{float_fmt_data % v}\n")

                    # ★ 응력(노드 평균)
                    if S_P is not None:
                        names = self._ensure_comp_names_2d(S_P.shape[1])
                        for j, name in enumerate(names):
                            _w(fhs, f"SCALARS {name}_P float 1\nLOOKUP_TABLE default\n")
                            for i in range(npts): _w(fhs, f"{float_fmt_data % S_P[i, j]}\n")
                        # Von Mises @ POINT
                        _w(fhs, "SCALARS MISES_P float 1\nLOOKUP_TABLE default\n")
                        for i in range(npts):
                            _w(fhs, f"{float_fmt_data % self._vm_plane(S_P[i])}\n")

                    # ★ 플럭스(노드 평균)
                    if (F_P is not None) and self.flux_name:
                        fname = self.flux_name
                        _w(fhs, f"VECTORS {fname}_P float\n")
                        for i in range(npts):
                            v = F_P[i]
                            _w(fhs, f"{float_fmt_data % v[0]} {float_fmt_data % v[1]} 0.0\n")
                        _w(fhs, f"SCALARS {fname}1_P float 1\nLOOKUP_TABLE default\n")
                        for i in range(npts): _w(fhs, f"{float_fmt_data % F_P[i,0]}\n")
                        _w(fhs, f"SCALARS {fname}2_P float 1\nLOOKUP_TABLE default\n")
                        for i in range(npts): _w(fhs, f"{float_fmt_data % F_P[i,1]}\n")
                        if fname == 'MFL':
                            _w(fhs, f"SCALARS {fname}MAG_P float 1\nLOOKUP_TABLE default\n")
                            for i in range(npts):
                                v = F_P[i]; mag = float(np.linalg.norm(v)) if not np.isnan(v).any() else np.nan
                                _w(fhs, f"{float_fmt_data % mag}\n")
                        if fname == 'HFL':
                            _w(fhs, f"SCALARS {fname}MAG_P float 1\nLOOKUP_TABLE default\n")
                            for i in range(npts):
                                v = F_P[i]; mag = float(np.linalg.norm(v)) if not np.isnan(v).any() else np.nan
                                _w(fhs, f"{float_fmt_data % mag}\n")

                # ---------- CELL_DATA ----------
                if (S_C is not None) or (F_C is not None):
                    _w(fhs, f"CELL_DATA {nelem}\n")
                if S_C is not None:
                    names = self._ensure_comp_names_2d(S_C.shape[1])
                    for j, name in enumerate(names):
                        _w(fhs, f"SCALARS {name}_C float 1\nLOOKUP_TABLE default\n")
                        for e in range(nelem): _w(fhs, f"{float_fmt_data % S_C[e, j]}\n")
                    _w(fhs, "SCALARS MISES_C float 1\nLOOKUP_TABLE default\n")
                    for e in range(nelem): _w(fhs, f"{float_fmt_data % self._vm_plane(S_C[e])}\n")

                if (F_C is not None) and self.flux_name:
                    fname = self.flux_name
                    _w(fhs, f"VECTORS {fname}_C float\n")
                    for e in range(nelem):
                        v = F_C[e]; vx = float(v[0]); vy = float(v[1])
                        _w(fhs, f"{float_fmt_data % vx} {float_fmt_data % vy} 0.0\n")
                    _w(fhs, f"SCALARS {fname}1_C float 1\nLOOKUP_TABLE default\n")
                    for e in range(nelem): _w(fhs, f"{float_fmt_data % F_C[e,0]}\n")
                    _w(fhs, f"SCALARS {fname}2_C float 1\nLOOKUP_TABLE default\n")
                    for e in range(nelem): _w(fhs, f"{float_fmt_data % F_C[e,1]}\n")
                    if fname == 'MFL':
                        _w(fhs, f"SCALARS {fname}MAG_C float 1\nLOOKUP_TABLE default\n")
                        for e in range(nelem):
                            v = F_C[e]
                            mag = np.sqrt(v[0]**2 + v[1]**2) if not np.isnan(v).any() else np.nan
                            _w(fhs, f"{float_fmt_data % mag}\n")
                    if fname == 'HFL':
                        _w(fhs, f"SCALARS {fname}MAG_C float 1\nLOOKUP_TABLE default\n")
                        for e in range(nelem):
                            v = F_C[e]
                            mag = np.sqrt(v[0]**2 + v[1]**2) if not np.isnan(v).any() else np.nan
                            _w(fhs, f"{float_fmt_data % mag}\n")

                if ftxt is not None: ftxt.close()

            print(f"[VTK] wrote: {filename}")
            if mirror_txt_path: print(f"[TXT] wrote: {mirror_txt_path}")
            return True
        except Exception as ex:
            print(f"[ERROR] Failed to write VTK/TXT: {ex}")
            return False


# ============================================================
# ========================  3D  ==============================
# ============================================================
class VTKExporter3D:
    def __init__(self):
        self.nodes: Optional[np.ndarray] = None      # (n_nodes, 3)
        self.elements: List[np.ndarray] = []         # 각 셀의 노드 인덱스 배열
        self.cell_types: List[int] = []              # 각 셀의 VTK 타입 코드

        # ----- POINT_DATA -----
        self.U: Optional[np.ndarray] = None          # (n_nodes, 3)
        self.point_scalars: Dict[str, np.ndarray] = {}

        # ----- CELL_DATA: Stress (6 comps) -----
        self.stress_gauss: Optional[np.ndarray] = None   # (n_e, n_gp, 6)
        self.stress_centroid: Optional[np.ndarray] = None# (n_e, 6)
        self.gauss_weights: Optional[np.ndarray] = None

        # ----- CELL_DATA: Flux (3 comps) -----
        self.flux_name: Optional[str] = None             # 'HFL' or 'MFL'
        self.flux_gauss: Optional[np.ndarray] = None     # (n_e, n_gp, 3)
        self.flux_centroid: Optional[np.ndarray] = None  # (n_e, 3)
        self.flux_gauss_weights: Optional[np.ndarray] = None

    # ---------- setters ----------
    def set_mesh(self,
                 node_coords: np.ndarray,
                 element_connectivity: Union[List[Sequence[int]], np.ndarray],
                 input_order: str = 'vtk',
                 conn_base: str = 'auto'):
        nodes = np.asarray(node_coords, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("3D node_coords must be (n,3)")
        self.nodes = nodes

        if isinstance(element_connectivity, np.ndarray) and element_connectivity.dtype != object:
            elems = [np.array(row, dtype=int) for row in element_connectivity]
        else:
            elems = [np.array(e, dtype=int) for e in element_connectivity]

        if conn_base not in ('0', '1', 'auto'):
            raise ValueError("conn_base must be 'auto', '0', or '1'")
        if conn_base == '1':
            elems = [e - 1 for e in elems]
        elif conn_base == 'auto':
            mx = max(int(e.max()) for e in elems)
            if self.nodes is not None and mx >= self.nodes.shape[0]:
                elems = [e - 1 for e in elems]

        if input_order.lower() == 'abaqus':
            elems = [self._abaqus_to_vtk_order_3d(e) for e in elems]

        self.elements = elems
        self.cell_types = [self._resolve_cell_type_3d(len(e)) for e in self.elements]

    def set_displacements(self, U: np.ndarray):
        U = np.asarray(U, dtype=float)
        assert self.nodes is not None, "set_mesh를 먼저 호출하세요."
        if U.ndim != 2 or U.shape != (self.nodes.shape[0], 3):
            raise AssertionError("3D U는 (n_nodes,3) 이어야 합니다.")
        self.U = U

    def add_point_scalar(self, name: str, values: np.ndarray):
        arr = np.asarray(values, dtype=float).reshape(-1)
        assert self.nodes is not None, "set_mesh를 먼저 호출하세요."
        if arr.shape[0] != self.nodes.shape[0]:
            raise AssertionError(f"POINT_SCALAR '{name}' 길이가 노드 수와 다릅니다.")
        self.point_scalars[name] = arr

    def set_stresses(self,
                     stress: Union[np.ndarray, List[np.ndarray]],
                     gauss_weights: Optional[Union[np.ndarray, Sequence[float]]] = None):
        if stress is None:
            self.stress_gauss = None
            self.stress_centroid = None
            self.gauss_weights = None
            return
        if isinstance(stress, list):
            arrs = [np.asarray(s, dtype=float) for s in stress]
            n_elems = len(arrs)
            n_comp = arrs[0].shape[1]
            max_gp = max(a.shape[0] for a in arrs)
            S = np.full((n_elems, max_gp, n_comp), np.nan, dtype=float)
            for e, a in enumerate(arrs):
                S[e, :a.shape[0], :] = a
            self.stress_gauss = S
            self.stress_centroid = None
        else:
            S = np.asarray(stress, dtype=float)
            if S.ndim == 3:
                self.stress_gauss = S
                self.stress_centroid = None
            elif S.ndim == 2:
                self.stress_gauss = None
                self.stress_centroid = S
            else:
                raise ValueError("stress 형식 오류")
        self.gauss_weights = None if gauss_weights is None else np.asarray(gauss_weights, dtype=float)

    def set_fluxes(self,
                   name: str,
                   flux: Union[np.ndarray, List[np.ndarray]],
                   gauss_weights: Optional[Union[np.ndarray, Sequence[float]]] = None):
        self.flux_name = str(name)
        if isinstance(flux, list):
            arrs = [np.asarray(f, dtype=float) for f in flux]
            n_elems = len(arrs)
            max_gp = max(a.shape[0] for a in arrs)
            F = np.full((n_elems, max_gp, 3), np.nan, dtype=float)
            for e, a in enumerate(arrs):
                F[e, :a.shape[0], :] = a
            self.flux_gauss = F
            self.flux_centroid = None
        else:
            F = np.asarray(flux, dtype=float)
            if F.ndim == 3:
                assert F.shape[2] == 3, "3D flux는 마지막 차원이 3이어야 합니다."
                self.flux_gauss = F
                self.flux_centroid = None
            elif F.ndim == 2:
                assert F.shape[1] == 3, "3D flux는 마지막 차원이 3이어야 합니다."
                self.flux_gauss = None
                self.flux_centroid = F
            else:
                raise ValueError("flux 형식 오류")
        self.flux_gauss_weights = None if gauss_weights is None else np.asarray(gauss_weights, dtype=float)

    # ---------- helpers ----------
    def _resolve_cell_type_3d(self, n: int) -> int:
        if n == 4:  return VTK_TETRA
        if n == 10: return VTK_QUADRATIC_TETRA
        if n == 8:  return VTK_HEXAHEDRON
        if n == 20: return VTK_QUADRATIC_HEXAHEDRON
        raise ValueError(f"지원하지 않는 3D 요소 노드수: {n} (지원: 4,10,8,20)")

    def _abaqus_to_vtk_order_3d(self, conn: np.ndarray) -> np.ndarray:
        """
        기본은 pass-through. 필요 시 Abaqus→VTK 노드 순서 매핑을 여기에 반영.
        일반적으로 C3D8/20, C3D4/10은 corners와 edge-mid 순서가 VTK와 동일하거나 호환됩니다.
        """
        return conn.copy()

    def _aggregate_centroid_stress6(self) -> np.ndarray:
        if self.stress_centroid is not None:
            return self.stress_centroid
        assert self.stress_gauss is not None, "집계할 가우스 응력이 없습니다."
        S = self.stress_gauss
        n_e, n_gp, n_comp = S.shape
        out = np.zeros((n_e, n_comp), dtype=float)
        for e in range(n_e):
            Se = S[e]
            mask = ~np.isnan(Se).any(axis=1)
            if not np.any(mask):
                out[e] = np.nan
                continue
            Sev = Se[mask]
            if self.gauss_weights is not None:
                if self.gauss_weights.ndim == 1 and self.gauss_weights.shape[0] == n_gp:
                    w = self.gauss_weights[mask]
                elif self.gauss_weights.ndim == 2 and self.gauss_weights.shape == (n_e, n_gp):
                    w = self.gauss_weights[e, mask]
                else:
                    w = np.ones(Sev.shape[0])
            else:
                w = np.ones(Sev.shape[0])
            wsum = w.sum()
            out[e] = (Sev * (w[:, None]/wsum)).sum(axis=0) if wsum != 0 else np.nanmean(Sev, axis=0)
        return out

    def _aggregate_centroid_flux3(self) -> np.ndarray:
        if self.flux_centroid is not None:
            return self.flux_centroid
        assert self.flux_gauss is not None, "집계할 가우스 플럭스가 없습니다."
        F = self.flux_gauss
        n_e, n_gp, _ = F.shape
        out = np.zeros((n_e, 3), dtype=float)
        for e in range(n_e):
            Fe = F[e]
            mask = ~np.isnan(Fe).any(axis=1)
            if not np.any(mask):
                out[e] = np.nan
                continue
            Fev = Fe[mask]
            if self.flux_gauss_weights is not None:
                if self.flux_gauss_weights.ndim == 1 and self.flux_gauss_weights.shape[0] == n_gp:
                    w = self.flux_gauss_weights[mask]
                elif self.flux_gauss_weights.ndim == 2 and self.flux_gauss_weights.shape == (n_e, n_gp):
                    w = self.flux_gauss_weights[e, mask]
                else:
                    w = np.ones(Fev.shape[0])
            else:
                w = np.ones(Fev.shape[0])
            wsum = w.sum()
            out[e] = (Fev * (w[:, None]/wsum)).sum(axis=0) if wsum != 0 else np.nanmean(Fev, axis=0)
        return out

    def _vm_3d(self, s6: np.ndarray) -> float:
        # s6 = [s11, s22, s33, s12, s13, s23] (engineering shear)
        s11, s22, s33, s12, s13, s23 = s6
        term = 0.5*((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2) + 3.0*(s12**2 + s13**2 + s23**2)
        return float(np.sqrt(term))

    # ---- volume & cell->point averaging (3D) ----
    def _cell_volume_3d(self, e: np.ndarray) -> float:
        xyz = self.nodes[e, :]
        n = len(e)
        vol = 0.0
        if n in (4, 10):  # Tet(quad은 코너 4개만 사용)
            a, b, c, d = xyz[0], xyz[1], xyz[2], xyz[3]
            vol = abs(np.dot((b-a), np.cross((c-a), (d-a)))) / 6.0
        elif n in (8, 20):  # Hex: 간단 테트라 분할 근사
            idx_sets = [(0,1,3,4),(1,2,3,6),(1,3,6,4),(1,6,5,4),(3,6,7,4)]
            for (i,j,k,l) in idx_sets:
                a,b,c,d = xyz[i], xyz[j], xyz[k], xyz[l]
                vol += abs(np.dot((b-a), np.cross((c-a), (d-a)))) / 6.0
        else:
            vol = 1.0
        return float(vol)

    def _cell_to_point_average_3d(self, C: np.ndarray) -> np.ndarray:
        """
        C: (n_elem, n_comp) -> (n_nodes, n_comp), volume-weighted
        """
        n_nodes = self.nodes.shape[0]
        n_comp  = C.shape[1]
        num   = np.zeros((n_nodes, n_comp), dtype=float)
        denom = np.zeros((n_nodes,), dtype=float)
        for e_idx, e in enumerate(self.elements):
            val = C[e_idx]
            if np.isnan(val).any():
                continue
            w = self._cell_volume_3d(e)
            for nid in e:
                num[nid]   += w * val
                denom[nid] += w
        out = np.full((n_nodes, n_comp), np.nan, dtype=float)
        mask = denom > 0
        out[mask] = num[mask] / denom[mask, None]
        return out

    # ---------- writer (3D) ----------
    def write_legacy_vtk(self,
                         filename: str,
                         write_U_vectors: bool = True,
                         write_U_scalars: bool = True,
                         float_fmt_coords: str = "%.15g",
                         float_fmt_data: str = "%.8e",
                         mirror_txt_path: Optional[str] = None,
                         export_point_from_cell: bool = False) -> bool:  # ★ 옵션 추가
        assert self.nodes is not None and len(self.elements) > 0, "메쉬가 설정되지 않았습니다."
        npts = self.nodes.shape[0]
        nelem = len(self.elements)
        total_ints = sum(1 + len(e) for e in self.elements)

        S_C = None
        if (self.stress_centroid is not None) or (self.stress_gauss is not None):
            S_C = self._aggregate_centroid_stress6() if self.stress_centroid is None else self.stress_centroid

        F_C = None
        if (self.flux_centroid is not None) or (self.flux_gauss is not None):
            F_C = self._aggregate_centroid_flux3() if self.flux_centroid is None else self.flux_centroid

        # ★ 노드 평균(옵션)
        S_P = None
        F_P = None
        if export_point_from_cell and (S_C is not None):
            S_P = self._cell_to_point_average_3d(S_C)
        if export_point_from_cell and (F_C is not None):
            F_P = self._cell_to_point_average_3d(F_C)

        def _w(fhs, s: str):
            for fh in fhs:
                fh.write(s)

        try:
            fhs = []
            with open(filename, 'w') as fvtk:
                fhs.append(fvtk)
                if mirror_txt_path:
                    ftxt = open(mirror_txt_path, 'w'); fhs.append(ftxt)
                else:
                    ftxt = None

                _w(fhs, "# vtk DataFile Version 3.0\n")
                _w(fhs, "3D FEA export (CENTROID + optional POINT averages)\n")
                _w(fhs, "ASCII\nDATASET UNSTRUCTURED_GRID\n")

                _w(fhs, f"POINTS {npts} float\n")
                for p in self.nodes:
                    _w(fhs, f"{float_fmt_coords % p[0]} {float_fmt_coords % p[1]} {float_fmt_coords % p[2]}\n")

                _w(fhs, f"CELLS {nelem} {total_ints}\n")
                for e in self.elements:
                    _w(fhs, str(len(e))); _w(fhs, "".join(f" {int(nid)}" for nid in e)); _w(fhs, "\n")

                _w(fhs, f"CELL_TYPES {nelem}\n")
                for ct in self.cell_types: _w(fhs, f"{ct}\n")

                # ---------- POINT_DATA ----------
                if (self.U is not None) or (len(self.point_scalars) > 0) or (S_P is not None) or (F_P is not None):
                    _w(fhs, f"POINT_DATA {npts}\n")

                    # U
                    if self.U is not None and (write_U_vectors or write_U_scalars):
                        if write_U_vectors:
                            _w(fhs, "VECTORS U float\n")
                            for u in self.U:
                                _w(fhs, f"{float_fmt_data % u[0]} {float_fmt_data % u[1]} {float_fmt_data % u[2]}\n")
                        if write_U_scalars:
                            _w(fhs, "SCALARS U1 float 1\nLOOKUP_TABLE default\n")
                            for u in self.U: _w(fhs, f"{float_fmt_data % u[0]}\n")
                            _w(fhs, "SCALARS U2 float 1\nLOOKUP_TABLE default\n")
                            for u in self.U: _w(fhs, f"{float_fmt_data % u[1]}\n")
                            _w(fhs, "SCALARS U3 float 1\nLOOKUP_TABLE default\n")
                            for u in self.U: _w(fhs, f"{float_fmt_data % u[2]}\n")

                    # 사용자 정의 POINT_SCALARS
                    for name, arr in self.point_scalars.items():
                        _w(fhs, f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
                        for v in arr: _w(fhs, f"{float_fmt_data % v}\n")

                    # ★ 응력(노드 평균)
                    if S_P is not None:
                        labels = ['S11','S22','S33','S12','S13','S23'] if S_P.shape[1] == 6 else ['S11','S22','S12']
                        for j, name in enumerate(labels):
                            _w(fhs, f"SCALARS {name}_P float 1\nLOOKUP_TABLE default\n")
                            for i in range(npts): _w(fhs, f"{float_fmt_data % S_P[i, j]}\n")
                        # Von Mises @ POINT
                        _w(fhs, "SCALARS MISES_P float 1\nLOOKUP_TABLE default\n")
                        if S_P.shape[1] == 6:
                            for i in range(npts): _w(fhs, f"{float_fmt_data % self._vm_3d(S_P[i])}\n")
                        else:
                            for i in range(npts): _w(fhs, f"{float_fmt_data % VTKExporter2D._vm_plane(self, S_P[i])}\n")

                    # ★ 플럭스(노드 평균)
                    if (F_P is not None) and self.flux_name:
                        fname = self.flux_name
                        _w(fhs, f"VECTORS {fname}_P float\n")
                        for i in range(npts):
                            v = F_P[i]
                            _w(fhs, f"{float_fmt_data % v[0]} {float_fmt_data % v[1]} {float_fmt_data % v[2]}\n")
                        _w(fhs, f"SCALARS {fname}1_P float 1\nLOOKUP_TABLE default\n")
                        for i in range(npts): _w(fhs, f"{float_fmt_data % F_P[i,0]}\n")
                        _w(fhs, f"SCALARS {fname}2_P float 1\nLOOKUP_TABLE default\n")
                        for i in range(npts): _w(fhs, f"{float_fmt_data % F_P[i,1]}\n")
                        _w(fhs, f"SCALARS {fname}3_P float 1\nLOOKUP_TABLE default\n")
                        for i in range(npts): _w(fhs, f"{float_fmt_data % F_P[i,2]}\n")
                        if fname == 'MFL':
                            _w(fhs, f"SCALARS {fname}MAG_P float 1\nLOOKUP_TABLE default\n")
                            for i in range(npts):
                                v = F_P[i]; mag = float(np.linalg.norm(v)) if not np.isnan(v).any() else np.nan
                                _w(fhs, f"{float_fmt_data % mag}\n")
                        if fname == 'HFL':
                            _w(fhs, f"SCALARS {fname}MAG_P float 1\nLOOKUP_TABLE default\n")
                            for i in range(npts):
                                v = F_P[i]; mag = float(np.linalg.norm(v)) if not np.isnan(v).any() else np.nan
                                _w(fhs, f"{float_fmt_data % mag}\n")

                # ---------- CELL_DATA ----------
                if (S_C is not None) or (F_C is not None):
                    _w(fhs, f"CELL_DATA {nelem}\n")

                if S_C is not None:
                    labels = ['S11','S22','S33','S12','S13','S23'] if S_C.shape[1] == 6 else ['S11','S22','S12']
                    for j, name in enumerate(labels):
                        _w(fhs, f"SCALARS {name}_C float 1\nLOOKUP_TABLE default\n")
                        for e in range(nelem): _w(fhs, f"{float_fmt_data % S_C[e, j]}\n")
                    _w(fhs, "SCALARS MISES_C float 1\nLOOKUP_TABLE default\n")
                    if S_C.shape[1] == 6:
                        for e in range(nelem): _w(fhs, f"{float_fmt_data % self._vm_3d(S_C[e])}\n")
                    else:
                        for e in range(nelem): _w(fhs, f"{float_fmt_data % VTKExporter2D._vm_plane(self, S_C[e])}\n")

                if (F_C is not None) and self.flux_name:
                    fname = self.flux_name  # HFL or MFL
                    _w(fhs, f"VECTORS {fname}_C float\n")
                    for e in range(nelem):
                        v = F_C[e]
                        _w(fhs, f"{float_fmt_data % v[0]} {float_fmt_data % v[1]} {float_fmt_data % v[2]}\n")
                    _w(fhs, f"SCALARS {fname}1_C float 1\nLOOKUP_TABLE default\n")
                    for e in range(nelem): _w(fhs, f"{float_fmt_data % F_C[e,0]}\n")
                    _w(fhs, f"SCALARS {fname}2_C float 1\nLOOKUP_TABLE default\n")
                    for e in range(nelem): _w(fhs, f"{float_fmt_data % F_C[e,1]}\n")
                    _w(fhs, f"SCALARS {fname}3_C float 1\nLOOKUP_TABLE default\n")
                    for e in range(nelem): _w(fhs, f"{float_fmt_data % F_C[e,2]}\n")
                    if fname == 'MFL':
                        _w(fhs, f"SCALARS {fname}MAG_C float 1\nLOOKUP_TABLE default\n")
                        for e in range(nelem):
                            v = F_C[e]; mag = float(np.linalg.norm(v)) if not np.isnan(v).any() else np.nan
                            _w(fhs, f"{float_fmt_data % mag}\n")
                    if fname == 'HFL':
                        _w(fhs, f"SCALARS {fname}MAG_C float 1\nLOOKUP_TABLE default\n")
                        for e in range(nelem):
                            v = F_C[e]; mag = float(np.linalg.norm(v)) if not np.isnan(v).any() else np.nan
                            _w(fhs, f"{float_fmt_data % mag}\n")

                if ftxt is not None: ftxt.close()

            print(f"[VTK] wrote: {filename}")
            if mirror_txt_path: print(f"[TXT] wrote: {mirror_txt_path}")
            return True
        except Exception as ex:
            print(f"[ERROR] Failed to write VTK/TXT: {ex}")
            return False


# ============================================================
# =============  FE convenience wrapper funcs  ===============
# ============================================================
# -------- 2D: Elasticity / Heat / Diffusion --------
def write_vtk_and_txt(nodes: np.ndarray,
                      elems: Union[np.ndarray, List[Sequence[int]]],
                      U: Optional[np.ndarray],
                      S_gp: Optional[Union[np.ndarray, List[np.ndarray]]],
                      prefix: str,
                      *,
                      plane_kind: str = 'stress',
                      input_order: str = 'vtk',
                      conn_base: str = 'auto',
                      gauss_weights: Optional[np.ndarray] = None,
                      write_U_vectors: bool = True,
                      write_U_scalars: bool = True,
                      mirror_txt: bool = False,
                      open_paraview_flag: bool = False,
                      paraview_path: Optional[str] = None,
                      export_point_from_cell: bool = False) -> str:
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.isdir(out_dir): os.makedirs(out_dir, exist_ok=True)

    exp = VTKExporter2D()
    exp.set_mesh(nodes, elems, input_order=input_order, conn_base=conn_base)
    if U is not None: exp.set_displacements(U)
    if S_gp is not None: exp.set_stresses(S_gp, gauss_weights=gauss_weights)

    vtk_path = f"{prefix}.vtk"; txt_path = f"{prefix}.txt" if mirror_txt else None
    ok = exp.write_legacy_vtk(vtk_path,
                              write_U_vectors=write_U_vectors,
                              write_U_scalars=write_U_scalars,
                              mirror_txt_path=txt_path,
                              export_point_from_cell=export_point_from_cell)
    if not ok: raise RuntimeError("VTK 저장에 실패했습니다.")
    if open_paraview_flag:
        try: open_in_paraview(vtk_path, paraview_path=paraview_path)
        except Exception as e: print(f"[WARN] ParaView 자동 실행 실패: {e}")
    return vtk_path

def write_vtk_and_txt_heat(nodes: np.ndarray,
                           elems: Union[np.ndarray, List[Sequence[int]]],
                           NT: np.ndarray,
                           HFL_gp: Optional[Union[np.ndarray, List[np.ndarray]]],
                           prefix: str,
                           *,
                           input_order: str = 'vtk',
                           conn_base: str = 'auto',
                           gauss_weights: Optional[np.ndarray] = None,
                           mirror_txt: bool = False,
                           open_paraview_flag: bool = False,
                           paraview_path: Optional[str] = None,
                           export_point_from_cell: bool = False) -> str:
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.isdir(out_dir): os.makedirs(out_dir, exist_ok=True)

    exp = VTKExporter2D()
    exp.set_mesh(nodes, elems, input_order=input_order, conn_base=conn_base)
    exp.add_point_scalar("NT", NT)
    if HFL_gp is not None: exp.set_fluxes("HFL", HFL_gp, gauss_weights=gauss_weights)

    vtk_path = f"{prefix}.vtk"; txt_path = f"{prefix}.txt" if mirror_txt else None
    ok = exp.write_legacy_vtk(vtk_path,
                              write_U_vectors=False,
                              write_U_scalars=False,
                              mirror_txt_path=txt_path,
                              export_point_from_cell=export_point_from_cell)
    if not ok: raise RuntimeError("VTK 저장에 실패했습니다.")
    if open_paraview_flag:
        try: open_in_paraview(vtk_path, paraview_path=paraview_path)
        except Exception as e: print(f"[WARN] ParaView 자동 실행 실패: {e}")
    return vtk_path

def write_vtk_and_txt_diffusion(nodes: np.ndarray,
                                elems: Union[np.ndarray, List[Sequence[int]]],
                                Conc: np.ndarray,
                                MFL_gp: Optional[Union[np.ndarray, List[np.ndarray]]],
                                prefix: str,
                                *,
                                input_order: str = 'vtk',
                                conn_base: str = 'auto',
                                gauss_weights: Optional[np.ndarray] = None,
                                mirror_txt: bool = False,
                                open_paraview_flag: bool = False,
                                paraview_path: Optional[str] = None,
                                export_point_from_cell: bool = False) -> str:
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.isdir(out_dir): os.makedirs(out_dir, exist_ok=True)

    exp = VTKExporter2D()
    exp.set_mesh(nodes, elems, input_order=input_order, conn_base=conn_base)
    exp.add_point_scalar("Conc", Conc)
    if MFL_gp is not None: exp.set_fluxes("MFL", MFL_gp, gauss_weights=gauss_weights)

    vtk_path = f"{prefix}.vtk"; txt_path = f"{prefix}.txt" if mirror_txt else None
    ok = exp.write_legacy_vtk(vtk_path,
                              write_U_vectors=False,
                              write_U_scalars=False,
                              mirror_txt_path=txt_path,
                              export_point_from_cell=export_point_from_cell)
    if not ok: raise RuntimeError("VTK 저장에 실패했습니다.")
    if open_paraview_flag:
        try: open_in_paraview(vtk_path, paraview_path=paraview_path)
        except Exception as e: print(f"[WARN] ParaView 자동 실행 실패: {e}")
    return vtk_path

# -------- 3D: Elasticity / Heat / Diffusion --------
def write_vtk_and_txt_elasticity_3d(nodes: np.ndarray,
                                    elems: Union[np.ndarray, List[Sequence[int]]],
                                    U: Optional[np.ndarray],
                                    S_gp: Optional[Union[np.ndarray, List[np.ndarray]]],
                                    prefix: str,
                                    *,
                                    input_order: str = 'vtk',
                                    conn_base: str = 'auto',
                                    gauss_weights: Optional[np.ndarray] = None,
                                    write_U_vectors: bool = True,
                                    write_U_scalars: bool = True,
                                    mirror_txt: bool = False,
                                    open_paraview_flag: bool = False,
                                    paraview_path: Optional[str] = None,
                                    export_point_from_cell: bool = False) -> str:
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.isdir(out_dir): os.makedirs(out_dir, exist_ok=True)

    exp = VTKExporter3D()
    exp.set_mesh(nodes, elems, input_order=input_order, conn_base=conn_base)
    if U is not None: exp.set_displacements(U)
    if S_gp is not None: exp.set_stresses(S_gp, gauss_weights=gauss_weights)

    vtk_path = f"{prefix}.vtk"; txt_path = f"{prefix}.txt" if mirror_txt else None
    ok = exp.write_legacy_vtk(vtk_path,
                              write_U_vectors=write_U_vectors,
                              write_U_scalars=write_U_scalars,
                              mirror_txt_path=txt_path,
                              export_point_from_cell=export_point_from_cell)
    if not ok: raise RuntimeError("VTK 저장에 실패했습니다.")
    if open_paraview_flag:
        try: open_in_paraview(vtk_path, paraview_path=paraview_path)
        except Exception as e: print(f"[WARN] ParaView 자동 실행 실패: {e}")
    return vtk_path

# -------- 3D: Heat / Diffusion --------
def write_vtk_and_txt_heat_3d(nodes: np.ndarray,
                              elems: Union[np.ndarray, List[Sequence[int]]],
                              NT: np.ndarray,
                              HFL_gp: Optional[Union[np.ndarray, List[np.ndarray]]],
                              prefix: str,
                              *,
                              input_order: str = 'vtk',
                              conn_base: str = 'auto',
                              gauss_weights: Optional[np.ndarray] = None,
                              mirror_txt: bool = False,
                              open_paraview_flag: bool = False,
                              paraview_path: Optional[str] = None,
                              export_point_from_cell: bool = True) -> str:  # ★ default True
    """
    3D Heat Transfer exporter
    - nodes: (NoN,3)
    - elems: connectivity
    - NT: nodal temperature
    - HFL_gp: (nElem, nGp, 3) heat flux at Gauss points
    """
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.isdir(out_dir): os.makedirs(out_dir, exist_ok=True)

    exp = VTKExporter3D()
    exp.set_mesh(nodes, elems, input_order=input_order, conn_base=conn_base)
    exp.add_point_scalar("NT", NT)
    if HFL_gp is not None:
        exp.set_fluxes("HFL", HFL_gp, gauss_weights=gauss_weights)

    vtk_path = f"{prefix}.vtk"; txt_path = f"{prefix}.txt" if mirror_txt else None
    ok = exp.write_legacy_vtk(vtk_path,
                              write_U_vectors=False,
                              write_U_scalars=False,
                              mirror_txt_path=txt_path,
                              export_point_from_cell=export_point_from_cell)
    if not ok: raise RuntimeError("VTK 저장에 실패했습니다.")
    if open_paraview_flag:
        try: open_in_paraview(vtk_path, paraview_path=paraview_path)
        except Exception as e: print(f"[WARN] ParaView 자동 실행 실패: {e}")
    return vtk_path


def write_vtk_and_txt_diffusion_3d(nodes: np.ndarray,
                                   elems: Union[np.ndarray, List[Sequence[int]]],
                                   Conc: np.ndarray,
                                   MFL_gp: Optional[Union[np.ndarray, List[np.ndarray]]],
                                   prefix: str,
                                   *,
                                   input_order: str = 'vtk',
                                   conn_base: str = 'auto',
                                   gauss_weights: Optional[np.ndarray] = None,
                                   mirror_txt: bool = False,
                                   open_paraview_flag: bool = False,
                                   paraview_path: Optional[str] = None,
                                   export_point_from_cell: bool = True) -> str:  # ★ default True
    """
    3D Diffusion exporter
    - nodes: (NoN,3)
    - elems: connectivity
    - Conc: nodal concentration
    - MFL_gp: (nElem, nGp, 3) mass flux at Gauss points
    """
    out_dir = os.path.dirname(prefix)
    if out_dir and not os.path.isdir(out_dir): os.makedirs(out_dir, exist_ok=True)

    exp = VTKExporter3D()
    exp.set_mesh(nodes, elems, input_order=input_order, conn_base=conn_base)
    exp.add_point_scalar("Conc", Conc)
    if MFL_gp is not None:
        exp.set_fluxes("MFL", MFL_gp, gauss_weights=gauss_weights)

    vtk_path = f"{prefix}.vtk"; txt_path = f"{prefix}.txt" if mirror_txt else None
    ok = exp.write_legacy_vtk(vtk_path,
                              write_U_vectors=False,
                              write_U_scalars=False,
                              mirror_txt_path=txt_path,
                              export_point_from_cell=export_point_from_cell)
    if not ok: raise RuntimeError("VTK 저장에 실패했습니다.")
    if open_paraview_flag:
        try: open_in_paraview(vtk_path, paraview_path=paraview_path)
        except Exception as e: print(f"[WARN] ParaView 자동 실행 실패: {e}")
    return vtk_path
