# -*- coding: utf-8 -*-
"""
Assembly_and_solve_eq_export_inp.py  (Strict-List + Legacy/Debug Edition)

Goal
----
Keep your original "everything in one file" style while:
- Guaranteeing **strict-list** behavior for Neumann (seed_nodes define faces 1:1).
- Preserving **legacy** boundary-detection paths and debuggers (CSV exporters, etc.).
- Supporting **all three physics** (Elasticity / Heat / Diffusion) in **2D/3D**.
- Providing clear diagnostics and text reports under ./vtk_out or ./bc_reports.

Key Notes
---------
1) Strict-List mode (preferred):
   - If an item in *list* has "seed_nodes" (or "nodes") of length 3/4,
     we **integrate that face directly** without boundary ownership checks.
     => **len(list) == number of integrated faces**.
   - Heat: {"seed_nodes":[n1,n2,n3(,n4)], "qn": 50}
   - Diffusion: {"seed_nodes":[...], "jn": 1.2}
   - Elasticity: {"seed_nodes":[...], "traction":[tx,ty,tz]} or {"pressure":p}
   - Order of seed_nodes is ignored for area/centroid computation.
   - For pressure, the face normal is the geometric normal from the node ordering; no outward check.

2) Legacy/sets mode:
   - If you use "nodes_txt"/"nodes"/"sets" (without seed_nodes triplets),
     we still have the robust old-style boundary face detection (ownership=1)
     and selection by nodeset containment. Use CSV exporters for verification.

3) Debug Reports:
   - heat_dirichlet.txt, heat_neumann.txt, diffusion_dirichlet.txt …
   - elasticity_neumann.txt, elasticity_dirichlet.txt
   - Optional CSV dumps of faces to help audit.

Author: changsub + assistant (2025-10-16)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, Iterable
from pathlib import Path
import numpy as np
import math
import os
import re
import csv

# External project dependencies (your existing modules)
from Preprocessing_export_inp import Mesh, Material_Property  # noqa: F401
from Element_formulation_export_inp import Gauss_Point, Shape_Function  # noqa: F401


# ============================================================================
# Generic helpers & geometry
# ============================================================================

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


class OneD_gauss_quadrature:
    """1D Gauss rule for 2D edge integration."""
    def __init__(self, n_gp: int, element_shape: str):
        self.n_gp = int(n_gp)
        if self.n_gp == 1:
            self.points = [0.0]; self.weights = [2.0]
        elif self.n_gp == 2:
            a = 1.0 / math.sqrt(3.0)
            self.points = [-a, a]; self.weights = [1.0, 1.0]
        elif self.n_gp == 3:
            a = math.sqrt(3.0/5.0)
            self.points = [-a, 0.0, a]; self.weights = [5/9, 8/9, 5/9]
        else:
            raise ValueError("Unsupported 1D Gauss n_gp (use 1,2,3).")
    def get_quadrature(self):
        return self.points, self.weights


def tet_local_faces() -> List[List[int]]:
    return [[0,1,2], [0,1,3], [1,2,3], [0,2,3]]


def hex_local_faces() -> List[List[int]]:
    return [
        [0,1,2,3], [4,5,6,7],
        [0,1,5,4], [1,2,6,5],
        [2,3,7,6], [3,0,4,7],
    ]


def element_face_topology(npe: int, etype_hint: Optional[str] = None) -> Tuple[List[List[int]], List[str]]:
    """
    Return (faces, face_shapes) for element with `npe` nodes.
      faces: list of local node index sets (per face)
      face_shapes: "Tri" or "Quad"
    Supports TET(4/10), HEX(8/20/27), WEDGE(6/15), PYRAMID(5/13). Falls back to HEX or TET.
    """
    et = (etype_hint or "").upper()

    # TET4/10
    if et.startswith("C3D4") or et.startswith("C3D10") or (npe in (4,10) and "TET" in et):
        faces = [[0,1,2], [0,1,3], [1,2,3], [0,2,3]]
        shapes = ["Tri"] * 4
        return faces, shapes

    # HEX8/20/27
    if et.startswith("C3D8") or et.startswith("C3D20") or npe in (8,20,27):
        faces = [
            [0,1,2,3], [4,5,6,7],
            [0,1,5,4], [1,2,6,5],
            [2,3,7,6], [3,0,4,7],
        ]
        shapes = ["Quad"] * 6
        return faces, shapes

    # WEDGE6/15
    if et.startswith("C3D6") or "WEDGE" in et or npe in (6,15):
        faces = [
            [0,1,2],      # Tri
            [3,4,5],      # Tri
            [0,1,4,3],    # Quad
            [1,2,5,4],    # Quad
            [2,0,3,5],    # Quad
        ]
        shapes = ["Tri","Tri","Quad","Quad","Quad"]
        return faces, shapes

    # PYRAMID5/13
    if et.startswith("C3D5") or "PYRAMID" in et or npe in (5,13):
        faces = [
            [0,1,2,3],    # Quad base
            [0,1,4], [1,2,4], [2,3,4], [3,0,4]
        ]
        shapes = ["Quad","Tri","Tri","Tri","Tri"]
        return faces, shapes

    # Fall back
    if npe >= 8:
        return hex_local_faces(), ["Quad"]*6
    else:
        return tet_local_faces(), ["Tri"]*4


# ---------- indexing helpers ----------
def _infer_conn_is_row0(conn, NoN) -> bool:
    sample = []
    for c in conn[:min(20, len(conn))]:
        c = np.asarray(c).astype(int)
        sample.extend(c[:min(8, len(c))].tolist())
    if not sample: return True
    if any(v == 0 for v in sample): return True
    mn, mx = min(sample), max(sample)
    if mn >= 1 and mx <= NoN: return False
    return True


def _label_to_row_func(mesh):
    id2row = getattr(mesh, "id2row", None)
    if id2row is None:
        return lambda n1: int(n1) - 1
    if isinstance(id2row, dict):
        return lambda n1: int(id2row.get(int(n1), int(n1)-1))
    else:
        return lambda n1: int(id2row[int(n1)])


def _row_to_label_func(mesh):
    row2id = getattr(mesh, "row2id", None)
    if row2id is None:
        return lambda ir: int(ir) + 1
    if isinstance(row2id, dict):
        return lambda ir: int(row2id.get(int(ir), int(ir)+1))
    else:
        return lambda ir: int(row2id[int(ir)])


def _conn_rows(mesh, conn, eidx: int, NoN: int) -> np.ndarray:
    c = np.asarray(conn[eidx], int)
    if _infer_conn_is_row0(conn, NoN):
        return c
    id2row = getattr(mesh, "id2row", None)
    if id2row is not None:
        if isinstance(id2row, dict):
            return np.asarray([int(id2row.get(int(v), int(v)-1)) for v in c], int)
        else:
            return np.asarray([int(id2row[int(v)]) for v in c], int)
    return c - 1


# ---------- geometry ----------
def tri_area_normal_centroid(xa: np.ndarray, xb: np.ndarray, xc: np.ndarray):
    v1 = xb - xa
    v2 = xc - xa
    n  = np.cross(v1, v2)
    A2 = np.linalg.norm(n)
    if A2 == 0.0:
        return 0.0, np.array([0.0,0.0,0.0]), xa
    n_hat = n / A2
    area = 0.5 * A2
    ctr  = (xa + xb + xc) / 3.0
    return area, n_hat, ctr


def polygon_area_normal_centroid(X: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute area and (rough) unit normal & centroid of a planar polygon X (m x 3).
    Fan-triangulate around vertex 0.
    """
    m = X.shape[0]
    if m < 3:
        return 0.0, np.array([0.0,0.0,0.0]), X.mean(axis=0)
    n_vec = np.array([0.0,0.0,0.0], dtype=float)
    area_total = 0.0
    ctr_acc = np.array([0.0,0.0,0.0], dtype=float)
    for i in range(1, m-1):
        a = X[i] - X[0]
        b = X[i+1] - X[0]
        c = np.cross(a, b)
        tri_area = 0.5 * np.linalg.norm(c)
        area_total += tri_area
        n_vec += c
        # centroid of triangle (0,i,i+1)
        tri_ctr = (X[0] + X[i] + X[i+1]) / 3.0
        ctr_acc += tri_ctr * tri_area
    n_norm = np.linalg.norm(n_vec)
    n_hat = (n_vec / n_norm) if n_norm > 0 else n_vec
    centroid = (ctr_acc / max(area_total, 1e-30)) if area_total > 0 else X.mean(axis=0)
    return float(area_total), n_hat, centroid


def _area_of_nodes(NL, rows):
    """
    rows: node row indices (0-based), len=3 or 4.
    Return (area, centroid, unit_normal).
    """
    X = np.asarray(NL[rows, :3], float)
    m = X.shape[0]
    if m == 3:
        a = X[1] - X[0]; b = X[2] - X[0]
        n = np.cross(a, b); A = 0.5 * np.linalg.norm(n)
        n_hat = n / (np.linalg.norm(n) if np.linalg.norm(n) > 0 else 1.0)
        ctr  = X.mean(axis=0)
        return float(A), ctr, n_hat
    elif m == 4:
        n1 = np.cross(X[1]-X[0], X[2]-X[0])
        n2 = np.cross(X[2]-X[0], X[3]-X[0])
        A = 0.5 * (np.linalg.norm(n1) + np.linalg.norm(n2))
        n = n1 + n2
        n_hat = n / (np.linalg.norm(n) if np.linalg.norm(n) > 0 else 1.0)
        ctr  = X.mean(axis=0)
        return float(A), ctr, n_hat
    else:
        raise ValueError("seed_nodes must have length 3 (Tri) or 4 (Quad)")


# ---------- nodes/range parsers ----------
def parse_range_string(s: str) -> List[int]:
    s = s.strip()
    if "-" in s and ":" not in s:
        a, b = [int(x.strip()) for x in s.split("-", 1)]
        step = 1 if a <= b else -1
        return list(range(a, b + (1 if step > 0 else -1), step))
    if ":" in s:
        parts = [int(p.strip()) for p in s.split(":")]
        if len(parts) == 2: a, b = parts; step = 1 if a <= b else -1
        elif len(parts) == 3:
            a, b, step = parts
            if step == 0: step = 1
            if (step > 0 and a > b) or (step < 0 and a < b): step = -step
        else:
            raise ValueError(f"Bad range: {s}")
        return list(range(a, b + (1 if step > 0 else -1), step))
    return [int(s)]


def expand_nodes_spec(spec) -> List[int]:
    nodes = []
    def push(v):
        if isinstance(v, (list, tuple, set)):
            for item in v: push(item)
        elif isinstance(v, dict):
            a = int(v.get("from")); b = int(v.get("to"))
            step = int(v.get("step", 1 if a <= b else -1))
            nodes.extend(range(a, b + (1 if step > 0 else -1), step))
        elif isinstance(v, int):
            nodes.append(v)
        elif isinstance(v, str):
            nodes.extend(parse_range_string(v))
        else:
            raise TypeError(f"Unsupported node spec type: {type(v)} -> {v}")
    push(spec)
    uniq = sorted(set(int(n) for n in nodes))
    return uniq


def parse_nodes_txt_simple(path: str) -> List[int]:
    """Simple node txt parser: numbers, ranges a-b, a:b[:s], commas/spaces/tabs, comments '# ...'"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"node txt not found: {path}")
    txt = p.read_text(encoding="utf-8", errors="ignore")
    txt = "\n".join(line.split("#", 1)[0] for line in txt.splitlines())
    txt = txt.replace(",", " ").replace("\t", " ")
    tokens = [t for t in re.split(r"\s+", txt) if t]
    out = []
    for t in tokens:
        if re.fullmatch(r"\d+", t):
            out.append(int(t))
        elif re.fullmatch(r"\d+\-\d+", t):
            a,b=[int(x) for x in t.split("-",1)]; step=1 if a<=b else -1
            out.extend(range(a,b+(1 if step>0 else -1), step))
        elif re.fullmatch(r"\d+:\d+(?::\-?\d+)?", t):
            parts=[int(x) for x in t.split(":")]
            if len(parts)==2:
                a,b=parts; step=1 if a<=b else -1
            else:
                a,b,step=parts
                if step==0: step=1
                if (step>0 and a>b) or (step<0 and a<b): step=-step
            out.extend(range(a,b+(1 if step>0 else -1), step))
        else:
            t2=t.strip().strip(",")
            if t2.isdigit(): out.append(int(t2))
            else: raise ValueError(f"bad token: {t}")
    return sorted(set(out))


# ============================================================================
# 2D boundary edges meta (for elasticity 2D)
# ============================================================================

def quad_local_edges_2d(npe: int) -> List[Tuple[int, int]]:
    return [(0,1), (1,2), (2,3), (3,0)]

def tri_local_edges_2d(npe: int) -> List[Tuple[int, int]]:
    return [(0,1), (1,2), (2,0)]

def edges_midside_indices_2d(npe: int, shape: str) -> List[List[int]]:
    if shape == "Quad":  return [[4],[5],[6],[7]] if npe >= 8 else [[],[],[],[]]
    if shape == "Tri":   return [[3],[4],[5]] if npe >= 6 else [[],[],[]]
    return []


def build_boundary_edges_meta_2d(NL, conn, NoN, NPE, element_shape, row2id=None):
    """Collect boundary edges meta for Quad/Tri (for elasticity 2D tractions)."""
    element_shape = str(element_shape).capitalize()
    edge_map = {}
    for eidx, c in enumerate(conn):
        c = np.asarray(c, int)
        if element_shape == "Quad":
            eds = quad_local_edges_2d(len(c))
        else:
            eds = tri_local_edges_2d(len(c))
        for k, (i, j) in enumerate(eds):
            a, b = int(c[i]), int(c[j])
            key = tuple(sorted((a, b)))
            edge_map.setdefault(key, []).append((eidx, k))

    boundary = [(owners[0][0], owners[0][1]) for owners in edge_map.values() if len(owners) == 1]
    metas = {}
    NL = np.asarray(NL, float)

    def row_to_label(rr: int) -> int:
        if row2id is None:
            return int(rr) + 1
        if isinstance(row2id, dict):
            return int(row2id.get(int(rr), int(rr)+1))
        return int(row2id[int(rr)])

    for (eidx, k) in boundary:
        c = np.asarray(conn[eidx], int)
        ed_pairs = quad_local_edges_2d(len(c)) if (element_shape == "Quad") else tri_local_edges_2d(len(c))
        i, j = ed_pairs[k]
        a, b = int(c[i]), int(c[j])
        xa, xb = NL[a, :2], NL[b, :2]
        mid = 0.5*(xa + xb)
        ctr = NL[c, :2].mean(axis=0)
        t = xb - xa
        n = np.array([-t[1], t[0]], dtype=float)
        L = float(np.linalg.norm(t))
        if L == 0.0:
            n_hat = np.array([0.0, 0.0])
        else:
            n_hat = n / (np.linalg.norm(n) if np.linalg.norm(n) > 0 else 1.0)
            if float(np.dot(mid - ctr, n_hat)) < 0.0:
                n_hat = -n_hat

        nodes_rows = [a, b]
        nodes1 = [row_to_label(r) for r in nodes_rows]

        metas[(eidx, k)] = dict(
            nodes1=nodes1,
            corners_rows=[a, b],
            mid=mid.reshape(2,),
            normal=n_hat.reshape(2,),
            length=max(L, 1e-20),
        )
    return metas


# ============================================================================
# Linear Elasticity
# ============================================================================

class Linear_elasticity_BC:
    """
    - Dirichlet via list/sets
    - Neumann (traction/pressure) strict-list support + legacy
      Strict-list:
        {"seed_nodes":[n1,n2,n3(,n4)], "traction":[tx,ty,tz]}  # vector traction (N/m^2)
        {"seed_nodes":[...], "pressure": p}                     # scalar pressure (N/m^2), along face normal
    """
    def __init__(self, system: str, mesh: Mesh,
                 element_shape: str, element_order: str,
                 load: str, n_gp: Optional[int],
                 analytical_conditions: str, integration: str,
                 Dimension: str,
                 dirichlet_list_3d: Optional[list] = None,
                 dirichlet_sets_3d: Optional[dict] = None,
                 traction_mode: str = "off",
                 traction_list_3d: Optional[list] = None,
                 traction_sets_3d: Optional[dict] = None,
                 concentrated_force_list_3d: Optional[list] = None):
        self.system = system
        self.mesh = mesh
        self.NL   = mesh.NL
        self.conn = mesh.conn
        self.NoN  = mesh.NoN
        self.NPE  = mesh.NPE
        self.Dimension = Dimension
        self.element_shape = element_shape
        self.element_order = element_order
        self.integration   = integration
        self.analytical_conditions = analytical_conditions
        self._concentrated_force_list_3d = concentrated_force_list_3d

        # reports dir
        self.bc_report_dir = Path("bc_reports"); ensure_dir(self.bc_report_dir / "dummy.txt")
        self.neumann_log_path = str(self.bc_report_dir / "elasticity_neumann_legacy.txt")

        self.sf = Shape_Function(analytical_conditions, element_order, element_shape, integration)
        self.edge_gp_pts, self.edge_gp_wts = OneD_gauss_quadrature(n_gp or 2, element_shape).get_quadrature()

        if Dimension == "2D":
            self.F = np.zeros((2*self.NoN, 1)); self.DoF = np.arange(2*self.NoN)
        else:
            self.F = np.zeros((3*self.NoN, 1)); self.DoF = np.arange(3*self.NoN)

        # Dirichlet inputs
        self._dirichlet_list_3d = dirichlet_list_3d or []
        self._dirichlet_sets_3d = dirichlet_sets_3d or {}
        self._dirichlet_manual_3d = {}

        # Neumann inputs
        self._traction_mode   = (traction_mode or "off")
        self._traction_list_3d = traction_list_3d or []
        self._traction_sets_3d = traction_sets_3d or {}
        self._pressure_sign = -1.0  # Abaqus sign

        self._conn_is_row0 = _infer_conn_is_row0(self.conn, self.NoN)

        # apply Dirichlet mapping
        self.Dirichlet_BC(load)

    def assemble_concentrated_forces(self, load: Optional[str] = None) -> np.ndarray:
        """
        집중하중(노드 힘, 단위 N)을 글로벌 RHS에 더한다.
        지원 포맷:
          {"nodes":[...], "F":[Fx,Fy,Fz]}  # 여러 노드에 동일 벡터
          {"node": n,   "F":[Fx,Fy,Fz]}    # 단일 노드
        """
        if self.Dimension == "2D":
            F = np.zeros((2*self.NoN, 1))
        else:
            F = np.zeros((3*self.NoN, 1))

        items = self._concentrated_force_list_3d or []
        if not items:
            return F

        label2row = _label_to_row_func(self.mesh)
        used = 0; total = np.zeros(3 if self.Dimension == "3D" else 2)

        def _accum(nid: int, vec: np.ndarray):
            r = label2row(nid)
            if self.Dimension == "2D":
                F[2*r+0,0] += vec[0]; F[2*r+1,0] += vec[1]
            else:
                F[3*r+0,0] += vec[0]; F[3*r+1,0] += vec[1]; F[3*r+2,0] += vec[2]

        for i, spec in enumerate(items):
            if "F" not in spec or spec["F"] is None:
                continue
            Fvec = np.asarray(spec["F"], float).reshape(-1,)
            if self.Dimension == "2D":
                if Fvec.size == 3: Fvec = Fvec[:2]
                elif Fvec.size == 2: pass
                else: continue
            else:
                if Fvec.size == 2: Fvec = np.array([Fvec[0], Fvec[1], 0.0])
                elif Fvec.size == 3: pass
                else: continue

            if "nodes" in spec and spec["nodes"]:
                for nid in spec["nodes"]:
                    _accum(int(nid), Fvec); used += 1; total[:Fvec.size] += Fvec
            elif "node" in spec:
                _accum(int(spec["node"]), Fvec); used += 1; total[:Fvec.size] += Fvec
            else:
                continue

        print(f"[Elasticity/Concentrated] applied={used}, total={total}")
        # 간단 리포트(선택): bc_reports/elasticity_concentrated.txt
        try:
            p = Path("bc_reports"); ensure_dir(p / "dummy.txt")
            (p / "elasticity_concentrated.txt").write_text(
                f"count,{used}\n"
                f"total,{','.join(f'{v:.16e}' for v in total)}\n",
                encoding="utf-8"
            )
        except Exception:
            pass

        return F   
    
    # Linear_elasticity_BC 내부에 추가
    def assemble_edge_tractions_2d(self, thickness: float = 1.0) -> np.ndarray:
        """2D Quad/Tri 요소 경계(edge)에 작용하는 traction/pressure를
        선적분하여 글로벌 힘 벡터를 구성한다.

        - self._traction_list_3d 에는 다음과 같은 항목들이 들어 있다고 가정:
          {"nodes":[n1,n2,...], "pressure": p}      # 스칼라 압력, 법선 방향으로 작용
          {"nodes":[...], "traction":[tx,ty,(tz)]}  # 평면 내 등분포 traction
        - thickness: out-of-plane 두께 (plane stress면 실제 두께, plane strain이면 통상 1.0)
        """
        F = np.zeros((2*self.NoN, 1))
        if self.Dimension != "2D":
            return F
        if (self._traction_mode or "off") == "off":
            return F

        # 2D 경계 edge 메타 정보: 각 edge의 노드 라벨, row index, 길이, 법선 등
        metas = build_boundary_edges_meta_2d(
            self.NL, self.conn, self.NoN, self.NPE,
            self.element_shape, getattr(self.mesh, "row2id", None)
        )
        if not metas:
            return F

        # traction_list_3d 기반 edge 찾기
        for spec in (self._traction_list_3d or []):
            if not isinstance(spec, dict):
                continue
            spec_local = dict(spec)

            # 적용 노드 집합 (node 라벨 기준)
            nodes_labels = []
            if "nodes" in spec_local and spec_local["nodes"] is not None:
                nodes_labels = [int(n) for n in spec_local["nodes"]]
            elif "seed_nodes" in spec_local and spec_local["seed_nodes"] is not None:
                nodes_labels = [int(n) for n in spec_local["seed_nodes"]]
            if not nodes_labels:
                continue
            nodes_set = set(nodes_labels)

            has_trac = ("traction" in spec_local) and (spec_local["traction"] is not None)
            has_p    = ("pressure" in spec_local) and (spec_local["pressure"] is not None)
            if not (has_trac or has_p):
                continue

            for meta in metas.values():
                n1, n2 = meta["nodes1"]          # 노드 라벨
                if (n1 not in nodes_set) or (n2 not in nodes_set):
                    continue
                rows = meta["corners_rows"]      # NL row index (0-based)
                L = float(meta["length"])
                if L <= 0.0:
                    continue

                if has_trac:
                    tvec = np.asarray(spec_local["traction"], float).reshape(-1,)
                    if tvec.size < 2:
                        continue
                    tx, ty = float(tvec[0]), float(tvec[1])
                else:
                    p = float(spec_local["pressure"])
                    n_hat = np.asarray(meta["normal"], float).reshape(-1,)
                    if n_hat.size < 2:
                        continue
                    # p>0 을 외향 법선 방향으로 작용하는 압력으로 해석
                    tx, ty = float(p * n_hat[0]), float(p * n_hat[1])

                # 등분포 선하중: 각 코너 노드에 t * L/2 배분 (∫ N_i ds = L/2)
                for r in rows:
                    F[2*r + 0, 0] += tx * thickness * (L * 0.5)
                    F[2*r + 1, 0] += ty * thickness * (L * 0.5)

        return F



    # ---------- Dirichlet assembly ----------
    def Dirichlet_BC(self, load: str):
        if self.Dimension == "2D":
            dof_map = {}
            for tpl in (self._dirichlet_list_3d or []):
                if len(tpl) >= 3:
                    node1, ux, uy = tpl[0], tpl[1], tpl[2]
                    n0 = int(node1) - 1
                    if 0 <= n0 < self.NoN:
                        base = 2*n0
                        if ux is not None: dof_map[base+0] = float(ux)
                        if uy is not None: dof_map[base+1] = float(uy)

            for set_name, cfg in (self._dirichlet_sets_3d or {}).items():
                if "nodes" not in cfg: continue
                ux = cfg.get("ux", None); uy = cfg.get("uy", None)
                nodes = cfg["nodes"]
                if isinstance(nodes, str): nlist = parse_nodes_txt_simple(nodes)
                else: nlist = [int(v) for v in nodes]
                for n1 in nlist:
                    r = _label_to_row_func(self.mesh)(n1)
                    if 0 <= r < self.NoN:
                        base = 2*r
                        if ux is not None: dof_map[base+0] = float(ux)
                        if uy is not None: dof_map[base+1] = float(uy)

            self.DoEs = sorted(dof_map.keys())
            self.d_E  = np.array([dof_map[d] for d in self.DoEs], dtype=float)
            self.DoFs = [i for i in self.DoF if i not in self.DoEs]
            print(f"[Dirichlet/2D] DoEs={len(self.DoEs)}, DoFs={len(self.DoFs)}")
            return

        # 3D
        dof_map = {}
        for (node1, ux, uy, uz) in (self._dirichlet_list_3d or []):
            n0 = int(node1) - 1
            if n0 < 0 or n0 >= self.NoN: continue
            base = 3*n0
            if ux is not None: dof_map[base + 0] = float(ux)
            if uy is not None: dof_map[base + 1] = float(uy)
            if uz is not None: dof_map[base + 2] = float(uz)

        for set_name, cfg in (self._dirichlet_sets_3d or {}).items():
            if "nodes" not in cfg: continue
            ux = cfg.get("ux", None); uy = cfg.get("uy", None); uz = cfg.get("uz", None)
            nodes = cfg["nodes"]
            if isinstance(nodes, str): nlist = parse_nodes_txt_simple(nodes)
            else: nlist = [int(v) for v in nodes]
            for n1 in nlist:
                r = _label_to_row_func(self.mesh)(n1)
                if 0 <= r < self.NoN:
                    base = 3*r
                    if ux is not None: dof_map[base + 0] = float(ux)
                    if uy is not None: dof_map[base + 1] = float(uy)
                    if uz is not None: dof_map[base + 2] = float(uz)

        self.DoEs = sorted(dof_map.keys())
        self.d_E  = np.array([dof_map[d] for d in self.DoEs], dtype=float)
        self.DoFs = [i for i in self.DoF if i not in self.DoEs]
        print(f"[Dirichlet/3D] DoEs={len(self.DoEs)}, DoFs={len(self.DoFs)}")

    # ---------- 3D: strict-list tractions/pressure ----------
    def assemble_surface_tractions_3d(self) -> np.ndarray:
        F = np.zeros((3*self.NoN, 1))
        if self.Dimension != "3D":
            return F
        if (self._traction_mode or "off") == "off":
            return F

        # strict-list?
        use_strict_list = False
        if self._flux_mode in ("list", "both") and self._flux_list_3d:
            use_strict_list = True

        if use_strict_list:
            label2row = _label_to_row_func(self.mesh)
            used = 0; skipped = 0
            for cfg in (self._traction_list_3d or []):
                seeds = cfg.get("seed_nodes", cfg.get("nodes", None))
                if seeds is None: continue
                try: nodes1 = [int(v) for v in seeds]
                except Exception: continue
                if len(nodes1) not in (3,4): skipped += 1; continue

                tvec = None; p = None
                if "traction" in cfg and cfg["traction"] is not None:
                    arr = np.asarray(cfg["traction"], float).reshape(3,)
                    tvec = arr
                elif "pressure" in cfg and cfg["pressure"] is not None:
                    p = float(cfg["pressure"])
                else:
                    skipped += 1; continue

                rows = [label2row(n1) for n1 in nodes1]
                if any((r<0 or r>=self.NoN) for r in rows):
                    skipped += 1; continue

                area, ctr, n_hat = _area_of_nodes(self.NL, rows)
                if area <= 0.0: skipped += 1; continue

                if tvec is None:
                    # pressure → vector along geometric normal
                    tvec = self._pressure_sign * p * n_hat  # sign follow Abaqus conv.

                # nodal equivalent force
                fe = (area / len(rows)) * tvec.reshape(3,)
                for r in rows:
                    F[3*r+0,0] += fe[0]
                    F[3*r+1,0] += fe[1]
                    F[3*r+2,0] += fe[2]
                used += 1

            print(f"[Elasticity/Neumann/strict-list] faces={used}, skipped={skipped}")
            return F

        # Legacy path omitted (user prefers strict list). Keep noop to avoid surprises.
        print("[Elasticity] no strict-list tractions provided; nothing assembled (legacy path skipped).")
        return F


# ============================================================================
# Heat Transfer
# ============================================================================

class Heat_transfer_BC:
    """
    Heat transfer BC/loads
    - Dirichlet(T): list / set
    - Neumann(qn): strict-list first (guaranteed 1:1), legacy nodeset fallback optional
    - Source(qv): elements
    """
    def __init__(self, system: str, mesh,
                 element_shape: str, element_order: str,
                 load: str, n_gp: Optional[int],
                 analytical_conditions: str, integration: str,
                 boundary_heat_flux: str, source_heat_flux: str,
                 Dimension: str,
                 dirichlet_mode: str = "off",
                 dirichlet_list_3d: Optional[list] = None,
                 dirichlet_sets_3d: Optional[dict] = None,
                 flux_mode: str = "off",
                 flux_list_3d: Optional[list] = None,
                 flux_sets_3d: Optional[dict] = None,
                 source_mode: str = "off",
                 source_list_3d: Optional[list] = None,
                 source_sets_3d: Optional[dict] = None):
        self.system = system
        self.mesh = mesh
        self.NL   = mesh.NL
        self.conn = mesh.conn
        self.NoN  = mesh.NoN
        self.NPE  = mesh.NPE
        self.element_shape = element_shape
        self.element_order = element_order
        self.integration   = integration
        self.analytical_conditions = analytical_conditions
        self.Dimension = Dimension

        self.sf = Shape_Function(analytical_conditions, element_order, element_shape, integration)

        self.F = np.zeros((self.NoN, 1))
        self.boundary_heat_flux = boundary_heat_flux  # "O"/"X"
        self.source_heat_flux   = source_heat_flux    # "O"/"X"

        # User inputs
        self._dirichlet_mode = (dirichlet_mode or "off")
        self._dirichlet_list_3d  = dirichlet_list_3d or []    # [(node_label, T)]
        self._dirichlet_sets_3d  = dirichlet_sets_3d or {}

        self._flux_mode   = (flux_mode or "off")
        self._flux_list_3d = flux_list_3d or []
        self._flux_sets_3d = flux_sets_3d or {}

        self._source_mode    = (source_mode or "off")
        self._source_list_3d = source_list_3d or []
        self._source_sets_3d = source_sets_3d or {}

        self.report_dir = Path("vtk_out"); ensure_dir(self.report_dir / "dummy.txt")

        self.DoEs: List[int] = []
        self.DoFs: List[int] = []
        self.d_E = np.zeros(0)

    # ------------------------------ Dirichlet(T) ------------------------------
    def Dirichlet_BC(self, load: Optional[str] = None):
        nodeT: Dict[int, float] = {}

        # list
        for (node_label, Tval) in (self._dirichlet_list_3d or []):
            r = int(node_label) - 1
            if 0 <= r < self.NoN:
                nodeT[r] = float(Tval)

        # sets
        if isinstance(self._dirichlet_sets_3d, dict):
            for set_name, cfg in self._dirichlet_sets_3d.items():
                if "nodes" not in cfg: continue
                Tval = cfg.get("T", None)
                if Tval is None: continue
                nodes = cfg["nodes"]
                if isinstance(nodes, str):
                    nlist = parse_nodes_txt_simple(nodes)
                else:
                    nlist = [int(v) for v in nodes]
                for n1 in sorted(set(nlist)):
                    r = _label_to_row_func(self.mesh)(n1)
                    if 0 <= r < self.NoN:
                        nodeT[r] = float(Tval)

        # finalize
        self.DoEs = sorted(nodeT.keys())
        self.d_E  = np.array([nodeT[d] for d in self.DoEs], float).reshape(-1,)
        self.DoF  = np.arange(self.NoN, dtype=int)
        self.DoFs = [i for i in self.DoF if i not in self.DoEs]
        print(f"[Heat/Dirichlet] DoEs={len(self.DoEs)}, DoFs={len(self.DoFs)}")

        # report
        lines = ["node_label,T"]
        row_to_label = _row_to_label_func(self.mesh)
        for r in self.DoEs:
            lines.append(f"{row_to_label(r)},{float(nodeT[r]):.16e}")
        (self.report_dir / "heat_dirichlet.txt").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------ Neumann (strict-list first) ------------------------------

    def assemble_boundary_heat_flux_2d(self, thickness: float = 1.0) -> np.ndarray:
        """2D boundary heat flux (Neumann) assembly.

        Expects strict-list entries such as:
            {"nodes":[...], "qn": value}
            {"nodes":[...], "flux": value}
            {"nodes":[...], "q": value}

        The nodes/seed_nodes field is interpreted as a collection of boundary
        nodes on which a uniform normal heat flux is applied. The contribution
        on each boundary edge is integrated as qn * L_edge and distributed
        equally to the two end nodes (∫ N_i ds = L/2 for linear edges).
        """
        Fq = np.zeros((self.NoN, 1))
        if self.Dimension != "2D":
            return Fq
        if (self._flux_mode or "off") == "off":
            return Fq
        if getattr(self, "boundary_heat_flux", "O") == "X":
            return Fq

        metas = build_boundary_edges_meta_2d(
            self.NL, self.conn, self.NoN, self.NPE,
            self.element_shape, getattr(self.mesh, "row2id", None)
        )

        use_strict_list = False
        if self._flux_mode in ("list", "both"):
            for cfg in (self._flux_list_3d or []):
                if ("seed_nodes" in cfg or "nodes" in cfg) and any(k in cfg for k in ("qn", "flux", "q")):
                    use_strict_list = True
                    break

        if not use_strict_list:
            print("[Heat/Flux/2D] no strict-list items; nothing assembled (legacy path skipped).")
            try:
                (self.report_dir / "heat_neumann.txt").write_text(
                    "type,entity,count,total_Q\nedge,edges,0,0\n",
                    encoding="utf-8"
                )
            except Exception:
                pass
            return Fq

        used_edges = 0
        skipped = 0
        total_Q = 0.0

        for i, cfg in enumerate(self._flux_list_3d or []):
            seeds = cfg.get("nodes", cfg.get("seed_nodes", None))
            if seeds is None:
                continue
            try:
                nodes1 = [int(v) for v in seeds]
            except Exception:
                print(f"[Heat/Neumann/2D/list#{i}] bad seeds: {seeds}")
                skipped += 1
                continue

            qn = None
            for k in ("qn", "flux", "q"):
                if k in cfg and cfg[k] is not None:
                    qn = float(cfg[k])
                    break
            if qn is None:
                print(f"[Heat/Neumann/2D/list#{i}] qn missing for seeds {nodes1}")
                skipped += 1
                continue

            nodes_set = set(nodes1)
            for key, meta in metas.items():
                n1, n2 = meta["nodes1"]
                if (n1 not in nodes_set) or (n2 not in nodes_set):
                    continue
                rows = meta["corners_rows"]
                L = float(meta["length"])
                if L <= 0.0:
                    continue

                share = qn * L * 0.5 * thickness
                for r in rows:
                    Fq[int(r), 0] += share
                used_edges += 1
                total_Q += qn * L * thickness

        print(f"[Heat/Neumann/2D] edges={used_edges}, totalQ={total_Q:.6g}, skipped={skipped}")
        try:
            (self.report_dir / "heat_neumann.txt").write_text(
                "type,entity,count,total_Q\nedge,edges,{:d},{:.16e}\n".format(used_edges, total_Q),
                encoding="utf-8"
            )
        except Exception:
            pass
        return Fq

    def assemble_boundary_heat_flux_3d(self) -> np.ndarray:
        Fq = np.zeros((self.NoN, 1))
        if self.Dimension != "3D":
            return Fq
        if (self._flux_mode or "off") == "off":
            return Fq

        # strict-list?
        use_strict_list = False
        if self._flux_mode in ("list","both"):
            for cfg in (self._flux_list_3d or []):
                if ("seed_nodes" in cfg or "nodes" in cfg) and any(k in cfg for k in ("qn","flux","q")):
                    use_strict_list = True; break

        if use_strict_list:
            label2row = _label_to_row_func(self.mesh)
            used = 0; skipped = 0; total_Q = 0.0
            for i, cfg in enumerate(self._flux_list_3d or []):
                seeds = cfg.get("seed_nodes", cfg.get("nodes", None))
                if seeds is None: continue
                try: nodes1 = [int(v) for v in seeds]
                except Exception:
                    print(f"[Heat/Neumann/list#{i}] bad seeds: {seeds}")
                    continue
                if len(nodes1) not in (3,4):
                    print(f"[Heat/Neumann/list#{i}] seeds must have 3 or 4 nodes: {nodes1}")
                    skipped += 1; continue
                qn = None
                for k in ("qn","flux","q"):
                    if k in cfg and cfg[k] is not None:
                        qn = float(cfg[k]); break
                if qn is None:
                    print(f"[Heat/Neumann/list#{i}] qn missing for seeds {nodes1}")
                    skipped += 1; continue

                rows = [label2row(n1) for n1 in nodes1]
                if any((r<0 or r>=self.NoN) for r in rows):
                    print(f"[Heat/Neumann/list#{i}] node label -> row out of range for {nodes1}")
                    skipped += 1; continue
                area, ctr, _ = _area_of_nodes(self.NL, rows)
                if area <= 0.0:
                    print(f"[Heat/Neumann/list#{i}] zero area for seeds {nodes1}")
                    skipped += 1; continue

                share = (area / len(rows)) * qn
                for r in rows: Fq[int(r),0] += share
                used += 1; total_Q += qn*area

            print(f"[Heat/Neumann/strict-list] faces={used}, totalQ={total_Q:.6g}, skipped={skipped}")
            (self.report_dir / "heat_neumann.txt").write_text(
                "type,entity,count,total_Q\nface,faces,{:d},{:.16e}\n".format(used, total_Q),
                encoding="utf-8"
            )
            return Fq

        # Legacy path skipped by design; strict-list is your requirement.
        print("[Heat/Flux] no strict-list items; nothing assembled (legacy path skipped).")
        (self.report_dir / "heat_neumann.txt").write_text("type,entity,count,total_Q\nface,faces,0,0\n", encoding="utf-8")
        return Fq

    # ------------------------------ Source (qv) ------------------------------

    def Heat_source(self, *args, **kwargs) -> np.ndarray:
        """Volumetric heat source assembler (2D or 3D).

        Expects source entries such as:
            {"elems": [e1, e2, ...], "qv": value}
            {"elems": [e1, e2, ...], "source": value}
        where qv/source is a uniform volumetric heat generation term.
        """
        Fv = np.zeros((self.NoN, 1))
        if (self._source_mode or "off") == "off":
            return Fv

        NL = np.asarray(self.NL, float)

        def _collect(container: Iterable[Tuple[str, dict]]):
            out = []
            for name, cfg in container:
                qv = float(cfg.get("source", cfg.get("qv", 0.0)))
                if "elems" in cfg:
                    for eidx in cfg["elems"]:
                        out.append((int(eidx), qv))
                elif "all" in cfg and bool(cfg.get("all", False)):
                    for eidx in range(len(self.conn)):
                        out.append((int(eidx), qv))
            return out

        targets: list[tuple[int, float]] = []
        if self._source_mode in ("list", "both"):
            targets += _collect([(f"list#{i}", cfg) for i, cfg in enumerate(self._source_list_3d)])
        if self._source_mode in ("set", "both"):
            targets += _collect(list((self._source_sets_3d or {}).items()))

        if not targets:
            return Fv

        if self.Dimension == "3D":
            gp3d, W3d, _ = Gauss_Point(None, "Linear", "Hex", "full").calculate_3D_gauss_points_weight()
            for (eidx, qv) in targets:
                c_rows = _conn_rows(self.mesh, self.conn, eidx, self.NoN)
                xIe = NL[c_rows, :3]
                for (xi, eta, zeta), w in zip(gp3d, W3d):
                    Nvals = self.sf.shape((xi, eta, zeta), self.NPE, "3D")
                    dN    = self.sf.gradshape((xi, eta, zeta), self.NPE, "3D")
                    J = np.vstack([dN[0, :] @ xIe, dN[1, :] @ xIe, dN[2, :] @ xIe]).T
                    dV = abs(np.linalg.det(J)) * w
                    f_e = (dV * qv) * Nvals.reshape(-1, 1)
                    for a, r in enumerate(c_rows):
                        Fv[int(r), 0] += f_e[a, 0]
        elif self.Dimension == "2D":
            gp2d, W2d, _ = Gauss_Point(None, "Linear", self.element_shape, "full").calculate_2D_gauss_points_weight()
            for (eidx, qv) in targets:
                c_rows = _conn_rows(self.mesh, self.conn, eidx, self.NoN)
                xIe = NL[c_rows, :2]
                for (xi, eta), w in zip(gp2d, W2d):
                    Nvals = self.sf.shape((xi, eta), self.NPE, "2D")
                    dN    = self.sf.gradshape((xi, eta), self.NPE, "2D")
                    J = np.array([
                        [dN[0, :] @ xIe[:, 0], dN[0, :] @ xIe[:, 1]],
                        [dN[1, :] @ xIe[:, 0], dN[1, :] @ xIe[:, 1]],
                    ])
                    dA = abs(np.linalg.det(J)) * w
                    f_e = (dA * qv) * Nvals.reshape(-1, 1)
                    for a, r in enumerate(c_rows):
                        Fv[int(r), 0] += f_e[a, 0]
        else:
            # Unknown Dimension; do nothing
            return Fv

        print(f"[Heat/Source] elements={len(targets)}")
        return Fv

    # -------- Compatibility wrapper to match old call sites --------
    def assemble_boundary_flux(self, *args, **kwargs) -> np.ndarray:
        """Dispatch to 2D or 3D Neumann assembler depending on Dimension."""
        if self.Dimension == "3D":
            return self.assemble_boundary_heat_flux_3d()
        else:
            return self.assemble_boundary_heat_flux_2d()


# ============================================================================
# Diffusion (same structure as Heat; different naming)
# ============================================================================

class Diffusion_BC:
    """
    Diffusion BC/loads
    - Dirichlet(C): list / set
    - Neumann(jn): strict-list first
    - Source(sv): elements
    """
    def __init__(self, system: str, mesh,
                 element_shape: str, element_order: str,
                 load: str, n_gp: Optional[int],
                 analytical_conditions: str, integration: str,
                 boundary_mass_concentration: str, body_mass_concentration: str,
                 Dimension: str,
                 dirichlet_mode: str = "off",
                 dirichlet_list_3d: Optional[list] = None,
                 dirichlet_sets_3d: Optional[dict] = None,
                 flux_mode: str = "off",
                 flux_list_3d: Optional[list] = None,
                 flux_sets_3d: Optional[dict] = None,
                 source_mode: str = "off",
                 source_list_3d: Optional[list] = None,
                 source_sets_3d: Optional[dict] = None):
        self.system = system
        self.mesh = mesh
        self.NL   = mesh.NL
        self.conn = mesh.conn
        self.NoN  = mesh.NoN
        self.NPE  = mesh.NPE
        self.element_shape = element_shape
        self.element_order = element_order
        self.integration   = integration
        self.analytical_conditions = analytical_conditions
        self.Dimension = Dimension

        self.sf = Shape_Function(analytical_conditions, element_order, element_shape, integration)

        self.F = np.zeros((self.NoN, 1))
        self._dirichlet_mode = dirichlet_mode
        self._dirichlet_list_3d = dirichlet_list_3d or []
        self._dirichlet_sets_3d = dirichlet_sets_3d or {}
        self._flux_mode = flux_mode
        self._flux_list_3d = flux_list_3d or []
        self._flux_sets_3d = flux_sets_3d or {}
        self._source_mode = source_mode
        self._source_list_3d = source_list_3d or []
        self._source_sets_3d = source_sets_3d or {}

        self.DoEs=[]; self.DoFs=[]; self.d_E=np.zeros(0)

    # -- Dirichlet(C) --
    def Dirichlet_BC(self, load: Optional[str] = None):
        nodeC: Dict[int, float] = {}

        for (node_label, Cval) in (self._dirichlet_list_3d or []):
            r = int(node_label) - 1
            if 0 <= r < self.NoN:
                nodeC[r] = float(Cval)

        if isinstance(self._dirichlet_sets_3d, dict):
            for set_name, cfg in self._dirichlet_sets_3d.items():
                if "nodes" not in cfg: continue
                Cval = cfg.get("C", None)
                if Cval is None: continue
                nodes = cfg["nodes"]
                if isinstance(nodes, str):
                    nlist = parse_nodes_txt_simple(nodes)
                else:
                    nlist = [int(v) for v in nodes]
                for n1 in sorted(set(nlist)):
                    r = _label_to_row_func(self.mesh)(n1)
                    if 0 <= r < self.NoN:
                        nodeC[r] = float(Cval)

        self.DoEs = sorted(nodeC.keys())
        self.d_E = np.array([nodeC[d] for d in self.DoEs], float).reshape(-1,)
        self.DoF = np.arange(self.NoN, dtype=int)
        self.DoFs = [i for i in self.DoF if i not in self.DoEs]
        print(f"[Diffusion/Dirichlet] DoEs={len(self.DoEs)}, DoFs={len(self.DoFs)}")

        lines = ["node_label,C"]
        row_to_label = _row_to_label_func(self.mesh)
        for r in self.DoEs: lines.append(f"{row_to_label(r)},{float(nodeC[r]):.16e}")
        (Path("vtk_out") / "diffusion_dirichlet.txt").write_text("\n".join(lines), encoding="utf-8")

    # -- Strict-list mass flux jn --

    def assemble_boundary_flux_2d(self) -> np.ndarray:
        """2D boundary mass flux (Neumann) assembly.

        Expects strict-list entries such as:
            {"nodes":[...], "jn": value}
            {"nodes":[...], "flux": value}
            {"nodes":[...], "q": value}
        where the nodes/seed_nodes field denotes boundary nodes with
        uniform normal mass flux jn.
        """
        Fj = np.zeros((self.NoN, 1))
        if self.Dimension != "2D":
            return Fj
        if (self._flux_mode or "off") == "off":
            return Fj

        metas = build_boundary_edges_meta_2d(
            self.NL, self.conn, self.NoN, self.NPE,
            self.element_shape, getattr(self.mesh, "row2id", None)
        )

        use_strict_list = False
        if self._flux_mode in ("list", "both"):
            for cfg in (self._flux_list_3d or []):
                if ("seed_nodes" in cfg or "nodes" in cfg) and any(k in cfg for k in ("jn", "flux", "q")):
                    use_strict_list = True
                    break

        if not use_strict_list:
            print("[Diffusion/Flux/2D] no strict-list items; nothing assembled (legacy path skipped).")
            try:
                (self.report_dir / "diffusion_neumann.txt").write_text(
                    "type,entity,count,total_J\nedge,edges,0,0\n",
                    encoding="utf-8"
                )
            except Exception:
                pass
            return Fj

        used_edges = 0
        skipped = 0
        total = 0.0

        for i, cfg in enumerate(self._flux_list_3d or []):
            seeds = cfg.get("seed_nodes", cfg.get("nodes", None))
            if seeds is None:
                continue
            try:
                nodes1 = [int(v) for v in seeds]
            except Exception:
                print(f"[Diffusion/Neumann/2D/list#{i}] bad seeds: {seeds}")
                skipped += 1
                continue

            jn = None
            for k in ("jn", "flux", "q"):
                if k in cfg and cfg[k] is not None:
                    jn = float(cfg[k])
                    break
            if jn is None:
                print(f"[Diffusion/Neumann/2D/list#{i}] jn missing for seeds {nodes1}")
                skipped += 1
                continue

            nodes_set = set(nodes1)
            for key, meta in metas.items():
                n1, n2 = meta["nodes1"]
                if (n1 not in nodes_set) or (n2 not in nodes_set):
                    continue
                rows = meta["corners_rows"]
                L = float(meta["length"])
                if L <= 0.0:
                    continue
                share = jn * L * 0.5
                for r in rows:
                    Fj[int(r), 0] += share
                used_edges += 1
                total += jn * L

        print(f"[Diffusion/Neumann/2D] edges={used_edges}, totalJ={total:.6g}, skipped={skipped}")
        try:
            (self.report_dir / "diffusion_neumann.txt").write_text(
                "type,entity,count,total_J\nedge,edges,{:d},{:.16e}\n".format(used_edges, total),
                encoding="utf-8"
            )
        except Exception:
            pass
        return Fj

    def assemble_boundary_flux_3d(self) -> np.ndarray:
        Fj = np.zeros((self.NoN, 1))
        if self.Dimension != "3D": return Fj
        if (self._flux_mode or "off") == "off": return Fj

        use_strict_list = False
        if self._flux_mode in ("list","both"):
            for cfg in (self._flux_list_3d or []):
                if ("seed_nodes" in cfg or "nodes" in cfg) and any(k in cfg for k in ("jn","flux","q")):
                    use_strict_list = True; break

        if use_strict_list:
            label2row = _label_to_row_func(self.mesh)
            used = 0; skipped = 0; total = 0.0
            for i, cfg in enumerate(self._flux_list_3d or []):
                seeds = cfg.get("seed_nodes", cfg.get("nodes", None))
                if seeds is None: continue
                try: nodes1 = [int(v) for v in seeds]
                except Exception:
                    print(f"[Diffusion/Neumann/list#{i}] bad seeds: {seeds}")
                    continue
                if len(nodes1) not in (3,4):
                    print(f"[Diffusion/Neumann/list#{i}] seeds must have 3 or 4 nodes: {nodes1}")
                    skipped += 1; continue
                jn = None
                for k in ("jn","flux","q"):
                    if k in cfg and cfg[k] is not None:
                        jn = float(cfg[k]); break
                if jn is None:
                    print(f"[Diffusion/Neumann/list#{i}] jn missing for seeds {nodes1}")
                    skipped += 1; continue

                rows = [label2row(n1) for n1 in nodes1]
                if any((r<0 or r>=self.NoN) for r in rows):
                    print(f"[Diffusion/Neumann/list#{i}] node label -> row out of range for {nodes1}")
                    skipped += 1; continue
                area, ctr, _ = _area_of_nodes(self.NL, rows)
                if area <= 0.0:
                    print(f"[Diffusion/Neumann/list#{i}] zero area for seeds {nodes1}")
                    skipped += 1; continue

                share = (area / len(rows)) * jn
                for r in rows: Fj[int(r),0] += share
                used += 1; total += jn*area

            print(f"[Diffusion/Neumann/strict-list] faces={used}, total={total:.6g}, skipped={skipped}")
            return Fj

        print("[Diffusion/Flux] no strict-list items; nothing assembled (legacy path skipped).")
        return Fj

    # -- Volume source sv --

    def Body_source(self) -> np.ndarray:
        """Volumetric mass source assembler (2D or 3D) for diffusion."""
        Fv = np.zeros((self.NoN, 1))
        if (self._source_mode or "off") == "off":
            return Fv

        NL = np.asarray(self.NL, float)

        def _collect(container: Iterable[Tuple[str, dict]]):
            out = []
            for name, cfg in container:
                sv = float(cfg.get("source", cfg.get("sv", 0.0)))
                if "elems" in cfg:
                    for eidx in cfg["elems"]:
                        out.append((int(eidx), sv))
                elif "all" in cfg and bool(cfg.get("all", False)):
                    for eidx in range(len(self.conn)):
                        out.append((int(eidx), sv))
            return out

        targets: list[tuple[int, float]] = []
        if self._source_mode in ("list", "both"):
            targets += _collect([(f"list#{i}", cfg) for i, cfg in enumerate(self._source_list_3d)])
        if self._source_mode in ("set", "both"):
            targets += _collect(list((self._source_sets_3d or {}).items()))

        if not targets:
            return Fv

        if self.Dimension == "3D":
            gp3d, W3d, _ = Gauss_Point(None, "Linear", "Hex", "full").calculate_3D_gauss_points_weight()
            for (eidx, sv) in targets:
                c_rows = _conn_rows(self.mesh, self.conn, eidx, self.NoN)
                xIe = NL[c_rows, :3]
                for (xi, eta, zeta), w in zip(gp3d, W3d):
                    Nvals = self.sf.shape((xi, eta, zeta), self.NPE, "3D")
                    dN    = self.sf.gradshape((xi, eta, zeta), self.NPE, "3D")
                    J = np.vstack([dN[0, :] @ xIe, dN[1, :] @ xIe, dN[2, :] @ xIe]).T
                    dV = abs(np.linalg.det(J)) * w
                    f_e = (dV * sv) * Nvals.reshape(-1, 1)
                    for a, r in enumerate(c_rows):
                        Fv[int(r), 0] += f_e[a, 0]
        elif self.Dimension == "2D":
            gp2d, W2d, _ = Gauss_Point(None, "Linear", self.element_shape, "full").calculate_2D_gauss_points_weight()
            for (eidx, sv) in targets:
                c_rows = _conn_rows(self.mesh, self.conn, eidx, self.NoN)
                xIe = NL[c_rows, :2]
                for (xi, eta), w in zip(gp2d, W2d):
                    Nvals = self.sf.shape((xi, eta), self.NPE, "2D")
                    dN    = self.sf.gradshape((xi, eta), self.NPE, "2D")
                    J = np.array([
                        [dN[0, :] @ xIe[:, 0], dN[0, :] @ xIe[:, 1]],
                        [dN[1, :] @ xIe[:, 0], dN[1, :] @ xIe[:, 1]],
                    ])
                    dA = abs(np.linalg.det(J)) * w
                    f_e = (dA * sv) * Nvals.reshape(-1, 1)
                    for a, r in enumerate(c_rows):
                        Fv[int(r), 0] += f_e[a, 0]
        else:
            # Unknown Dimension; do nothing
            return Fv

        print(f"[Diffusion/Source] elements={len(targets)}")
        return Fv


    # -------- Compatibility wrappers --------
    def assemble_boundary_flux(self, *args, **kwargs) -> np.ndarray:
        """Dispatch to 2D or 3D Neumann assembler depending on Dimension."""
        if self.Dimension == "3D":
            return self.assemble_boundary_flux_3d()
        else:
            return self.assemble_boundary_flux_2d()

    def Body_source_compat(self, *args, **kwargs) -> np.ndarray:
        return self.Body_source()


# ============================================================================
# Partition/Solve
# ============================================================================

class Partition_Method:
    def __init__(self, K_matrix: np.ndarray,
                 system: str, Dimension: str,
                 element_shape: str, element_order: str,
                 analytical_conditions: str, integration: str,
                 load: str, n_gp: Optional[int], distributed_type: Optional[str],
                 boundary_force: Optional[str], body_force: Optional[str],
                 T0: Optional[float], T1: Optional[float], alpha: Optional[float],
                 E: Optional[float], nu: Optional[float],
                 k_scalar: Optional[float], D_scalar: Optional[float],
                 boundary_heat_flux: Optional[str], source_heat_flux: Optional[str],
                 boundary_mass_concentration: Optional[str], body_mass_concentration: Optional[str],
                 mesh: Mesh,
                 # ──(elasticity)────────
                 dirichlet_list_3d: Optional[list] = None,
                 dirichlet_sets_3d: Optional[dict] = None,
                 traction_mode: str = "off",
                 traction_list_3d: Optional[list] = None,
                 concentrated_force_list_3d: Optional[list] = None,
                 traction_sets_3d: Optional[dict] = None,
                 body_vec: Optional[tuple] = None,    # (bx,by) for 2D, (bx,by,bz) for 3D
                 thickness2d: float = 1.0,
                 # ──(heat)─────────────
                 heat_dirichlet_mode: str = "off",
                 heat_dirichlet_list_3d: Optional[list] = None,
                 heat_dirichlet_sets_3d: Optional[dict] = None,
                 heat_flux_mode: str = "off",
                 heat_flux_list_3d: Optional[list] = None,
                 heat_flux_sets_3d: Optional[dict] = None,
                 heat_source_mode: str = "off",
                 heat_source_list_3d: Optional[list] = None,
                 heat_source_sets_3d: Optional[dict] = None,
                 # ──(diffusion)────────
                 diff_dirichlet_mode: str = "off",
                 diff_dirichlet_list_3d: Optional[list] = None,
                 diff_dirichlet_sets_3d: Optional[dict] = None,
                 diff_flux_mode: str = "off",
                 diff_flux_list_3d: Optional[list] = None,
                 diff_flux_sets_3d: Optional[dict] = None,
                 diff_source_mode: str = "off",
                 diff_source_list_3d: Optional[list] = None,
                 diff_source_sets_3d: Optional[dict] = None,
                 ):

        # --- Body-force defaults from main.py (safe import) ---
        self._body_vec = None
        self._thickness2d = 1.0
        try:
            from main import setting as _cfg  # noqa: F401
        except Exception:
            _cfg = None
        
        # Dimension 판단 (가능하면 기존 값을 사용)
        dim_str = str(getattr(self, "Dimension", getattr(getattr(self, "bc", None), "Dimension", ""))).upper()
        if not dim_str:
            try:
                NL = getattr(getattr(self, "bc", None), "NL", None)
                dim_str = "2D" if (NL is not None and getattr(NL, "shape", (0,0))[1] == 2) else "3D"
            except Exception:
                dim_str = "3D"
        self.Dimension = dim_str
        
        if _cfg is not None:
            if self.Dimension.startswith("2D"):
                self._body_vec = getattr(_cfg, "body_force_vec_2d", None)
                tk = getattr(_cfg, "thickness", 1.0)
                if tk is not None:
                    self._thickness2d = float(tk)
            else:
                self._body_vec = getattr(_cfg, "body_force_vec_3d", None)
        
        self.system = system
        self.Dimension = Dimension
        self.element_shape = element_shape
        self.element_order = element_order
        self.integration = integration
        self.analytical_conditions = analytical_conditions
        self.mesh = mesh
        self.sf = Shape_Function(analytical_conditions, element_order, element_shape, integration)

        self.NL = mesh.NL
        self.conn = mesh.conn
        self.NoN = mesh.NoN
        self.NPE = mesh.NPE
        self.K = K_matrix
        self._body_vec = body_vec
        self._thickness2d = float(thickness2d)

        if system == "linear elasticity":
            self.bc = Linear_elasticity_BC(
                system, mesh, element_shape, element_order,
                load, n_gp, analytical_conditions, integration, Dimension,
                dirichlet_list_3d=dirichlet_list_3d,
                dirichlet_sets_3d=dirichlet_sets_3d,
                traction_mode=traction_mode,
                traction_list_3d=traction_list_3d,
                traction_sets_3d=traction_sets_3d,
                concentrated_force_list_3d=concentrated_force_list_3d
            )
        elif system == "heat transfer":
            self.bc = Heat_transfer_BC(
                system, mesh, element_shape, element_order,
                load, n_gp, analytical_conditions, integration,
                boundary_heat_flux, source_heat_flux, Dimension,
                dirichlet_mode = heat_dirichlet_mode,
                dirichlet_list_3d = heat_dirichlet_list_3d,
                dirichlet_sets_3d = heat_dirichlet_sets_3d,
                flux_mode = heat_flux_mode,
                flux_list_3d = heat_flux_list_3d,
                flux_sets_3d = heat_flux_sets_3d,
                source_mode = heat_source_mode,
                source_list_3d = heat_source_list_3d,
                source_sets_3d = heat_source_sets_3d,
            )
            self.bc.Dirichlet_BC()

        elif system == "diffusion":
            self.bc = Diffusion_BC(
                system, mesh, element_shape, element_order,
                load, n_gp, analytical_conditions, integration,
                boundary_mass_concentration, body_mass_concentration, Dimension,
                dirichlet_mode = diff_dirichlet_mode,
                dirichlet_list_3d = diff_dirichlet_list_3d,
                dirichlet_sets_3d = diff_dirichlet_sets_3d,
                flux_mode = diff_flux_mode,
                flux_list_3d = diff_flux_list_3d,
                flux_sets_3d = diff_flux_sets_3d,
                source_mode = diff_source_mode,
                source_list_3d = diff_source_list_3d,
                source_sets_3d = diff_source_sets_3d,
            )
            self.bc.Dirichlet_BC()

        else:
            raise ValueError("Unsupported system")

        self.DoEs = self.bc.DoEs
        self.DoFs = self.bc.DoFs
        self.d_E  = self.bc.d_E

        self.K_F    = K_matrix[np.ix_(self.DoFs, self.DoFs)]
        self.K_EF   = K_matrix[np.ix_(self.DoEs, self.DoFs)]
        self.K_EF_T = K_matrix[np.ix_(self.DoFs, self.DoEs)]
        self.K_E    = K_matrix[np.ix_(self.DoEs, self.DoEs)]

        # ---------- RHS assemble & solve ----------
        if Dimension == "3D":
            if system == "linear elasticity":
                mat = Material_Property(system, Dimension)
                D3 = mat.Hookean_matrix_3D(E, nu, analytical_conditions)

                # Thermal load optional
                delta_T = (T1 - T0) if (T1 is not None and T0 is not None) else 0.0
                F_thermal = np.zeros((3*self.NoN, 1))
                if abs(delta_T) > 0 and alpha is not None:
                    for c in self.conn:
                        c_arr = np.asarray(c, dtype=int)
                        xIe = self.NL[c_arr, :]
                        B = np.zeros((6, 3*self.NPE))
                        f_e = np.zeros((3*self.NPE, 1))
                        for gp, w in zip(self.sf.gp, self.sf.W):
                            dN = self.sf.gradshape(gp, self.NPE, "3D")
                            J  = dN @ xIe
                            invJ = np.linalg.inv(J)
                            gN = invJ @ dN
                            B[0, 0::3] = gN[0,:]
                            B[1, 1::3] = gN[1,:]
                            B[2, 2::3] = gN[2,:]
                            B[3, 0::3] = gN[1,:]; B[3, 1::3] = gN[0,:]
                            B[4, 1::3] = gN[2,:]; B[4, 2::3] = gN[1,:]
                            B[5, 2::3] = gN[0,:]; B[5, 0::3] = gN[2,:]
                            e0 = np.array([[1],[1],[1],[0],[0],[0]])
                            sigma_th = D3 @ (alpha*delta_T*e0)
                            f_e += B.T @ sigma_th * np.linalg.det(J) * w
                        for a, gna in enumerate(c_arr):
                            F_thermal[3*gna+0,0] += f_e[3*a+0,0]
                            F_thermal[3*gna+1,0] += f_e[3*a+1,0]
                            F_thermal[3*gna+2,0] += f_e[3*a+2,0]

                F_neu = self.bc.assemble_surface_tractions_3d()
                F_conc = self.bc.assemble_concentrated_forces(load)
                F = F_neu + F_thermal
                
                if hasattr(self, "_body_vec") and (self._body_vec is not None):
                    F += self.bc.assemble_body_force(self._body_vec)

                self.F_E = F[self.DoEs].flatten()
                self.F_F = F[self.DoFs].flatten()
                self.r   = np.zeros((3*self.NoN,))
                self.r_F = np.zeros(len(self.DoFs))
                self.d_F = np.linalg.solve(self.K_F, (self.F_F - (self.K_EF_T @ self.d_E)))

                self.d = np.zeros(3*self.NoN)
                self.d[self.DoEs] = self.d_E
                self.d[self.DoFs] = self.d_F

            elif system == "heat transfer":
                F = self.bc.assemble_boundary_flux(load, distributed_type) + self.bc.Heat_source(load, distributed_type)
                self.F_E = F[self.DoEs].flatten()
                self.F_F = F[self.DoFs].flatten()
                self.r   = np.zeros((self.NoN,))
                self.r_F = np.zeros(len(self.DoFs))
                self.d_F = np.linalg.solve(self.K_F, (self.F_F - (self.K_EF_T @ self.d_E)))

                self.d = np.zeros(self.NoN)
                self.d[self.DoEs] = self.d_E
                self.d[self.DoFs] = self.d_F

            elif system == "diffusion":
                F = self.bc.assemble_boundary_flux(load, distributed_type) + self.bc.Body_source()
                self.F_E = F[self.DoEs].flatten()
                self.F_F = F[self.DoFs].flatten()
                self.r   = np.zeros((self.NoN,))
                self.r_F = np.zeros(len(self.DoFs))
                self.d_F = np.linalg.solve(self.K_F, (self.F_F - (self.K_EF_T @ self.d_E)))

                self.d = np.zeros(self.NoN)
                self.d[self.DoEs] = self.d_E
                self.d[self.DoFs] = self.d_F

        elif Dimension == "2D":
            if system == "linear elasticity":
                F = np.zeros((2*self.NoN,1))

                # Neumann (edge traction / pressure)
                thickness2d = getattr(self, "_thickness2d", 1.0)
                F_neu = self.bc.assemble_edge_tractions_2d(thickness=thickness2d)

                # Concentrated nodal forces
                F_conc = self.bc.assemble_concentrated_forces(load)

                F = F + F_neu + F_conc

                # Body force (if specified)
                if hasattr(self, "_body_vec") and (self._body_vec is not None):
                    F += self.bc.assemble_body_force(self._body_vec, thickness=thickness2d)
                    
                print(F)

                self.F_E = F[self.DoEs].flatten()
                self.F_F = F[self.DoFs].flatten()
                self.r_F = np.zeros(len(self.DoFs))
                self.d_F = np.linalg.solve(self.K_F, (self.F_F - (self.K_EF_T @ self.d_E)))

                # 전체 변위 벡터 저장(옵션)
                self.d = np.zeros(2*self.NoN)
                self.d[self.DoEs] = self.d_E
                self.d[self.DoFs] = self.d_F
            elif system == "heat transfer":
                F = self.bc.assemble_boundary_flux(load, distributed_type) + self.bc.Heat_source(load, distributed_type)
                self.F_E = F[self.DoEs].flatten()
                self.F_F = F[self.DoFs].flatten()
                self.r_F = np.zeros(len(self.DoFs))
                self.d_F = np.linalg.solve(self.K_F, (self.F_F - (self.K_EF_T @ self.d_E)))
            elif system == "diffusion":
                F = self.bc.assemble_boundary_flux(load, distributed_type) + self.bc.Body_source()
                self.F_E = F[self.DoEs].flatten()
                self.F_F = F[self.DoFs].flatten()
                self.r_F = np.zeros(len(self.DoFs))
                self.d_F = np.linalg.solve(self.K_F, (self.F_F - (self.K_EF_T @ self.d_E)))

__all__ = [
    "OneD_gauss_quadrature",
    "tet_local_faces", "hex_local_faces", "element_face_topology",
    "Linear_elasticity_BC",
    "Heat_transfer_BC",
    "Diffusion_BC",
    "Partition_Method",
]
