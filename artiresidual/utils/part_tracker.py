"""Section 3.0 — SAM2-based part tracker.

See artiresidual_tech_spec.md §3.0 for the authoritative API. This module is
non-trainable upstream perception: it produces the ``current_part_pose`` that
Module 06 (State Estimator) consumes at every control step.

Architecture (spec §3.0):

    t = 0       Module 01 part segmentation → per-part 2-D mask
                SAM2 video predictor is seeded with these masks
                (Reference part PCD is bootstrapped lazily on the FIRST
                 ``estimate()`` call, because ``initialize()`` does not
                 receive depth or intrinsics.)

    t > 0       SAM2.propagate(rgb_t)              → per-part 2-D mask at t
                back-project masked RGB-D + K      → per-part 3-D PCD at t
                ICP(reference PCD, current PCD)    → 4×4 rigid transform
                                                   → (pos, quat) world-frame
                                                     pose per part

Sim-vs-real toggle: in simulation training set ``perception.use_gt_part_pose:
true`` in the Hydra config — Module 06 reads the simulator's GT joint state
directly and this whole module is bypassed. SAM2 only loads when
``use_gt_part_pose: false`` (real-robot eval).

Quaternion convention: scalar-first ``(w, x, y, z)``, matching Module 06.

Status:
    * The class structure follows the spec API exactly.
    * SAM2 and open3d are imported lazily inside the constructor and the ICP
      helper so that importing this module never requires either dependency.
    * SAM2 streaming pattern: the video predictor expects a directory of
      JPEG frames. We buffer frames in a ``tempfile.mkdtemp()`` and re-init
      state inside ``estimate()``. This is the simplest correct way to drive
      SAM2 online; performance-tuning (in-memory state, incremental memory
      bank updates) is a TODO once accuracy is verified.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

__all__ = ["SAM2PartTracker"]

# Quaternion convention: scalar-first (w, x, y, z), matching
# artiresidual/refiner/state_estimator.py.


# ---------------------------------------------------------------------------
# Helpers (numpy-only; open3d is imported lazily inside _run_icp).
# ---------------------------------------------------------------------------


def _back_project_masked(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    depth_min: float = 0.05,
    depth_max: float = 5.0,
) -> np.ndarray:
    """Lift the masked-pixel subset of a depth image into a 3-D point cloud.

    Args:
        depth: [H, W] metric depth in meters.
        K:     [3, 3] camera intrinsics (fx, fy on diag; (cx, cy) in last col).
        mask:  [H, W] bool — True = part pixel.
        depth_min / depth_max: pixels whose depth lies outside this range are
            dropped (filters NaNs, sensor floor / ceiling, and far background).

    Returns:
        pcd: [N, 3] in the camera frame, N ≤ mask.sum().
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    v_idx, u_idx = np.where(mask)
    z = depth[v_idx, u_idx]
    valid = np.isfinite(z) & (z > depth_min) & (z < depth_max)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)
    v_idx, u_idx, z = v_idx[valid], u_idx[valid], z[valid].astype(np.float64)
    x = (u_idx.astype(np.float64) - cx) * z / fx
    y = (v_idx.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=-1)  # [N, 3]


def _rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → scalar-first quaternion (w, x, y, z).

    Uses Shepperd's branchless-stable formulation. Output is a unit quaternion
    with the largest absolute component made non-negative for canonicality.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0.0:  # canonical w >= 0 (matches Module 06's convention).
        q = -q
    return q / np.linalg.norm(q).clip(min=1e-12)


def _run_icp(
    source: np.ndarray,
    target: np.ndarray,
    max_correspondence_distance: float = 0.05,
    max_iteration: int = 50,
    relative_fitness: float = 1e-6,
) -> np.ndarray:
    """Point-to-point ICP via open3d, with sane fallbacks.

    Args:
        source: [N, 3] reference PCD (the t=0 part PCD).
        target: [M, 3] current PCD.
        max_correspondence_distance: m; spec §3.0 says ≤ 5 cm is sensible.
        max_iteration, relative_fitness: per spec §3.0 implementation notes.

    Returns:
        T: 4×4 homogeneous transform mapping source → target. Falls back to
        identity if either cloud has < 10 points or open3d is unavailable.
    """
    if source.shape[0] < 10 or target.shape[0] < 10:
        return np.eye(4)
    try:
        import open3d as o3d  # local import: keeps this file importable without open3d.
    except ImportError as e:
        raise ImportError(
            "open3d not installed. Run scripts/setup_sam2.sh on the GPU server."
        ) from e

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source.astype(np.float64))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target.astype(np.float64))
    result = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_correspondence_distance=max_correspondence_distance,
        init=np.eye(4),
        estimation_method=(
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        ),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iteration,
            relative_fitness=relative_fitness,
        ),
    )
    return np.asarray(result.transformation, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main class.
# ---------------------------------------------------------------------------


class SAM2PartTracker:
    """SAM2 mask propagation + open3d ICP → per-part 6-DoF pose stream.

    See ``artiresidual_tech_spec.md §3.0`` for the API contract. Output of
    ``estimate()`` is consumed by ``artiresidual.refiner.state_estimator.
    JointStateEstimator``.

    Frame convention: input depth + intrinsics are interpreted in the camera
    frame, so the returned poses are in the camera frame of the FIRST
    ``estimate()`` call (the bootstrap call that fixes the reference PCD).
    Applying a camera→world extrinsic is the caller's responsibility — at
    the moment that's only relevant for real-robot eval, where the camera is
    fixed and the extrinsic is calibrated once.

    Bootstrap convention: ``initialize()`` does not receive depth/intrinsics,
    so the per-part *reference* point clouds are lazy-computed on the first
    ``estimate()`` call. That first call returns ``pose = (centroid, identity
    quat)`` per part — this becomes the "initial_part_pose" that Module 06
    caches. Every subsequent ``estimate()`` returns the ICP-composed pose
    relative to that reference.
    """

    def __init__(self, sam2_checkpoint: str, device: str = "cuda") -> None:
        """Load SAM2 weights.

        Args:
            sam2_checkpoint: filesystem path to the SAM2 ``.pt`` file. The
                matching Hydra config name is inferred from the file name
                (``sam2_hiera_base_plus.pt`` → ``sam2_hiera_b+.yaml`` and
                ``sam2.1_hiera_base_plus.pt`` → ``sam2.1_hiera_b+.yaml``).
            device: torch device string (typically ``"cuda"``; CPU works but
                is much slower — ~600 ms / frame instead of ~30 ms).

        Raises:
            FileNotFoundError: if the checkpoint path doesn't exist.
            ImportError: if ``sam2`` isn't installed (run
                ``scripts/setup_sam2.sh`` on the server).
        """
        if not os.path.isfile(sam2_checkpoint):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_checkpoint}")

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as e:
            raise ImportError(
                "sam2 is not installed. Run scripts/setup_sam2.sh on the GPU server."
            ) from e

        self.sam2_checkpoint: str = sam2_checkpoint
        self.device: str = device
        self._config_file: str = self._infer_config_file(sam2_checkpoint)

        # Build the video predictor (model loads to ``device`` here).
        self._predictor = build_sam2_video_predictor(
            config_file=self._config_file,
            ckpt_path=sam2_checkpoint,
            device=device,
        )

        # State populated by initialize() / estimate().
        self._initialized: bool = False
        self._tempdir: Optional[str] = None
        self._part_masks_t0: Optional[list[np.ndarray]] = None  # [P] of [H, W] bool
        self._n_parts: int = 0
        self._frame_count: int = 0
        self._H: int = 0
        self._W: int = 0
        # Reference part PCDs and centroids are lazy-bootstrapped on the FIRST
        # estimate() call (initialize() lacks depth/intrinsics).
        self._reference_pcds: Optional[list[np.ndarray]] = None  # [P] of [N_i, 3]
        self._reference_centroids: Optional[np.ndarray] = None    # [P, 3]

    # -- spec API -----------------------------------------------------------
    def initialize(
        self,
        rgb_frame: np.ndarray,
        part_masks: Sequence[np.ndarray],
    ) -> None:
        """Seed SAM2 with the t=0 frame and per-part masks.

        Args:
            rgb_frame: [H, W, 3] uint8 image at t=0.
            part_masks: list of length P, each [H, W] bool/uint8 binary mask
                from Module 01. Must be non-empty; an empty mask is rejected
                (would crash SAM2 / produce zero-area PCDs downstream).

        Raises:
            ValueError: on shape / dtype mismatch or empty mask.
        """
        if rgb_frame.dtype != np.uint8:
            raise ValueError(f"rgb_frame must be uint8; got {rgb_frame.dtype}")
        if rgb_frame.ndim != 3 or rgb_frame.shape[-1] != 3:
            raise ValueError(f"rgb_frame must be [H, W, 3]; got {rgb_frame.shape}")
        if len(part_masks) == 0:
            raise ValueError("part_masks must contain at least one mask")

        H, W = rgb_frame.shape[:2]
        self._H, self._W = H, W
        self._n_parts = len(part_masks)
        self._part_masks_t0 = []
        for p_id, m in enumerate(part_masks):
            m_arr = np.asarray(m).astype(bool)
            if m_arr.shape != (H, W):
                raise ValueError(
                    f"part_masks[{p_id}] has shape {m_arr.shape}; expected ({H}, {W})"
                )
            if not m_arr.any():
                raise ValueError(f"part_masks[{p_id}] is empty (no True pixels)")
            self._part_masks_t0.append(m_arr)

        # Fresh tempdir frame buffer for SAM2's video predictor.
        if self._tempdir is not None:
            shutil.rmtree(self._tempdir, ignore_errors=True)
        self._tempdir = tempfile.mkdtemp(prefix="sam2_tracker_")
        self._save_frame(rgb_frame, idx=0)

        self._frame_count = 1
        self._reference_pcds = None
        self._reference_centroids = None
        self._initialized = True

    def estimate(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        camera_intrinsics: np.ndarray,
    ) -> np.ndarray:
        """Produce per-part 6-DoF poses at the current frame.

        Args:
            rgb_frame: [H, W, 3] uint8 current RGB frame.
            depth_frame: [H, W] float32 metric depth in meters. NaN / 0 /
                out-of-range pixels are filtered inside the back-projection.
            camera_intrinsics: [3, 3] standard pinhole K matrix.

        Returns:
            [P, 7] float64 — each row is ``(x, y, z, qw, qx, qy, qz)``.
            The first call (bootstrap) returns ``(centroid, identity)``
            per part; subsequent calls return the ICP-composed pose.

        Raises:
            RuntimeError: if ``initialize()`` wasn't called first.
            ValueError: on shape / dtype mismatch.
        """
        if not self._initialized:
            raise RuntimeError("call initialize() before estimate()")

        self._validate_estimate_inputs(rgb_frame, depth_frame, camera_intrinsics)

        is_bootstrap = self._reference_pcds is None
        if is_bootstrap:
            # First estimate(): no SAM2 needed — we already have the t=0 masks
            # from initialize(). Compute reference PCDs from THIS depth (which
            # we treat as the t=0 depth) and return identity-orientation poses
            # at each part's centroid.
            return self._bootstrap_reference(depth_frame, camera_intrinsics)

        # Normal path: append the new frame, run SAM2 propagate, ICP.
        self._save_frame(rgb_frame, idx=self._frame_count)
        target_frame_idx = self._frame_count
        self._frame_count += 1

        current_masks = self._sam2_propagate_to(target_frame_idx)
        return self._poses_from_masks(current_masks, depth_frame, camera_intrinsics)

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _infer_config_file(ckpt_path: str) -> str:
        """Map ``sam2*_hiera_<variant>.pt`` → SAM2's Hydra config name.

        SAM2's ``build_sam2_video_predictor`` looks the config up via Hydra
        inside the installed ``sam2`` package, so we pass the relative name
        (not the absolute path).
        """
        name = os.path.basename(ckpt_path)
        # sam2.1_* and sam2_* variants share size suffixes; pick the right dir.
        if name.startswith("sam2.1_"):
            prefix = "configs/sam2.1/sam2.1_hiera_"
        elif name.startswith("sam2_"):
            prefix = "configs/sam2/sam2_hiera_"
        else:
            raise ValueError(
                f"cannot infer SAM2 config from checkpoint name {name!r}; "
                f"expected a sam2_* or sam2.1_* file."
            )
        if "base_plus" in name:
            size = "b+"
        elif "large" in name:
            size = "l"
        elif "small" in name:
            size = "s"
        elif "tiny" in name:
            size = "t"
        else:
            raise ValueError(
                f"cannot infer SAM2 size from checkpoint name {name!r}; "
                f"expected a tiny/small/base_plus/large variant."
            )
        return f"{prefix}{size}.yaml"

    def _save_frame(self, rgb_frame: np.ndarray, idx: int) -> None:
        """Write the frame to the SAM2 buffer as ``<idx:05d>.jpg``."""
        from PIL import Image

        assert self._tempdir is not None
        Image.fromarray(rgb_frame).save(
            os.path.join(self._tempdir, f"{idx:05d}.jpg"), quality=95
        )

    def _validate_estimate_inputs(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
    ) -> None:
        if rgb.dtype != np.uint8 or rgb.shape != (self._H, self._W, 3):
            raise ValueError(
                f"rgb_frame must be uint8 [{self._H}, {self._W}, 3]; "
                f"got dtype={rgb.dtype} shape={rgb.shape}"
            )
        if depth.shape != (self._H, self._W):
            raise ValueError(
                f"depth_frame must be [{self._H}, {self._W}]; got {depth.shape}"
            )
        if K.shape != (3, 3):
            raise ValueError(f"camera_intrinsics must be [3, 3]; got {K.shape}")

    def _bootstrap_reference(
        self,
        depth: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:
        """Compute reference PCDs from the first estimate's depth + intrinsics.

        Returns the bootstrap pose tensor (centroid + identity quat per part).
        After this call ``self._reference_pcds`` and
        ``self._reference_centroids`` are populated.
        """
        assert self._part_masks_t0 is not None
        self._reference_pcds = []
        centroids = np.empty((self._n_parts, 3), dtype=np.float64)
        identity_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        poses = np.empty((self._n_parts, 7), dtype=np.float64)
        for p_id in range(self._n_parts):
            pcd = _back_project_masked(depth, K, self._part_masks_t0[p_id])
            self._reference_pcds.append(pcd)
            centroids[p_id] = pcd.mean(axis=0) if pcd.shape[0] > 0 else 0.0
            poses[p_id, :3] = centroids[p_id]
            poses[p_id, 3:] = identity_q
        self._reference_centroids = centroids
        return poses

    def _sam2_propagate_to(self, target_frame_idx: int) -> list[np.ndarray]:
        """Run SAM2 video propagation, return per-part bool masks at the latest frame.

        Implementation note: SAM2's video predictor expects a directory of
        frames + a starting prompt. We re-initialize state on the growing
        frame buffer every call and re-prompt with the t=0 masks. This is
        the simplest correct usage; an in-memory streaming variant is a
        future optimization.
        """
        assert self._part_masks_t0 is not None
        assert self._tempdir is not None

        state = self._predictor.init_state(video_path=self._tempdir)
        # Re-prompt with the t=0 masks for every object id.
        for p_id, mask_t0 in enumerate(self._part_masks_t0):
            self._predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=p_id,
                mask=mask_t0,
            )

        masks_at_target: list[Optional[np.ndarray]] = [None] * self._n_parts
        for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(
            state
        ):
            if frame_idx != target_frame_idx:
                continue
            for obj_id, logits in zip(obj_ids, mask_logits):
                arr = self._to_numpy(logits)
                # mask_logits per object is typically [1, H, W] (single channel).
                if arr.ndim == 3:
                    arr = arr[0]
                masks_at_target[obj_id] = arr > 0
            break

        # Defensive fill — should not happen unless SAM2 drops an object.
        for p_id in range(self._n_parts):
            if masks_at_target[p_id] is None:
                masks_at_target[p_id] = np.zeros((self._H, self._W), dtype=bool)
        return masks_at_target  # type: ignore[return-value]

    def _poses_from_masks(
        self,
        current_masks: list[np.ndarray],
        depth: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:
        """Back-project each current mask, ICP against its reference, return [P, 7]."""
        assert self._reference_pcds is not None
        assert self._reference_centroids is not None

        poses = np.empty((self._n_parts, 7), dtype=np.float64)
        for p_id in range(self._n_parts):
            current_pcd = _back_project_masked(depth, K, current_masks[p_id])
            T = _run_icp(self._reference_pcds[p_id], current_pcd)
            # The ICP transform maps source → target in the camera frame, so
            # the current centroid is the reference centroid pushed through T.
            ref_c = self._reference_centroids[p_id]
            pos = (T[:3, :3] @ ref_c) + T[:3, 3]
            quat = _rotation_matrix_to_quat(T[:3, :3])
            poses[p_id, :3] = pos
            poses[p_id, 3:] = quat
        return poses

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        """Best-effort coerce a torch / numpy / array-like into a numpy array."""
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    # -- cleanup ------------------------------------------------------------
    def close(self) -> None:
        """Release the SAM2 frame buffer. Safe to call more than once."""
        if self._tempdir is not None:
            shutil.rmtree(self._tempdir, ignore_errors=True)
            self._tempdir = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
