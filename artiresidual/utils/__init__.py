"""Utilities — geometry primitives, visualization helpers, part tracker.

The part tracker is the spec §3.0 SAM2-based pose pipeline that feeds
Module 06. It is gated by ``perception.use_gt_part_pose`` in
``configs/base.yaml``: during sim training the GT joint state is used and
the tracker is never loaded, so we ``lazy``-import it.
"""
