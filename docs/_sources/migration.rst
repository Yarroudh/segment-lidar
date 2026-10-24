Migrating from the earlier documentation
========================================

Version 0.3 replaced the old SAM 1 and segment-geospatial backend with SAM 3.
Scripts from the old site therefore need API updates:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Earlier documentation
     - Current 0.3 series
   * - ``SamLidar(ckpt_path="sam_vit_h_...pth")``
     - ``SamLidar()`` plus approved access to ``facebook/sam3`` and
       ``HF_TOKEN`` or ``hf_token``. No manual checkpoint path.
   * - ``algorithm``, ``model_type``, ``sam_kwargs``
     - Removed; SAM 3 is the inference backend.
   * - ``model.mask.*``
     - Use ``model.automatic_masks.*`` for automatic masks or
       ``model.text_prompt.*`` for prompted inference.
   * - ``PinholeView(intrinsic=K, rotation=R, translation=T)``
     - Set ``intrinsics``, ``rotation``, and ``translation`` on ``SamLidar``;
       pass ``view=PinholeView()`` to ``segment``.
   * - ``PinholeView(interactive=True)``
     - Interactive camera capture is unavailable. Supply matrices; see
       :doc:`pinhole`.
   * - ``labels, *_ = model.segment(...)``
     - The first result is now point ``segment_ids``. The second is the 2D
       label raster. Pinhole results also include ``K`` and ``pose``.

Ground filtering remains optional. The writer still adds ``segment_id`` to
LAS/LAZ, but a signed 32 bit field is now used so ``-1`` can represent
unlabeled points. Existing files with older field types should be handled
according to their stored values rather than assumed to use this convention.

The previous site showed an AHN-4 tutorial and SAM 1 checkpoints. Those
examples are historical. :doc:`tutorial` and :doc:`pinhole` are the current
workflows; the old HTML URLs continue to point to updated pages.
