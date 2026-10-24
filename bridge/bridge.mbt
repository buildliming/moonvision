///|
/// Process an annotation-aware augmentation or replay JSON request.
pub fn augment_json(request : String) -> String {
  @aug.process_json(request)
}

///|
/// Process a differentiable sampling or affine registration JSON request.
pub fn warp_json(request : String) -> String {
  @warp.process_json(request)
}
