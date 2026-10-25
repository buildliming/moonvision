///|
/// Equality for packed images.
pub extend Image with Eq::{equal, not_equal}

///|
/// Debug representation for packed images.
pub extend Image with @debug.Debug::{to_repr}

///|
/// Equality for labeled boxes.
pub extend BBox with Eq::{equal, not_equal}

///|
/// Debug representation for labeled boxes.
pub extend BBox with @debug.Debug::{to_repr}

///|
/// JSON representation for labeled boxes.
pub extend BBox with ToJson::{to_json}

///|
/// Equality for annotated samples.
pub extend Sample with Eq::{equal, not_equal}

///|
/// Debug representation for annotated samples.
pub extend Sample with @debug.Debug::{to_repr}

///|
/// Equality for transform specifications.
pub extend Transform with Eq::{equal, not_equal}

///|
/// Debug representation for transform specifications.
pub extend Transform with @debug.Debug::{to_repr}

///|
/// Equality for probabilistic steps.
pub extend Step with Eq::{equal, not_equal}

///|
/// Debug representation for probabilistic steps.
pub extend Step with @debug.Debug::{to_repr}

///|
/// Equality for replay records.
pub extend Replay with Eq::{equal, not_equal}

///|
/// Debug representation for replay records.
pub extend Replay with @debug.Debug::{to_repr}

///|
/// Equality for augmentation results.
pub extend Augmented with Eq::{equal, not_equal}

///|
/// Debug representation for augmentation results.
pub extend Augmented with @debug.Debug::{to_repr}

///|
/// Equality for augmentation errors.
pub extend AugError with Eq::{equal, not_equal}

///|
/// Debug representation for augmentation errors.
pub extend AugError with @debug.Debug::{to_repr}

///|
pub extend Padding with Eq::{equal, not_equal}

///|
pub extend Padding with @debug.Debug::{to_repr}

///|
pub extend LetterboxOptions with Eq::{equal, not_equal}

///|
pub extend LetterboxOptions with @debug.Debug::{to_repr}

///|
pub extend ResizedCropOptions with Eq::{equal, not_equal}

///|
pub extend ResizedCropOptions with @debug.Debug::{to_repr}

///|
pub extend Keypoint with Eq::{equal, not_equal}

///|
pub extend Keypoint with @debug.Debug::{to_repr}

///|
pub extend Keypoint with ToJson::{to_json}

///|
pub extend ExtendedSample with Eq::{equal, not_equal}

///|
pub extend ExtendedSample with @debug.Debug::{to_repr}

///|
pub extend ExtendedAugmented with Eq::{equal, not_equal}

///|
pub extend ExtendedAugmented with @debug.Debug::{to_repr}

///|
pub extend BatchItem with Eq::{equal, not_equal}

///|
pub extend BatchItem with @debug.Debug::{to_repr}

///|
pub extend BatchResult with Eq::{equal, not_equal}

///|
pub extend BatchResult with @debug.Debug::{to_repr}

///|
pub extend BatchReplay with Eq::{equal, not_equal}

///|
pub extend BatchReplay with @debug.Debug::{to_repr}
